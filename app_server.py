"""
DetectDump — FastAPI backend.
Serves the reference UI and exposes a REST API for the real CV pipeline.

Endpoints:
  GET  /                              → Serve the DetectDump HTML UI
  POST /api/upload                    → Upload a video file, returns {file_id, filename, size}
  POST /api/analyze                   → Start real analysis, returns {analysis_id}
  GET  /api/progress/{analysis_id}    → Poll progress (frame count, percentage, stage)
  GET  /api/results/{analysis_id}     → Get full results when complete
  GET  /api/evidence/{analysis_id}/{track_id}/{idx} → Serve evidence frame as JPEG
  GET  /api/video/{analysis_id}       → Serve annotated output video
  DELETE /api/cleanup/{analysis_id}   → Remove temp files for an analysis
"""

import base64
import cv2
import json
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from temporal_engine import DumpingEvent, State, TemporalEventEngine, Thresholds
from action_candidate_detector import ActionCandidateDetector, CandidateConfig
from vlm_verify import verify_dumping_event

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

app = FastAPI(title="DetectDump")

BASE_DIR = Path(__file__).resolve().parent
UI_DIR = BASE_DIR / "ui"
UPLOAD_DIR = Path(tempfile.gettempdir()) / "detectdump_uploads"
OUTPUT_DIR = Path(tempfile.gettempdir()) / "detectdump_output"
EVIDENCE_DIR = Path(tempfile.gettempdir()) / "detectdump_evidence"

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
EVIDENCE_DIR.mkdir(exist_ok=True)

# In-memory store for analysis sessions
analyses: dict[str, dict] = {}
lock = threading.Lock()

ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
MAX_FILE_SIZE = 200 * 1024 * 1024  # 200MB


# ---------------------------------------------------------------------------
# HTML serving
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    html_path = UI_DIR / "detectdump.html"
    if not html_path.exists():
        raise HTTPException(500, "UI file not found")
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# File upload
# ---------------------------------------------------------------------------

@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    # Validate extension
    ext = Path(file.filename or "video.mp4").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}")

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large. Maximum: {MAX_FILE_SIZE // (1024*1024)}MB")

    file_id = str(uuid.uuid4())[:8]
    save_path = UPLOAD_DIR / f"{file_id}{ext}"
    save_path.write_bytes(content)

    return {
        "file_id": file_id,
        "filename": file.filename,
        "size": len(content),
        "ext": ext,
    }


# ---------------------------------------------------------------------------
# Analysis — runs real CV pipeline in background thread
# ---------------------------------------------------------------------------

@app.post("/api/analyze")
async def start_analysis(payload: dict):
    file_id = payload.get("file_id")
    if not file_id:
        raise HTTPException(400, "file_id required")

    # Find uploaded file
    upload_match = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not upload_match:
        raise HTTPException(404, "Uploaded file not found. Upload again.")

    video_path = str(upload_match[0])
    analysis_id = str(uuid.uuid4())[:8]

    # Create evidence dir for this analysis
    ev_dir = EVIDENCE_DIR / analysis_id
    ev_dir.mkdir(exist_ok=True)

    with lock:
        analyses[analysis_id] = {
            "status": "running",
            "stage": "initializing",
            "current_frame": 0,
            "total_frames": 0,
            "percentage": 0,
            "events": [],
            "output_path": None,
            "video_path": video_path,
            "evidence_dir": str(ev_dir),
            "error": None,
            "video_info": None,
        }

    thread = threading.Thread(
        target=_run_pipeline,
        args=(analysis_id, video_path),
        daemon=True,
    )
    thread.start()

    return {"analysis_id": analysis_id}


def _run_pipeline(analysis_id: str, video_path: str):
    """Run the real CV pipeline in a background thread."""
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            with lock:
                analyses[analysis_id]["status"] = "error"
                analyses[analysis_id]["error"] = "Cannot open video file"
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

        with lock:
            analyses[analysis_id]["total_frames"] = total_frames
            analyses[analysis_id]["video_info"] = {
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames,
            }

        engine = TemporalEventEngine(Thresholds(
            movement_threshold=50.0,
            persistence_frames=30,
            actor_absence_frames=15,
            association_radius=400.0,
            min_track_length=5,
            video_fps=fps,
        ))

        candidate_config = CandidateConfig(
            persist_frames=max(20, int(fps * 1.5)),
            actor_absence_frames=max(10, 15),
        )
        candidate_detector = ActionCandidateDetector(candidate_config, frame_size=(width, height))
        model = YOLO("yolov8n.pt")

        output_path = str(OUTPUT_DIR / f"{analysis_id}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_num = 0
        events = []
        evidence_frames = {}
        evidence_idx = {}
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            pct = min(int((frame_num / total_frames) * 100), 100) if total_frames > 0 else 0

            # Determine pipeline stage
            if pct < 15:
                stage = "yolo_detection"
            elif pct < 35:
                stage = "bytetrack_tracking"
            elif pct < 60:
                stage = "temporal_reasoning"
            elif pct < 85:
                stage = "vlm_verification"
            else:
                stage = "finalizing"

            with lock:
                analyses[analysis_id]["current_frame"] = frame_num
                analyses[analysis_id]["percentage"] = pct
                analyses[analysis_id]["stage"] = stage

            results = model.track(frame, persist=True, conf=0.25,
                                  tracker="bytetrack_ultralow.yaml", verbose=False)
            r = results[0]

            detections = []
            if r.boxes is not None and len(r.boxes) > 0 and r.boxes.id is not None:
                for i in range(len(r.boxes)):
                    tid = int(r.boxes.id[i])
                    cls = int(r.boxes.cls[i])
                    conf = float(r.boxes.conf[i])
                    name = r.names[cls]
                    x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy().astype(int)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    detections.append({
                        "track_id": tid,
                        "class_name": name,
                        "centroid": (cx, cy),
                        "confidence": conf,
                        "bbox": (x1, y1, x2, y2),
                    })

            new_events = engine.update(detections, frame_num)

            # Complementary path: candidate discovery via background subtraction
            person_tracks = [d for d in detections if d["class_name"] == "person"]
            yolo_track_ids = {d["track_id"] for d in detections}
            new_candidates = candidate_detector.update(frame, person_tracks, yolo_track_ids)

            # Annotate frame
            annotated = frame.copy()
            for det in detections:
                tid = det["track_id"]
                x1, y1, x2, y2 = det["bbox"]
                cx, cy = det["centroid"]
                obj = engine.objects.get(tid)
                state = obj.state if obj else State.IDLE

                if state == State.DUMPING_CANDIDATE:
                    color, thickness = (0, 0, 255), 3
                    label = "UNATTENDED OBJECT"
                elif state == State.ACTOR_LEFT:
                    color, thickness = (0, 200, 255), 2
                    label = "ACTOR LEFT"
                elif state == State.SUSPICIOUS:
                    color, thickness = (0, 255, 255), 2
                    label = "SUSPICIOUS"
                elif state == State.OBSERVING:
                    color, thickness = (0, 255, 0), 1
                    label = "OBSERVING"
                else:
                    color, thickness = (200, 200, 200), 1
                    label = det["class_name"]

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
                cv2.circle(annotated, (cx, cy), 4, color, -1)
                cv2.putText(annotated, f"ID:{tid}", (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(annotated, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            for tid, obj in engine.objects.items():
                if obj.state == State.DUMPING_CANDIDATE and obj.last_centroid:
                    cx, cy = obj.last_centroid
                    cv2.circle(annotated, (cx, cy), 40, (0, 0, 255), 3)
                    cv2.putText(annotated, "!! UNATTENDED OBJECT !!", (cx - 100, cy - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Annotate candidate regions from complementary path
            for bid, blob in candidate_detector.blobs.items():
                bx1, by1, bx2, by2 = blob.bbox
                if blob.emitted:
                    color = (255, 0, 255)
                    label = f"CANDIDATE #{blob.id}"
                elif blob.person_was_nearby and blob.frames_since_person > 0:
                    color = (255, 165, 0)
                    label = f"PERSISTING #{blob.id}"
                elif blob.person_was_nearby:
                    color = (255, 255, 0)
                    label = f"PROXIMITY #{blob.id}"
                else:
                    color = (128, 128, 128)
                    label = f"BG #{blob.id}"
                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), color, 1)
                cv2.putText(annotated, label, (bx1, by1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

            writer.write(annotated)

            for event in new_events:
                vlm_result = verify_dumping_event(
                    frame=frame,
                    track_id=event.track_id,
                    class_name=event.class_name,
                    centroid=event.centroid,
                    bbox=event.bbox,
                )
                event.vlm = vlm_result
                obj = engine.objects.get(event.track_id)
                if obj:
                    obj._last_vlm = vlm_result
                events.append(event)
                evidence_frames[event.track_id] = annotated.copy()

                # Save evidence frames
                ev_dir = Path(analyses[analysis_id]["evidence_dir"])
                idx = evidence_idx.get(event.track_id, 0)
                ev_path = ev_dir / f"{event.track_id}_{idx}.jpg"
                cv2.imwrite(str(ev_path), annotated)
                evidence_idx[event.track_id] = idx + 1

            # Complementary path: VLM verification for emitted candidates
            for cand in new_candidates:
                vlm_result = verify_dumping_event(
                    frame=frame,
                    track_id=cand.id,
                    class_name="candidate_region",
                    centroid=cand.centroid,
                    bbox=cand.bbox,
                )
                cand._vlm_result = vlm_result
                # Track confirmed candidates as events
                if vlm_result.verified and vlm_result.confirmed:
                    candidate_event = DumpingEvent(
                        track_id=cand.id,
                        class_name="candidate_region",
                        frame_num=cand.first_frame,
                        timestamp=cand.first_frame / fps,
                        stationary_duration_frames=cand.frames_active,
                        actor_status="LEFT",
                        centroid=cand.centroid,
                        bbox=cand.bbox,
                        vlm=vlm_result,
                    )
                    events.append(candidate_event)
                    evidence_frames[cand.id] = annotated.copy()
                    ev_dir = Path(analyses[analysis_id]["evidence_dir"])
                    idx = evidence_idx.get(cand.id, 0)
                    ev_path = ev_dir / f"candidate_{cand.id}_{idx}.jpg"
                    cv2.imwrite(str(ev_path), annotated)
                    evidence_idx[cand.id] = idx + 1

            # Log progress every 30 frames
            if frame_num % 30 == 0:
                elapsed = time.time() - start_time
                cur_fps = frame_num / elapsed if elapsed > 0 else 0
                print(f"  [{analysis_id}] Frame {frame_num}/{total_frames} | "
                      f"FPS: {cur_fps:.1f} | Events: {len(events)}")

        cap.release()
        writer.release()

        # Re-encode to H.264 for browser playback
        h264_path = output_path.replace(".mp4", "_h264.mp4")
        try:
            import subprocess
            result = subprocess.run(
                ["ffmpeg", "-y", "-i", output_path, "-c:v", "libx264", "-preset", "fast",
                 "-crf", "23", "-pix_fmt", "yuv420p", h264_path],
                check=True, capture_output=True, timeout=120,
            )
            output_path = h264_path
        except Exception as ex:
            print(f"  [{analysis_id}] ffmpeg re-encode failed: {ex}")

        elapsed = time.time() - start_time
        avg_fps = frame_num / elapsed if elapsed > 0 else 0

        # Serialize events for JSON
        events_json = []
        for e in events:
            ev = {
                "track_id": e.track_id,
                "class_name": e.class_name,
                "frame_num": e.frame_num,
                "timestamp": round(e.timestamp, 2),
                "stationary_duration_frames": e.stationary_duration_frames,
                "stationary_duration_sec": round(e.stationary_duration_frames / fps, 1) if fps else 0,
                "actor_status": e.actor_status,
                "centroid": [int(x) for x in e.centroid] if e.centroid else None,
                "bbox": [int(x) for x in e.bbox] if e.bbox else None,
            }
            if hasattr(e, "vlm") and e.vlm:
                vlm = e.vlm
                ev["vlm"] = {
                    "confirmed": vlm.confirmed,
                    "event_type": vlm.event_type,
                    "severity": vlm.severity,
                    "summary": vlm.summary,
                    "verified": vlm.verified,
                    "latency_ms": round(vlm.latency_ms, 0),
                    "model": vlm.model,
                }
            else:
                ev["vlm"] = None
            events_json.append(ev)

        with lock:
            analyses[analysis_id]["status"] = "complete"
            analyses[analysis_id]["stage"] = "complete"
            analyses[analysis_id]["percentage"] = 100
            analyses[analysis_id]["events"] = events_json
            analyses[analysis_id]["output_path"] = output_path
            analyses[analysis_id]["video_info"]["avg_fps"] = round(avg_fps, 1)
            analyses[analysis_id]["video_info"]["elapsed_sec"] = round(elapsed, 1)

        print(f"  [{analysis_id}] Analysis complete: {frame_num} frames, "
              f"{len(events)} events, {elapsed:.1f}s")

    except Exception as e:
        import traceback
        traceback.print_exc()
        with lock:
            analyses[analysis_id]["status"] = "error"
            analyses[analysis_id]["error"] = str(e)


# ---------------------------------------------------------------------------
# Progress polling
# ---------------------------------------------------------------------------

@app.get("/api/progress/{analysis_id}")
async def get_progress(analysis_id: str):
    with lock:
        a = analyses.get(analysis_id)
    if not a:
        raise HTTPException(404, "Analysis not found")
    return {
        "status": a["status"],
        "stage": a["stage"],
        "current_frame": a["current_frame"],
        "total_frames": a["total_frames"],
        "percentage": a["percentage"],
        "events_count": len(a["events"]),
        "error": a["error"],
    }


# ---------------------------------------------------------------------------
# Full results
# ---------------------------------------------------------------------------

@app.get("/api/results/{analysis_id}")
async def get_results(analysis_id: str):
    with lock:
        a = analyses.get(analysis_id)
    if not a:
        raise HTTPException(404, "Analysis not found")
    if a["status"] != "complete":
        return {"status": a["status"], "stage": a["stage"]}
    return {
        "status": "complete",
        "events": a["events"],
        "video_info": a["video_info"],
    }


# ---------------------------------------------------------------------------
# Evidence frames
# ---------------------------------------------------------------------------

@app.get("/api/evidence/{analysis_id}/{track_id}/{idx}")
async def get_evidence_frame(analysis_id: str, track_id: str, idx: int):
    with lock:
        a = analyses.get(analysis_id)
    if not a:
        raise HTTPException(404, "Analysis not found")

    ev_dir = Path(a["evidence_dir"])
    frame_path = ev_dir / f"{track_id}_{idx}.jpg"
    if not frame_path.exists():
        raise HTTPException(404, "Evidence frame not found")

    return FileResponse(str(frame_path), media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Annotated video
# ---------------------------------------------------------------------------

@app.get("/api/video/{analysis_id}")
async def get_annotated_video(analysis_id: str, request: Request):
    with lock:
        a = analyses.get(analysis_id)
    if not a:
        raise HTTPException(404, "Analysis not found")
    if not a.get("output_path") or not os.path.exists(a["output_path"]):
        raise HTTPException(404, "Video not yet ready")
    path = a["output_path"]
    file_size = os.path.getsize(path)
    range_header = request.headers.get("range")
    if range_header:
        start, end = range_header.replace("bytes=", "").split("-")
        start = int(start)
        end = int(end) if end else file_size - 1
        end = min(end, file_size - 1)
        content_length = end - start + 1
        def iter_file():
            with open(path, "rb") as f:
                f.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk = f.read(min(8192, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
        return StreamingResponse(
            iter_file(),
            status_code=206,
            media_type="video/mp4",
            headers={
                "Content-Range": f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges": "bytes",
                "Content-Length": str(content_length),
            },
        )
    return FileResponse(path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

@app.delete("/api/cleanup/{analysis_id}")
async def cleanup(analysis_id: str):
    with lock:
        a = analyses.pop(analysis_id, None)
    if a:
        # Remove temp files
        ev_dir = Path(a.get("evidence_dir", ""))
        if ev_dir.exists():
            shutil.rmtree(ev_dir, ignore_errors=True)
        for p in OUTPUT_DIR.glob(f"{analysis_id}_*"):
            p.unlink(missing_ok=True)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  DetectDump — FastAPI server")
    print("  http://127.0.0.1:8080")
    print("=" * 60)
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")
