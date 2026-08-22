"""
Dumping Detection Pipeline
VIDEO → YOLO + TRACKING → TEMPORAL ENGINE → VLM VERIFICATION → INCIDENT RESULT
"""

import cv2
import time
import sys
from ultralytics import YOLO
from temporal_engine import TemporalEventEngine, Thresholds, State
from vlm_verify import verify_dumping_event


def run_dumping_detection(video_path: str, output_path: str = "output_dumping.mp4",
                          thresholds: Thresholds = None):
    """Full pipeline: detection + tracking + temporal event engine."""

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Use sensible defaults if no thresholds provided
    if thresholds is None:
        thresholds = Thresholds(
            movement_threshold=30.0,
            persistence_frames=60,     # ~2.5s at 24fps
            actor_absence_frames=15,   # ~0.6s at 24fps
            association_radius=200.0,
            min_track_length=5,
            video_fps=fps,
        )
    engine = TemporalEventEngine(thresholds)
    model = YOLO("yolov8n.pt")

    print(f"Model: yolov8n.pt")
    print(f"Video: {video_path}")
    print(f"Thresholds: movement={engine.thresholds.movement_threshold}px, "
          f"persistence={engine.thresholds.persistence_frames}f, "
          f"actor_absence={engine.thresholds.actor_absence_frames}f")
    print(f"Resolution: {width}x{height}, Frames: {total_frames}, FPS: {fps}")
    print("=" * 60)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    events_this_run = []
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        results = model.track(frame, persist=True, conf=0.01,
                              tracker="bytetrack_ultralow.yaml", verbose=False)
        r = results[0]

        # Extract detections for temporal engine
        detections = []
        if r.boxes is not None and len(r.boxes) > 0 and r.boxes.id is not None:
            for i in range(len(r.boxes)):
                tid = int(r.boxes.id[i])
                cls = int(r.boxes.cls[i])
                conf = float(r.boxes.conf[i])
                name = r.names[cls]
                x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy().astype(int)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                detections.append({
                    "track_id": tid,
                    "class_name": name,
                    "centroid": (cx, cy),
                    "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                })

        # Run temporal engine
        new_events = engine.update(detections, frame_num)

        # Annotate frame
        annotated = frame.copy()

        # Draw all detections with state-dependent colors
        for det in detections:
            tid = det["track_id"]
            name = det["class_name"]
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["centroid"]

            obj = engine.objects.get(tid)
            state = obj.state if obj else State.IDLE

            # Color by state
            if state == State.DUMPING_CANDIDATE:
                color = (0, 0, 255)       # Red - CRITICAL
                thickness = 3
                label = f"DUMPING CANDIDATE"
            elif state == State.PERSISTING:
                color = (0, 140, 255)      # Orange
                thickness = 2
                label = f"PERSISTING"
            elif state == State.ACTOR_LEFT:
                color = (0, 200, 255)      # Yellow-orange
                thickness = 2
                label = f"ACTOR LEFT"
            elif state == State.SUSPICIOUS:
                color = (0, 255, 255)      # Yellow
                thickness = 2
                label = f"SUSPICIOUS"
            elif state == State.OBSERVING:
                color = (0, 255, 0)        # Green
                thickness = 1
                label = f"OBSERVING"
            else:
                color = (200, 200, 200)    # Gray
                thickness = 1
                label = name

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Draw centroid
            cv2.circle(annotated, (cx, cy), 4, color, -1)

            # Draw track ID
            cv2.putText(annotated, f"ID:{tid}", (x1, y1 - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw state label
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw confidence
            cv2.putText(annotated, f"{det['confidence']:.2f}", (x2 + 5, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Highlight dumping candidates prominently
        for tid, obj in engine.objects.items():
            if obj.state == State.DUMPING_CANDIDATE and obj.last_centroid:
                cx, cy = obj.last_centroid
                # Pulsing alert circle
                cv2.circle(annotated, (cx, cy), 40, (0, 0, 255), 3)
                cv2.putText(annotated, "!! DUMPING DETECTED !!", (cx - 80, cy - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Show VLM status if available
                if hasattr(obj, "_last_vlm") and obj._last_vlm:
                    vlm = obj._last_vlm
                    status = "VLM: CONFIRMED" if vlm.confirmed else "VLM: NOT CONFIRMED"
                    color = (0, 255, 0) if vlm.confirmed else (0, 165, 255)
                    cv2.putText(annotated, status, (cx - 80, cy + 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    cv2.putText(annotated, f"{vlm.severity} | {vlm.event_type}", (cx - 80, cy + 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Frame info overlay
        elapsed = time.time() - start_time
        current_fps = frame_num / elapsed if elapsed > 0 else 0
        info = f"Frame {frame_num}/{total_frames} | FPS: {current_fps:.1f} | Events: {len(engine.events)}"
        cv2.putText(annotated, info, (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        writer.write(annotated)

        # Console output and VLM verification for new events
        for event in new_events:
            events_this_run.append(event)
            print(f"\n{'!'*60}")
            print(f"DUMPING_CANDIDATE")
            print(f"  track_id:             {event.track_id}")
            print(f"  class:                {event.class_name}")
            print(f"  frame:                {event.frame_num}")
            print(f"  timestamp:            {event.timestamp:.2f}")
            print(f"  stationary_duration:  {event.stationary_duration_frames} frames")
            print(f"  actor_status:         {event.actor_status}")
            print(f"  centroid:             {event.centroid}")

            # VLM verification
            print(f"  VLM: Verifying...")
            vlm_result = verify_dumping_event(
                frame=frame,
                track_id=event.track_id,
                class_name=event.class_name,
                centroid=event.centroid,
                bbox=event.bbox,
            )
            event.vlm = vlm_result

            if vlm_result.verified:
                print(f"  VLM: CONFIRMED={vlm_result.confirmed} | "
                      f"type={vlm_result.event_type} | "
                      f"severity={vlm_result.severity}")
                print(f"  VLM: {vlm_result.summary}")
                print(f"  VLM: latency={vlm_result.latency_ms:.0f}ms model={vlm_result.model}")
            else:
                print(f"  VLM: UNAVAILABLE - {vlm_result.summary}")

            # Store VLM result on tracked object for frame annotation
            obj = engine.objects.get(event.track_id)
            if obj:
                obj._last_vlm = vlm_result

            print(f"{'!'*60}")

        if frame_num % 30 == 0 or frame_num == 1:
            summary = engine.get_state_summary()
            active_states = [s["state"] for s in summary.values()]
            print(f"  Frame {frame_num}/{total_frames} | FPS: {current_fps:.1f} | "
                  f"Tracked: {len(summary)} | States: {active_states}")

    elapsed = time.time() - start_time
    avg_fps = frame_num / elapsed if elapsed > 0 else 0

    cap.release()
    writer.release()

    print(f"\n{'='*60}")
    print(f"DUMPING DETECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Frames processed:    {frame_num}")
    print(f"Elapsed time:        {elapsed:.2f}s")
    print(f"Average FPS:         {avg_fps:.1f}")
    print(f"Events detected:     {len(events_this_run)}")
    print(f"Output saved to:     {output_path}")

    if events_this_run:
        print(f"\nEVENT SUMMARY:")
        for e in events_this_run:
            vlm_status = ""
            if hasattr(e, "vlm") and e.vlm:
                if e.vlm.verified:
                    vlm_status = f" | VLM: {'CONFIRMED' if e.vlm.confirmed else 'NOT CONFIRMED'} ({e.vlm.severity})"
                else:
                    vlm_status = " | VLM: unavailable"
            print(f"  Track {e.track_id} ({e.class_name}): "
                  f"stationary {e.stationary_duration_frames} frames, "
                  f"actor {e.actor_status}{vlm_status}")
    else:
        print(f"\nNo dumping events detected.")

    print(f"{'='*60}")

    return {
        "frames": frame_num,
        "elapsed": elapsed,
        "fps": avg_fps,
        "events": events_this_run,
    }


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "test_positive.mp4"
    output = sys.argv[2] if len(sys.argv) > 2 else "output_dumping.mp4"
    run_dumping_detection(video, output)
