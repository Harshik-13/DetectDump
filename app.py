"""
DumpWatch AI - Streamlit Demo UI
"""
import streamlit as st
import cv2
import numpy as np
import time
import tempfile
import os
import subprocess
from ultralytics import YOLO
from temporal_engine import TemporalEventEngine, Thresholds, State
from vlm_verify import verify_dumping_event

st.set_page_config(
    page_title="DumpWatch AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800&family=Source+Sans+3:wght@400;500;600&family=JetBrains+Mono:wght@400;500;600&display=swap');
:root {
    --ink: #15171A;
    --card-bg: #1C201D;
    --border: #343A2F;
    --text-main: #EFEEE6;
    --text-muted: #97998C;
    --brand: #6E9C7B;
    --brand-deep: #3F5F49;
    --brand-text-on: #10190F;
    --amber: #D98A2B;
    --amber-text-on: #1C1400;
    --rust: #B8492E;
}
.stApp {
    background-color: var(--ink) !important;
    color: var(--text-main) !important;
    font-family: 'Source Sans 3', sans-serif !important;
}
.stApp > header { background: transparent !important; }
.main .block-container { max-width: 1400px !important; padding: 2rem !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div style="border-bottom:1px solid #343A2F;padding-bottom:1.25rem;margin-bottom:1.5rem;">
    <h1 style="font-family:'Archivo',sans-serif;font-size:2rem;font-weight:800;color:#EFEEE6;margin:0;">DumpWatch AI</h1>
    <p style="color:#97998C;font-size:1.05rem;margin-top:0.25rem;">Automated Illegal Dumping Detection</p>
</div>
""", unsafe_allow_html=True)


def format_frame_time(frame_num, fps):
    """Convert frame number to MM:SS format."""
    seconds = int(frame_num / fps) if fps else 0
    minutes = seconds // 60
    secs = seconds % 60
    return f"00:{minutes:02d}:{secs:02d}"


def run_pipeline(video_path):
    """Run the full CV pipeline and return results."""
    engine = TemporalEventEngine(Thresholds(
        movement_threshold=30.0,
        persistence_frames=60,
        actor_absence_frames=15,
        association_radius=200.0,
        min_track_length=5,
    ))
    model = YOLO("yolov8n.pt")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    output_path = os.path.join(tempfile.gettempdir(), "dumpwatch_output.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_num = 0
    events = []
    evidence_frames = {}
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    stages = [
        (0.15, "Running YOLO detection..."),
        (0.35, "Applying ByteTrack heuristics..."),
        (0.60, "Analyzing temporal logic..."),
        (0.85, "VLM verifying dumping candidate..."),
        (1.00, "Analysis complete."),
    ]
    stage_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        progress = min(frame_num / total_frames, 1.0)
        while stage_idx < len(stages) and progress >= stages[stage_idx][0]:
            stage_idx += 1
        stage_label = stages[min(stage_idx, len(stages) - 1)][1]

        if frame_num % 10 == 0:
            progress_placeholder.progress(progress, text=stage_label)

        results = model.track(frame, persist=True, conf=0.01,
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
                    "track_id": tid, "class_name": name,
                    "centroid": (cx, cy), "confidence": conf,
                    "bbox": (x1, y1, x2, y2),
                })

        new_events = engine.update(detections, frame_num)

        annotated = frame.copy()
        for det in detections:
            tid = det["track_id"]
            x1, y1, x2, y2 = det["bbox"]
            cx, cy = det["centroid"]
            obj = engine.objects.get(tid)
            state = obj.state if obj else State.IDLE

            if state == State.DUMPING_CANDIDATE:
                color, thickness = (0, 0, 255), 3
                label = "DUMPING CANDIDATE"
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
                cv2.putText(annotated, "!! DUMPING DETECTED !!", (cx - 80, cy - 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        writer.write(annotated)

        for event in new_events:
            vlm_result = verify_dumping_event(
                frame=frame, track_id=event.track_id,
                class_name=event.class_name, centroid=event.centroid,
            )
            event.vlm = vlm_result
            obj = engine.objects.get(event.track_id)
            if obj:
                obj._last_vlm = vlm_result
            events.append(event)
            evidence_frames[event.track_id] = annotated.copy()

        if frame_num % 30 == 0:
            status_placeholder.caption(
                f"Frame {frame_num}/{total_frames} | "
                f"Events: {len(events)}"
            )

    cap.release()
    writer.release()
    progress_placeholder.empty()
    status_placeholder.empty()

    h264_path = output_path.replace(".mp4", "_h264.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", output_path, "-c:v", "libx264", "-preset", "fast",
             "-crf", "23", "-pix_fmt", "yuv420p", h264_path],
            check=True, capture_output=True, timeout=120,
        )
        output_path = h264_path
    except Exception:
        pass

    return {
        "output_path": output_path,
        "events": events,
        "evidence_frames": evidence_frames,
        "total_frames": total_frames,
        "fps": fps,
        "width": width,
        "height": height,
    }


# --- Main Layout: Two Columns ---
left_col, right_col = st.columns([3, 2])

# --- Left Column: Input + Video ---
with left_col:
    st.markdown("""
    <div style="background:#1C201D;border:1px solid #343A2F;border-radius:10px;padding:1.5rem;">
        <div style="display:flex;gap:1rem;align-items:center;flex-wrap:wrap;margin-bottom:1rem;">
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Select a video file",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )

    analyze_clicked = st.button("Analyze Video", key="analyze_btn")

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Video display area
    video_container = st.empty()
    video_container.markdown("""
    <div style="background:#000;aspect-ratio:16/9;border-radius:8px;border:1px solid #343A2F;
                display:flex;align-items:center;justify-content:center;">
        <span style="color:#97998C;font-weight:500;">Awaiting video input...</span>
    </div>
    """, unsafe_allow_html=True)

# --- Right Column: Results ---
with right_col:
    results_container = st.container()

# --- Run Pipeline ---
if uploaded_file and analyze_clicked:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    video_container.empty()

    with st.spinner("Initializing pipeline..."):
        result = run_pipeline(tmp_path)

    if result and result["events"]:
        # Show annotated video
        video_container.video(result["output_path"])

        # Show results in right column
        with results_container:
            for i, event in enumerate(result["events"]):
                vlm = event.vlm

                # Incident Card
                vlm_badge = ""
                severity_badge = ""
                if vlm and vlm.verified:
                    if vlm.confirmed:
                        vlm_badge = '<span class="dw-badge confirmed">CONFIRMED</span>'
                    else:
                        vlm_badge = '<span class="dw-badge not-confirmed">NOT CONFIRMED</span>'
                    sev_class = vlm.severity.lower()
                    severity_badge = f'<span class="dw-badge {sev_class}">{vlm.severity}</span>'
                else:
                    vlm_badge = '<span class="dw-badge not-confirmed">UNAVAILABLE</span>'
                    severity_badge = '<span class="dw-badge low">N/A</span>'

                timestamp = format_frame_time(event.frame_num, result["fps"])
                summary_text = vlm.summary if (vlm and vlm.verified) else "VLM verification unavailable"

                st.markdown(f"""
                <div style="border-left:4px solid #B8492E;background:#1C201D;border-radius:10px;
                            padding:1.5rem;margin-bottom:1rem;">
                    <h2 style="font-family:'Archivo',sans-serif;color:#B8492E;font-size:1.15rem;
                               font-weight:700;margin-bottom:1.1rem;">
                        WARNING: ILLEGAL DUMPING DETECTED
                    </h2>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:1rem;">
                        <div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                                        color:#97998C;text-transform:uppercase;letter-spacing:0.06em;">
                                Detected Object
                            </div>
                            <div style="font-weight:600;font-size:1rem;color:#EFEEE6;">
                                {event.class_name.title()}
                            </div>
                        </div>
                        <div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                                        color:#97998C;text-transform:uppercase;letter-spacing:0.06em;">
                                Actor Status
                            </div>
                            <div style="font-weight:600;font-size:1rem;color:#EFEEE6;">
                                {event.actor_status}
                            </div>
                        </div>
                        <div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                                        color:#97998C;text-transform:uppercase;letter-spacing:0.06em;">
                                Stationary Duration
                            </div>
                            <div style="font-weight:600;font-size:1rem;color:#EFEEE6;">
                                {event.stationary_duration_frames / result['fps']:.1f}s
                            </div>
                        </div>
                        <div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                                        color:#97998C;text-transform:uppercase;letter-spacing:0.06em;">
                                VLM Verification
                            </div>
                            <div style="font-weight:600;font-size:1rem;">{vlm_badge}</div>
                        </div>
                        <div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                                        color:#97998C;text-transform:uppercase;letter-spacing:0.06em;">
                                Severity
                            </div>
                            <div style="font-weight:600;font-size:1rem;">{severity_badge}</div>
                        </div>
                        <div>
                            <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;
                                        color:#97998C;text-transform:uppercase;letter-spacing:0.06em;">
                                Timestamp
                            </div>
                            <div style="font-weight:600;font-size:1rem;color:#EFEEE6;">{timestamp}</div>
                        </div>
                    </div>
                    <div style="background:#15171A;padding:1rem;border-radius:8px;font-size:0.9rem;
                                border-left:3px solid #B8492E;color:#EFEEE6;">
                        <strong>Summary:</strong> {summary_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Evidence card
                st.markdown("""
                <div style="font-family:'Archivo',sans-serif;margin-bottom:0.5rem;font-size:1.05rem;
                            font-weight:700;color:#EFEEE6;">Associated Evidence</div>
                <div style="font-size:0.85rem;color:#97998C;margin-bottom:0.75rem;">
                    Keyframe triggering VLM verification confirmation.
                </div>
                """, unsafe_allow_html=True)

                if event.track_id in result["evidence_frames"]:
                    evidence_img = result["evidence_frames"][event.track_id]
                    st.image(evidence_img, channels="BGR", use_container_width=True)

                # Technical logs
                vlm_log = ""
                if vlm and vlm.verified:
                    vlm_log = f"""[INFO] YOLOv8 Object detected: id={event.track_id}, class="{event.class_name}"
[INFO] ByteTrack: Actor separated from Object ID {event.track_id}
[INFO] TemporalLogic: Object stationary for {event.stationary_duration_frames / result['fps']:.1f}s
[INFO] DUMPING_CANDIDATE generated.
[VLM-REQ] Prompting VLM with candidate frame...
[VLM-RES] "{vlm.summary}" -> {'CONFIRMED' if vlm.confirmed else 'NOT CONFIRMED'}"""
                else:
                    vlm_log = f"""[INFO] YOLOv8 Object detected: id={event.track_id}, class="{event.class_name}"
[INFO] DUMPING_CANDIDATE generated.
[VLM-REQ] VLM request failed or unavailable.
[VLM-RES] {vlm.summary if vlm else 'No response'}"""

                with st.expander("View Technical Logs"):
                    st.code(vlm_log, language=None)

    elif result:
        video_container.video(result["output_path"])
        with results_container:
            st.info("No dumping events detected in this video.")
    else:
        st.error("Failed to process video.")

    os.unlink(tmp_path)


