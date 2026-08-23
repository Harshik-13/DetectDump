# DetectDump

Illegal Dumping Event Detector — CV pipeline that detects when an actor abandons an object by analyzing temporal persistence after departure.

## How It Works

```
CCTV/Video → YOLO Detection + Background Subtraction → ByteTrack Tracking → Temporal Event Engine → DUMPING_CANDIDATE → VLM Verification → Evidence
```

1. **YOLOv8n** detects persons, bags, bottles, and other objects per frame (conf ≥ 0.25)
2. **ByteTrack** assigns persistent IDs across frames (track_high_thresh = 0.25)
3. **Complementary Path** — MOG2 background subtraction discovers candidate regions near persons that YOLO may miss
4. **Temporal Engine** tracks each object's state:
   - `IDLE → OBSERVING → SUSPICIOUS → ACTOR_LEFT → DUMPING_CANDIDATE`
   - Tolerates brief YOLO detection gaps without resetting progress
5. When an object remains stationary after its associated actor leaves, a dumping event is triggered
6. **VLM (Gemini 2.5 Flash)** verifies each candidate event against the visual scene
7. **FastAPI backend + reference UI** serves the 4-stage flow: Upload → Analyze → Review → Evidence

## Quick Start

```bash
pip install opencv-python ultralytics fastapi uvicorn python-dotenv openai
# Set your API key in .env
# Run the server
python app_server.py
# Open http://127.0.0.1:8080
```

Or run the pipeline directly from CLI:

```bash
python dumping_detector.py <video_path> <output_path>
```

## Project Structure

```
├── app_server.py                 # FastAPI backend (API + serves UI)
├── app.py                        # Streamlit demo UI (legacy)
├── dumping_detector.py           # Full pipeline: YOLO + tracking + temporal engine + VLM
├── temporal_engine.py            # State machine for dumping event detection
├── action_candidate_detector.py  # Complementary CV path — background subtraction + person proximity
├── vlm_verify.py                 # VLM verification via Gemini 2.5 Flash
├── test_temporal_engine.py       # Unit tests for temporal engine
├── bytetrack_ultralow.yaml       # ByteTrack config tuned for low-confidence objects
├── ui/
│   ├── detectdump.html           # Production UI (connected to FastAPI backend)
│   └── dumpdetect-clean-v3.html  # Visual design source of truth
├── test_videos/                  # Test video inputs
├── .env                          # API keys (not committed)
├── .gitignore                    # Git ignore rules
├── AGENTS.md                     # Hackathon operating constitution
└── README.md                     # This file
```

## Demo

```bash
python app_server.py
# Open http://127.0.0.1:8080
# Upload a video → Click "Run analysis" → View results → View evidence
```

The demo shows:
- **4-stage flow**: Upload → Analyzing (real progress) → Review detection → Associated evidence
- **Annotated video** with bounding boxes, track IDs, and state labels
- **Dual-path detection** — YOLO tracking + complementary background subtraction candidates
- **Detection review** with detected object, actor status, stationary duration, VLM verification, severity, and timestamp
- **Evidence grid** with keyframe images from the actual analysis
- **Technical logs** with real pipeline info (video dimensions, FPS, frame count, event details)
- **New scan** resets everything without browser refresh

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the DetectDump UI |
| `POST` | `/api/upload` | Upload a video file |
| `POST` | `/api/analyze` | Start real CV analysis |
| `GET` | `/api/progress/{id}` | Poll analysis progress |
| `GET` | `/api/results/{id}` | Get full results |
| `GET` | `/api/evidence/{id}/{track}/{idx}` | Serve evidence frame |
| `GET` | `/api/video/{id}` | Serve annotated video (Range request support) |
| `DELETE` | `/api/cleanup/{id}` | Clean up temp files |

## Temporal Engine States

| State | Meaning |
|-------|---------|
| `IDLE` | Object not yet tracked enough |
| `OBSERVING` | Object tracked, evaluating |
| `SUSPICIOUS` | Object stationary + actor nearby |
| `ACTOR_LEFT` | Associated actor disappeared |
| `DUMPING_CANDIDATE` | Object persisted alone beyond threshold |

## Configurable Thresholds

```python
Thresholds(
    movement_threshold=50.0,    # px — centroid spread to be "stationary"
    persistence_frames=30,      # frames object must persist alone (~1.25s at 24fps)
    actor_absence_frames=15,    # frames before actor considered "left"
    association_radius=400.0,   # px — max distance for actor-object association
    min_track_length=5,         # minimum frames before evaluation
)
```

## Tech Stack

- **Python 3.11**
- **YOLOv8n** (Ultralytics) — object detection
- **ByteTrack** — multi-object tracking (low-confidence tuned)
- **OpenCV** — video I/O, annotation, MOG2 background subtraction
- **PyTorch** (CPU) — inference backend
- **Gemini 2.5 Flash** (Google AI) — VLM verification
- **FastAPI + Uvicorn** — backend server
- **FFmpeg** — H.264 video re-encoding for browser playback

## Git Tags

| Tag | Description |
|-----|-------------|
| `v2.3-detection-fixed` | Detection pipeline working end-to-end with VLM confirmation |
| `v2.2-stable-mvp` | Stable MVP — dual-path detection with complementary candidate discovery |
| `v2.0-fastapi-ui` | FastAPI backend + reference UI frontend |
| `v1.1-generalized` | Generalized detection (behavioral, not waste-class-specific) |
| `v1.0-code-freeze` | Critical fix pass |
| `v0.3-phase5-ui` | Streamlit demo UI |
| `v0.2-phase4-vlm` | VLM verification added |
