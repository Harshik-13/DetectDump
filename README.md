# DetectDump

Illegal Dumping Event Detector — CV pipeline that detects when an actor abandons an object by analyzing temporal persistence after departure.

## How It Works

```
CCTV/Video → YOLO Detection → ByteTrack Tracking → Temporal Event Engine → DUMPING_CANDIDATE → VLM Verification → Evidence
```

1. **YOLOv8n** detects persons, bags, bottles, and other objects per frame
2. **ByteTrack** assigns persistent IDs across frames
3. **Temporal Engine** tracks each object's state:
   - `IDLE → OBSERVING → SUSPICIOUS → ACTOR_LEFT → DUMPING_CANDIDATE`
4. When an object remains stationary after its associated actor leaves, a dumping event is triggered
5. **VLM (GPT-4o-mini via OpenRouter)** verifies each candidate event against the visual scene
6. **FastAPI backend + reference UI** serves the 4-stage flow: Upload → Analyze → Review → Evidence

## Quick Start

```bash
pip install opencv-python ultralytics fastapi uvicorn python-dotenv openai
# Set your OpenRouter API key
export OPENROUTER_API_KEY="your-key-here"
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
├── vlm_verify.py                 # VLM verification via OpenRouter (GPT-4o-mini)
├── test_temporal_engine.py       # Unit tests for temporal engine
├── bytetrack_ultralow.yaml       # ByteTrack config for low-confidence objects
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
| `GET` | `/api/video/{id}` | Serve annotated video |
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
    movement_threshold=30.0,    # px — centroid spread to be "stationary"
    persistence_frames=60,      # frames object must persist alone
    actor_absence_frames=15,    # frames before actor considered "left"
    association_radius=200.0,   # px — max distance for actor-object association
    min_track_length=5,         # minimum frames before evaluation
)
```

## Tech Stack

- **Python 3.11**
- **YOLOv8n** (Ultralytics) — object detection
- **ByteTrack** — multi-object tracking
- **OpenCV** — video I/O and annotation
- **PyTorch** (CPU) — inference backend
- **OpenRouter / GPT-4o-mini** — VLM verification
- **FastAPI + Uvicorn** — backend server
- **FFmpeg** — H.264 video re-encoding for browser playback
