# DetectDump

Illegal Dumping Event Detector — CV pipeline that detects when an actor abandons an object by analyzing temporal persistence after departure.

## How It Works

```
CCTV/Video → YOLO Detection → ByteTrack Tracking → Temporal Event Engine → DUMPING_CANDIDATE → VLM Verification → Evidence Replay
```

1. **YOLOv8n** detects persons, bags, bottles, and other objects per frame
2. **ByteTrack** assigns persistent IDs across frames
3. **Temporal Engine** tracks each object's state:
   - `IDLE → OBSERVING → SUSPICIOUS → ACTOR_LEFT → DUMPING_CANDIDATE`
4. When an object remains stationary after its associated actor leaves, a dumping event is triggered
5. **VLM (GPT-4o-mini via OpenRouter)** verifies each candidate event against the visual scene
6. **Streamlit UI** displays annotated video, incident cards, evidence keyframes, and verification results

## Quick Start

```bash
pip install opencv-python ultralytics streamlit
# Set your OpenRouter API key
export OPENROUTER_API_KEY="your-key-here"
# Run the demo
streamlit run app.py
```

Or run the pipeline directly:

```bash
python dumping_detector.py <video_path> <output_path>
```

## Project Structure

```
├── app.py                        # Streamlit demo UI
├── dumping_detector.py           # Full pipeline: YOLO + tracking + temporal engine + VLM
├── temporal_engine.py            # State machine for dumping event detection
├── vlm_verify.py                 # VLM verification via OpenRouter (GPT-4o-mini)
├── pipeline_test.py              # CV foundation test (detection + tracking only)
├── test_temporal_engine.py       # Unit tests for temporal engine
├── bytetrack_ultralow.yaml       # ByteTrack config for low-confidence objects
├── test_videos/                  # Test video inputs
├── .env                          # API keys (not committed)
├── AGENTS.md                     # Hackathon operating constitution
└── ui/                           # HTML reference design
```

## Demo

```bash
streamlit run app.py
# Open http://localhost:8501
# Upload a video → Click "Analyze Video" → View results
```

The demo shows:
- **Annotated video** with bounding boxes, track IDs, and state labels
- **Incident cards** with detected object, actor status, stationary duration, severity, and timestamp
- **VLM verification** summary and confirmation badge
- **Associated evidence** keyframes with detection overlays
- **Technical logs** with pipeline event trace

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
- **Streamlit** — demo UI
- **FFmpeg** — H.264 video re-encoding for browser playback
