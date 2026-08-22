# DetectDump

Illegal Dumping Event Detector — CV pipeline that detects when an actor abandons an object by analyzing temporal persistence after departure.

## How It Works

```
CCTV/Video → YOLO Detection → ByteTrack Tracking → Temporal Event Engine → DUMPING_CANDIDATE
```

1. **YOLOv8n** detects persons, bags, bottles, and other objects per frame
2. **ByteTrack** assigns persistent IDs across frames
3. **Temporal Engine** tracks each object's state:
   - `IDLE → OBSERVING → SUSPICIOUS → ACTOR_LEFT → DUMPING_CANDIDATE`
4. When an object remains stationary after its associated actor leaves, a dumping event is triggered

## Quick Start

```bash
pip install opencv-python ultralytics
python dumping_detector.py <video_path> <output_path>
```

## Project Structure

```
├── dumping_detector.py          # Full pipeline: YOLO + tracking + temporal engine
├── temporal_engine.py           # State machine for dumping event detection
├── pipeline_test.py             # CV foundation test (detection + tracking only)
├── test_temporal_engine.py      # Unit tests for temporal engine
├── bytetrack_ultralow.yaml      # ByteTrack config for low-confidence objects
├── test_videos/                 # Test video inputs
└── AGENTS.md                    # Hackathon operating constitution
```

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
