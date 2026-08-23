"""
Temporal Event Engine for Illegal Dumping Detection

State Machine:
  IDLE → OBSERVING → SUSPICIOUS → ACTOR_LEFT → PERSISTING → DUMPING_CANDIDATE → RESET

Core Logic:
  For each tracked object, track centroid history.
  Detect when an object becomes stationary while its associated actor leaves.
  If the object persists long enough, flag as DUMPING_CANDIDATE.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

VEHICLE_CLASSES = {"car", "truck", "bus", "motorcycle", "bicycle"}


class State(Enum):
    IDLE = "IDLE"
    OBSERVING = "OBSERVING"
    SUSPICIOUS = "SUSPICIOUS"
    ACTOR_LEFT = "ACTOR_LEFT"
    PERSISTING = "PERSISTING"
    DUMPING_CANDIDATE = "DUMPING_CANDIDATE"
    RESET = "RESET"


@dataclass
class Thresholds:
    movement_threshold: float = 15.0       # pixels — centroid must move less than this to be "stationary"
    persistence_frames: int = 150           # frames object must be stationary (5s @ 30fps)
    actor_absence_frames: int = 30          # frames actor must be gone before counting persistence
    association_radius: float = 150.0       # pixels — how close actor must be to object to be "associated"
    min_track_length: int = 10              # minimum frames an object must be tracked before evaluation
    video_fps: float = 30.0                # video FPS for computing video-relative timestamps


@dataclass
class TrackedObject:
    track_id: int
    class_name: str
    state: State = State.IDLE
    centroid_history: list = field(default_factory=list)
    last_centroid: Optional[tuple] = None
    last_bbox: Optional[tuple] = None
    frames_stationary: int = 0
    frames_since_actor: int = 0
    associated_actor_id: Optional[int] = None
    actor_left_frame: int = 0
    candidate_frame: int = 0
    total_frames_tracked: int = 0
    frames_missing: int = 0


@dataclass
class DumpingEvent:
    track_id: int
    class_name: str
    frame_num: int
    timestamp: float
    stationary_duration_frames: int
    actor_status: str
    centroid: tuple
    bbox: Optional[tuple] = None
    vlm: object = None  # VerificationResult from vlm_verify.py, set after creation


class TemporalEventEngine:
    def __init__(self, thresholds: Optional[Thresholds] = None):
        self.thresholds = thresholds or Thresholds()
        self.video_fps = self.thresholds.video_fps
        self.objects: dict[int, TrackedObject] = {}
        self.events: list[DumpingEvent] = []
        self.frame_num: int = 0
        self.active_actor_ids: set = set()

    def update(self, detections: list[dict], frame_num: int) -> list[DumpingEvent]:
        """
        Process one frame of detections.

        detections: list of dicts with keys:
            track_id (int), class_name (str), centroid (tuple), confidence (float)

        Returns list of new DumpingEvents triggered this frame.
        """
        self.frame_num = frame_num
        current_ids = set()
        new_events = []

        # Separate actors (persons) from objects (potential waste)
        actor_ids = set()
        object_detections = []

        for det in detections:
            tid = det["track_id"]
            current_ids.add(tid)
            cls = det["class_name"]

            if cls == "person":
                actor_ids.add(tid)
            elif cls not in VEHICLE_CLASSES:
                object_detections.append(det)

        self.active_actor_ids = actor_ids

        # Update existing tracked objects
        active_object_ids = set()
        for det in object_detections:
            tid = det["track_id"]
            active_object_ids.add(tid)
            centroid = det["centroid"]

            if tid not in self.objects:
                self.objects[tid] = TrackedObject(
                    track_id=tid,
                    class_name=det["class_name"],
                )

            obj = self.objects[tid]
            obj.total_frames_tracked += 1
            obj.frames_missing = 0
            obj.centroid_history.append(centroid)
            obj.last_centroid = centroid
            obj.last_bbox = det.get("bbox")

            # Keep only recent history to bound memory
            if len(obj.centroid_history) > 300:
                obj.centroid_history = obj.centroid_history[-300:]

            # Check if object is stationary
            is_stationary = self._is_stationary(obj.centroid_history)

            # Find nearest actor for association
            nearest_actor = self._find_nearest_actor(centroid, actor_ids, detections)

            # State transitions
            event = self._transition(obj, is_stationary, nearest_actor, frame_num)
            if event:
                new_events.append(event)

        # Handle objects that disappeared — don't reset immediately;
        # allow brief YOLO detection gaps. Only reset after prolonged absence.
        for tid in list(self.objects.keys()):
            if tid not in active_object_ids:
                obj = self.objects[tid]
                if obj.state in (State.OBSERVING, State.SUSPICIOUS):
                    obj.frames_missing += 1
                    if obj.frames_missing > self.thresholds.actor_absence_frames * 2:
                        obj.state = State.IDLE
                        obj.frames_stationary = 0
                        obj.frames_since_actor = 0
                        obj.frames_missing = 0
                elif obj.state == State.ACTOR_LEFT:
                    pass  # don't reset — waiting for persistence or movement
                elif obj.state == State.DUMPING_CANDIDATE:
                    pass  # keep candidate even if temporarily unseen

        return new_events

    def _is_stationary(self, history: list) -> bool:
        """Check if centroid has moved less than threshold over recent frames."""
        if len(history) < 5:
            return False

        recent = history[-10:] if len(history) >= 10 else history
        xs = [p[0] for p in recent]
        ys = [p[1] for p in recent]
        spread = max(xs) - min(xs) + max(ys) - min(ys)
        return spread < self.thresholds.movement_threshold

    def _find_nearest_actor(self, obj_centroid, actor_ids, detections):
        """Find the nearest actor to the object."""
        nearest_id = None
        nearest_dist = float("inf")

        for det in detections:
            if det["track_id"] in actor_ids:
                ax, ay = det["centroid"]
                ox, oy = obj_centroid
                dist = ((ax - ox) ** 2 + (ay - oy) ** 2) ** 0.5
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_id = det["track_id"]

        if nearest_dist <= self.thresholds.association_radius:
            return nearest_id
        return None

    def _transition(self, obj: TrackedObject, is_stationary: bool,
                    nearest_actor_id: Optional[int], frame_num: int) -> Optional[DumpingEvent]:
        """Apply state machine transitions. Returns DumpingEvent if triggered."""

        if obj.state == State.IDLE:
            if obj.total_frames_tracked >= self.thresholds.min_track_length:
                obj.state = State.OBSERVING

        elif obj.state == State.OBSERVING:
            if is_stationary and nearest_actor_id is not None:
                obj.state = State.SUSPICIOUS
                obj.associated_actor_id = nearest_actor_id
            elif not is_stationary:
                obj.frames_stationary = 0

        elif obj.state == State.SUSPICIOUS:
            if nearest_actor_id is not None:
                obj.associated_actor_id = nearest_actor_id
                obj.frames_since_actor = 0
            else:
                obj.frames_since_actor += 1
                if obj.frames_since_actor >= self.thresholds.actor_absence_frames:
                    obj.state = State.ACTOR_LEFT
                    obj.actor_left_frame = frame_num

        elif obj.state == State.ACTOR_LEFT:
            if is_stationary:
                obj.frames_stationary += 1
                if obj.frames_stationary >= self.thresholds.persistence_frames:
                    obj.state = State.DUMPING_CANDIDATE
                    obj.candidate_frame = frame_num
                    event = DumpingEvent(
                        track_id=obj.track_id,
                        class_name=obj.class_name,
                        frame_num=frame_num,
                        timestamp=frame_num / self.video_fps,
                        stationary_duration_frames=obj.frames_stationary,
                        actor_status="LEFT",
                        centroid=obj.last_centroid,
                        bbox=obj.last_bbox,
                    )
                    self.events.append(event)
                    return event
            else:
                # Object moved again after actor left — reset
                obj.state = State.OBSERVING
                obj.frames_stationary = 0

        elif obj.state == State.DUMPING_CANDIDATE:
            pass  # Stay in candidate state

        return None

    def get_state_summary(self) -> dict:
        """Return current state of all tracked objects."""
        summary = {}
        for tid, obj in self.objects.items():
            summary[tid] = {
                "class": obj.class_name,
                "state": obj.state.value,
                "frames_tracked": obj.total_frames_tracked,
                "frames_stationary": obj.frames_stationary,
                "associated_actor": obj.associated_actor_id,
            }
        return summary

    def reset(self):
        """Reset engine state."""
        self.objects.clear()
        self.events.clear()
        self.frame_num = 0
