"""
Action Candidate Detector — Complementary CV path for abandoned-object discovery.

Uses background subtraction + person proximity analysis to discover candidate
regions when YOLO cannot reliably detect the discarded object itself.

Assumes a mostly static camera. If the camera is moving, this module will
produce unreliable detections and should be disabled or gated by a motion check.

Separation of concerns:
  - temporal_engine.py: temporal reasoning on YOLO-tracked objects
  - action_candidate_detector.py: candidate region discovery via background change
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CandidateConfig:
    """Tuning knobs for the complementary path."""
    # Background subtractor
    bg_history: int = 500
    bg_var_threshold: int = 50
    bg_detect_shadows: bool = True
    bg_learning_rate: float = -1  # auto

    # Candidate filtering
    min_blob_area: int = 200          # pixels — ignore tiny noise
    max_blob_area: int = 50000        # pixels — ignore huge blobs (person body)
    person_overlap_ratio: float = 0.6 # if blob overlaps a person this much, ignore
    person_proximity_px: float = 150  # pixels — how close a person must be to a blob

    # Persistence
    persist_frames: int = 45          # frames blob must exist after person departure
    persist_movement_px: float = 20.0 # max centroid drift during persistence
    max_idle_frames: int = 120        # frames before candidate is evicted
    actor_absence_frames: int = 15    # frames person must be gone


@dataclass
class CandidateBlob:
    """A tracked candidate region from background change."""
    id: int
    bbox: tuple  # (x1, y1, x2, y2)
    centroid: tuple  # (cx, cy)
    area: int
    first_frame: int
    last_frame: int
    frames_active: int = 0
    frames_since_person: int = 0
    person_was_nearby: bool = False
    associated_person_id: Optional[int] = None
    centroid_history: list = field(default_factory=list)
    emitted: bool = False

    @property
    def is_persistent(self) -> bool:
        return self.frames_since_person > 0 and self.frames_active >= self.frames_since_person


class ActionCandidateDetector:
    """
    Complementary path: background-subtraction-based candidate discovery.

    Produces CandidateBlob objects with bounding boxes suitable for VLM verification.
    Does NOT perform final dumping classification — that is the VLM's role.
    """

    def __init__(self, config: Optional[CandidateConfig] = None, frame_size: tuple = (640, 480)):
        self.config = config or CandidateConfig()
        self.w, self.h = frame_size
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.bg_history,
            varThreshold=self.config.bg_var_threshold,
            detectShadows=self.config.bg_detect_shadows,
        )
        self.blobs: dict[int, CandidateBlob] = {}
        self._next_blob_id = 1
        self.frame_num = 0
        self._erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self._dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    def update(self, frame: np.ndarray, person_tracks: list[dict],
               yolo_track_ids: set[int]) -> list[CandidateBlob]:
        """
        Process one frame.

        Args:
            frame: BGR image (H, W, 3)
            person_tracks: list of dicts with keys: track_id, centroid, bbox
            yolo_track_ids: set of track IDs already handled by the YOLO pipeline
                (used to avoid double-counting)

        Returns:
            List of newly-emitted CandidateBlob objects (ready for VLM).
        """
        self.frame_num += 1
        emitted = []

        # --- Background subtraction ---
        fg_mask = self.bg_subtractor.apply(frame, learningRate=self.config.bg_learning_rate)

        # Remove shadows (shadow pixels = 127 in MOG2)
        fg_mask[fg_mask == 127] = 0

        # Clean up
        fg_mask = cv2.erode(fg_mask, self._erode_kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, self._dilate_kernel, iterations=2)

        # --- Find contours (candidate blobs) ---
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        blob_bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.config.min_blob_area or area > self.config.max_blob_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10:
                continue

            # Reject if this blob heavily overlaps with a person bbox
            bx1, by1, bx2, by2 = x, y, x + w, y + h
            if self._overlaps_person((bx1, by1, bx2, by2), person_tracks):
                continue

            blob_bboxes.append((bx1, by1, bx2, by2))

        # --- Match new blobs to existing candidates or create new ---
        active_ids = set()
        for bb in blob_bboxes:
            matched_id = self._match_to_existing(bb)
            if matched_id is not None:
                active_ids.add(matched_id)
                self._update_blob(matched_id, bb, person_tracks)
            else:
                new_id = self._create_blob(bb, person_tracks)
                active_ids.add(new_id)

        # --- Evict stale blobs ---
        stale_ids = []
        for bid, blob in self.blobs.items():
            if bid not in active_ids:
                blob.frames_since_person += 1
                blob.frames_active += 1
                if blob.frames_since_person > self.config.max_idle_frames:
                    stale_ids.append(bid)
            # Check persistence and emit
            if not blob.emitted and self._should_emit(blob):
                blob.emitted = True
                emitted.append(blob)
            # Extra stale check
            if blob.frames_active > self.config.max_idle_frames and not blob.emitted:
                stale_ids.append(bid)

        for bid in stale_ids:
            del self.blobs[bid]

        return emitted

    def _overlaps_person(self, blob_bbox: tuple, person_tracks: list[dict]) -> bool:
        """Check if a blob heavily overlaps any person bounding box."""
        bx1, by1, bx2, by2 = blob_bbox
        ba = (bx2 - bx1) * (by2 - by1)
        if ba == 0:
            return False

        for pt in person_tracks:
            if "bbox" not in pt or pt["bbox"] is None:
                continue
            px1, py1, px2, py2 = pt["bbox"]

            # Intersection
            ix1 = max(bx1, px1)
            iy1 = max(by1, py1)
            ix2 = min(bx2, px2)
            iy2 = min(by2, py2)
            if ix1 >= ix2 or iy1 >= iy2:
                continue

            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter / ba > self.config.person_overlap_ratio:
                return True

        return False

    def _match_to_existing(self, bbox: tuple) -> Optional[int]:
        """Find the existing blob whose centroid is closest to the new bbox center."""
        bx1, by1, bx2, by2 = bbox
        bcx = (bx1 + bx2) / 2
        bcy = (by1 + by2) / 2

        best_id = None
        best_dist = float("inf")

        for bid, blob in self.blobs.items():
            ocx, ocy = blob.centroid
            dist = ((bcx - ocx) ** 2 + (bcy - ocy) ** 2) ** 0.5
            # Match if centroid is within a reasonable range (roughly the max dimension of the blob)
            max_dim = max(bx2 - bx1, by2 - by1, blob.bbox[2] - blob.bbox[0], blob.bbox[3] - blob.bbox[1])
            if dist < max_dim * 1.5 and dist < best_dist:
                best_dist = dist
                best_id = bid

        return best_id

    def _create_blob(self, bbox: tuple, person_tracks: list[dict]) -> int:
        """Create a new candidate blob."""
        bx1, by1, bx2, by2 = bbox
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2
        area = (bx2 - bx1) * (by2 - by1)

        pid = self._find_nearest_person_id((cx, cy), person_tracks)
        was_nearby = pid is not None

        blob = CandidateBlob(
            id=self._next_blob_id,
            bbox=bbox,
            centroid=(cx, cy),
            area=area,
            first_frame=self.frame_num,
            last_frame=self.frame_num,
            frames_active=1,
            frames_since_person=0,
            person_was_nearby=was_nearby,
            associated_person_id=pid,
            centroid_history=[(cx, cy)],
        )
        self.blobs[self._next_blob_id] = blob
        self._next_blob_id += 1
        return blob.id

    def _update_blob(self, blob_id: int, bbox: tuple, person_tracks: list[dict]):
        """Update an existing blob's state."""
        blob = self.blobs[blob_id]
        bx1, by1, bx2, by2 = bbox
        cx = (bx1 + bx2) / 2
        cy = (by1 + by2) / 2

        blob.bbox = bbox
        blob.centroid = (cx, cy)
        blob.area = (bx2 - bx1) * (by2 - by1)
        blob.last_frame = self.frame_num
        blob.frames_active += 1
        blob.centroid_history.append((cx, cy))

        # Keep centroid history bounded
        if len(blob.centroid_history) > 150:
            blob.centroid_history = blob.centroid_history[-150:]

        # Check if a person is currently near
        pid = self._find_nearest_person_id((cx, cy), person_tracks)
        if pid is not None:
            blob.frames_since_person = 0
            blob.person_was_nearby = True
            blob.associated_person_id = pid
        else:
            blob.frames_since_person += 1

    def _find_nearest_person_id(self, centroid: tuple, person_tracks: list[dict]) -> Optional[int]:
        """Find the nearest person track to a centroid, or None if too far."""
        best_id = None
        best_dist = float("inf")

        for pt in person_tracks:
            px, py = pt["centroid"]
            dist = ((centroid[0] - px) ** 2 + (centroid[1] - py) ** 2) ** 0.5
            if dist < self.config.person_proximity_px and dist < best_dist:
                best_dist = dist
                best_id = pt["track_id"]

        return best_id

    def _should_emit(self, blob: CandidateBlob) -> bool:
        """Determine if a blob should be emitted as a candidate for VLM verification."""
        if blob.emitted:
            return False

        # Must have had a person nearby at some point
        if not blob.person_was_nearby:
            return False

        # Must have been absent from person for enough frames
        if blob.frames_since_person < self.config.actor_absence_frames:
            return False

        # Must have persisted long enough
        if blob.frames_active < self.config.persist_frames:
            return False

        # Centroid must be stable (not drifting)
        if len(blob.centroid_history) >= 10:
            recent = blob.centroid_history[-10:]
            xs = [p[0] for p in recent]
            ys = [p[1] for p in recent]
            spread = max(xs) - min(xs) + max(ys) - min(ys)
            if spread > self.config.persist_movement_px:
                return False

        return True

    def reset(self):
        """Reset all state."""
        self.blobs.clear()
        self._next_blob_id = 1
        self.frame_num = 0
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.config.bg_history,
            varThreshold=self.config.bg_var_threshold,
            detectShadows=self.config.bg_detect_shadows,
        )

    def get_summary(self) -> dict:
        """Return current state summary."""
        return {
            "active_blobs": len(self.blobs),
            "total_emitted": sum(1 for b in self.blobs.values() if b.emitted),
            "frame_num": self.frame_num,
            "blobs": {
                bid: {
                    "centroid": b.centroid,
                    "frames_active": b.frames_active,
                    "frames_since_person": b.frames_since_person,
                    "person_was_nearby": b.person_was_nearby,
                    "emitted": b.emitted,
                }
                for bid, b in self.blobs.items()
            },
        }
