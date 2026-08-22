"""
Unit test for temporal_event_engine.
Bypasses YOLO — feeds synthetic detections directly to verify state machine logic.
"""

from unittest.mock import patch
from temporal_engine import TemporalEventEngine, Thresholds, State, WASTE_CLASSES


def test_positive_case():
    """Simulate: person enters, drops object, person leaves, object persists."""
    print("=" * 60)
    print("TEST: POSITIVE CASE (dumping scenario)")
    print("=" * 60)

    thresholds = Thresholds(
        movement_threshold=15.0,
        persistence_frames=50,
        actor_absence_frames=10,
        association_radius=150.0,
        min_track_length=5,
        video_fps=24.0,
    )
    engine = TemporalEventEngine(thresholds)
    events = []

    for f in range(1, 6):
        detections = [
            {"track_id": 1, "class_name": "person", "centroid": (100, 300), "confidence": 0.9},
        ]
        new_events = engine.update(detections, f)
        events.extend(new_events)

    print(f"After frame 5: {engine.get_state_summary()}")

    for f in range(6, 16):
        detections = [
            {"track_id": 1, "class_name": "person", "centroid": (100 + f, 300), "confidence": 0.9},
            {"track_id": 2, "class_name": "backpack", "centroid": (200, 350), "confidence": 0.8,
             "bbox": (180, 330, 220, 370)},
        ]
        new_events = engine.update(detections, f)
        events.extend(new_events)

    summary = engine.get_state_summary()
    print(f"After frame 15 (person + object): {summary.get(2, 'NOT FOUND')}")

    for f in range(16, 26):
        detections = [
            {"track_id": 2, "class_name": "backpack", "centroid": (200, 350), "confidence": 0.8,
             "bbox": (180, 330, 220, 370)},
        ]
        new_events = engine.update(detections, f)
        events.extend(new_events)

    summary = engine.get_state_summary()
    print(f"After frame 25 (person gone, object persists): {summary.get(2, 'NOT FOUND')}")

    for f in range(26, 81):
        detections = [
            {"track_id": 2, "class_name": "backpack", "centroid": (200, 350), "confidence": 0.8,
             "bbox": (180, 330, 220, 370)},
        ]
        new_events = engine.update(detections, f)
        events.extend(new_events)

    summary = engine.get_state_summary()
    obj_state = summary.get(2, {})
    print(f"After frame 80 (persistence complete): {obj_state}")

    print(f"\nEvents detected: {len(events)}")
    for e in events:
        print(f"  Track {e.track_id} ({e.class_name}): "
              f"stationary {e.stationary_duration_frames} frames, "
              f"actor {e.actor_status}")

    assert len(events) > 0, "FAIL: Expected dumping event but none detected!"
    assert events[0].track_id == 2, "FAIL: Expected event on track 2 (backpack)"
    assert events[0].actor_status == "LEFT", "FAIL: Expected actor_status LEFT"
    assert events[0].stationary_duration_frames >= 50, "FAIL: Expected sufficient stationary duration"
    assert events[0].bbox is not None, "FAIL: Expected bbox on event"
    expected_ts = events[0].frame_num / 24.0
    assert abs(events[0].timestamp - expected_ts) < 0.01, "FAIL: Expected video-relative timestamp"
    print(f"\nRESULT: PASS - Dumping event correctly detected (timestamp={events[0].timestamp:.2f}s)")
    return True


def test_negative_case():
    """Simulate: person walks through with object (no abandonment)."""
    print("\n" + "=" * 60)
    print("TEST: NEGATIVE CASE (person carrying object)")
    print("=" * 60)

    thresholds = Thresholds(
        movement_threshold=15.0,
        persistence_frames=50,
        actor_absence_frames=10,
        association_radius=150.0,
        min_track_length=5,
    )
    engine = TemporalEventEngine(thresholds)
    events = []

    for f in range(1, 41):
        px = 100 + f * 10
        bx = px + 50
        detections = [
            {"track_id": 1, "class_name": "person", "centroid": (px, 300), "confidence": 0.9},
            {"track_id": 2, "class_name": "backpack", "centroid": (bx, 320), "confidence": 0.8},
        ]
        new_events = engine.update(detections, f)
        events.extend(new_events)

    summary = engine.get_state_summary()
    print(f"After frame 40 (both moved off-screen): {summary}")

    print(f"\nEvents detected: {len(events)}")

    assert len(events) == 0, f"FAIL: Expected no events but got {len(events)}!"
    print("\nRESULT: PASS - No false positive detected")
    return True


def test_negative_sports_ball():
    """Simulate: person + sports ball, person leaves, ball persists.
    Sports ball is NOT in WASTE_CLASSES so should produce NO event."""
    print("\n" + "=" * 60)
    print("TEST: NEGATIVE CASE (sports ball)")
    print("=" * 60)

    assert "sports ball" not in WASTE_CLASSES, "FAIL: sports ball must NOT be in WASTE_CLASSES"

    thresholds = Thresholds(
        movement_threshold=15.0,
        persistence_frames=50,
        actor_absence_frames=10,
        association_radius=150.0,
        min_track_length=5,
    )
    engine = TemporalEventEngine(thresholds)
    events = []

    for f in range(1, 6):
        detections = [
            {"track_id": 1, "class_name": "person", "centroid": (100, 300), "confidence": 0.9},
        ]
        new_events = engine.update(detections, f)
        events.extend(new_events)

    for f in range(6, 16):
        detections = [
            {"track_id": 1, "class_name": "person", "centroid": (100 + f, 300), "confidence": 0.9},
            {"track_id": 2, "class_name": "sports ball", "centroid": (200, 350), "confidence": 0.8},
        ]
        new_events = engine.update(detections, f)
        events.extend(new_events)

    print(f"After frame 15 (person + sports ball): {engine.get_state_summary()}")

    for f in range(16, 80):
        detections = [
            {"track_id": 2, "class_name": "sports ball", "centroid": (200, 350), "confidence": 0.8},
        ]
        new_events = engine.update(detections, f)
        events.extend(new_events)

    summary = engine.get_state_summary()
    print(f"After frame 80 (sports ball persists alone): {summary}")

    print(f"\nEvents detected: {len(events)}")

    assert len(events) == 0, f"FAIL: Expected no events for sports ball but got {len(events)}!"
    assert 2 not in engine.objects, "FAIL: sports ball should not be tracked as waste object"
    print("\nRESULT: PASS - Sports ball correctly ignored (not in WASTE_CLASSES)")
    return True


def test_vlm_receives_candidate_bbox():
    """Verify that VLM receives bbox when a dumping event is generated."""
    print("\n" + "=" * 60)
    print("TEST: VLM RECEIVES CANDIDATE BBOX")
    print("=" * 60)

    thresholds = Thresholds(
        movement_threshold=15.0,
        persistence_frames=50,
        actor_absence_frames=10,
        association_radius=150.0,
        min_track_length=5,
    )
    engine = TemporalEventEngine(thresholds)

    for f in range(1, 6):
        engine.update([{"track_id": 1, "class_name": "person", "centroid": (100, 300), "confidence": 0.9}], f)

    for f in range(6, 16):
        engine.update([
            {"track_id": 1, "class_name": "person", "centroid": (100 + f, 300), "confidence": 0.9},
            {"track_id": 2, "class_name": "handbag", "centroid": (200, 350), "confidence": 0.8,
             "bbox": (180, 330, 220, 370)},
        ], f)

    for f in range(16, 26):
        engine.update([{"track_id": 2, "class_name": "handbag", "centroid": (200, 350), "confidence": 0.8,
                        "bbox": (180, 330, 220, 370)}], f)

    events = []
    for f in range(26, 81):
        new_events = engine.update([{"track_id": 2, "class_name": "handbag", "centroid": (200, 350),
                                     "confidence": 0.8, "bbox": (180, 330, 220, 370)}], f)
        events.extend(new_events)

    assert len(events) > 0, "FAIL: Expected dumping event"
    event = events[0]
    assert event.bbox is not None, "FAIL: Event should have bbox"
    assert event.bbox == (180, 330, 220, 370), "FAIL: bbox mismatch"

    from vlm_verify import crop_candidate_evidence
    import numpy as np
    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    crop = crop_candidate_evidence(fake_frame, event.bbox)
    assert crop.shape[0] > 0 and crop.shape[1] > 0, "FAIL: Crop should be non-empty"

    print(f"  Event bbox: {event.bbox}")
    print(f"  Crop shape: {crop.shape}")
    print("\nRESULT: PASS - VLM will receive candidate-focused evidence")
    return True


def test_threshold_config():
    """Verify thresholds are configurable."""
    print("\n" + "=" * 60)
    print("TEST: THRESHOLD CONFIGURATION")
    print("=" * 60)

    t1 = Thresholds()
    t2 = Thresholds(movement_threshold=25.0, persistence_frames=100, video_fps=60.0)

    print(f"Default movement_threshold: {t1.movement_threshold}")
    print(f"Custom movement_threshold:  {t2.movement_threshold}")
    print(f"Default persistence_frames: {t1.persistence_frames}")
    print(f"Custom persistence_frames:  {t2.persistence_frames}")
    print(f"Default video_fps: {t1.video_fps}")
    print(f"Custom video_fps:  {t2.video_fps}")

    assert t1.movement_threshold == 15.0
    assert t2.movement_threshold == 25.0
    assert t1.persistence_frames == 150
    assert t2.persistence_frames == 100
    assert t1.video_fps == 30.0
    assert t2.video_fps == 60.0
    print("\nRESULT: PASS - Thresholds configurable")
    return True


if __name__ == "__main__":
    test_threshold_config()
    test_positive_case()
    test_negative_case()
    test_negative_sports_ball()
    test_vlm_receives_candidate_bbox()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
