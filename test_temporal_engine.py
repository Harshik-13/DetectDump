"""
Unit test for temporal_event_engine.
Bypasses YOLO — feeds synthetic detections directly to verify state machine logic.
"""

from temporal_engine import TemporalEventEngine, Thresholds, State


def test_positive_case():
    """Simulate: person enters, drops object, person leaves, object persists."""
    print("=" * 60)
    print("TEST: POSITIVE CASE (dumping scenario)")
    print("=" * 60)

    thresholds = Thresholds(
        movement_threshold=15.0,
        persistence_frames=50,        # shortened for test
        actor_absence_frames=10,      # shortened for test
        association_radius=150.0,
        min_track_length=5,
    )
    engine = TemporalEventEngine(thresholds)
    events = []

    # Frame 1-5: Only person present (building track history)
    for f in range(1, 6):
        detections = [
            {"track_id": 1, "class_name": "person", "centroid": (100, 300), "confidence": 0.9},
        ]
        new_events = engine.update(detections, f)
        events.extend(new_events)

    print(f"After frame 5: {engine.get_state_summary()}")

    # Frame 6-15: Person and object both present, object stationary
    for f in range(6, 16):
        detections = [
            {"track_id": 1, "class_name": "person", "centroid": (100 + f, 300), "confidence": 0.9},
            {"track_id": 2, "class_name": "backpack", "centroid": (200, 350), "confidence": 0.8},
        ]
        new_events = engine.update(detections, f)
        events.extend(new_events)

    summary = engine.get_state_summary()
    print(f"After frame 15 (person + object): {summary.get(2, 'NOT FOUND')}")

    # Frame 16-25: Person leaves, object stays
    for f in range(16, 26):
        detections = [
            {"track_id": 2, "class_name": "backpack", "centroid": (200, 350), "confidence": 0.8},
        ]
        new_events = engine.update(detections, f)
        events.extend(new_events)

    summary = engine.get_state_summary()
    print(f"After frame 25 (person gone, object persists): {summary.get(2, 'NOT FOUND')}")

    # Frame 26-80: Object continues to persist
    for f in range(26, 81):
        detections = [
            {"track_id": 2, "class_name": "backpack", "centroid": (200, 350), "confidence": 0.8},
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
    print("\nRESULT: PASS - Dumping event correctly detected")
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

    # Frame 1-40: Person walks through with object (both moving together)
    for f in range(1, 41):
        px = 100 + f * 10  # person moves right
        bx = px + 50       # box follows person
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


def test_threshold_config():
    """Verify thresholds are configurable."""
    print("\n" + "=" * 60)
    print("TEST: THRESHOLD CONFIGURATION")
    print("=" * 60)

    t1 = Thresholds()
    t2 = Thresholds(movement_threshold=25.0, persistence_frames=100)

    print(f"Default movement_threshold: {t1.movement_threshold}")
    print(f"Custom movement_threshold:  {t2.movement_threshold}")
    print(f"Default persistence_frames: {t1.persistence_frames}")
    print(f"Custom persistence_frames:  {t2.persistence_frames}")

    assert t1.movement_threshold == 15.0
    assert t2.movement_threshold == 25.0
    assert t1.persistence_frames == 150
    assert t2.persistence_frames == 100
    print("\nRESULT: PASS - Thresholds configurable")
    return True


if __name__ == "__main__":
    test_threshold_config()
    test_positive_case()
    test_negative_case()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
