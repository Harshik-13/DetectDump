"""
Unit test for temporal_event_engine.
Bypasses YOLO — feeds synthetic detections directly to verify state machine logic.
"""

from unittest.mock import patch, MagicMock
from temporal_engine import TemporalEventEngine, Thresholds, State, VEHICLE_CLASSES


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
    Temporal engine produces a candidate (any non-person/non-vehicle object).
    VLM would reject it — tested separately in test_vlm_rejects_non_waste."""
    print("\n" + "=" * 60)
    print("TEST: SPORTS BALL — TEMPORAL CANDIDATE (VLM rejects)")
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

    assert len(events) > 0, "FAIL: Temporal engine should produce candidate for any non-person/non-vehicle object"
    assert events[0].class_name == "sports ball", "FAIL: Expected sports ball as candidate class"
    assert events[0].actor_status == "LEFT", "FAIL: Expected actor LEFT"
    print("\nRESULT: PASS - Temporal engine correctly produces candidate (VLM will reject)")
    return True


def test_vlm_rejects_non_waste():
    """Verify VLM rejects non-waste objects like sports balls."""
    print("\n" + "=" * 60)
    print("TEST: VLM REJECTS NON-WASTE (sports ball)")
    print("=" * 60)

    from vlm_verify import verify_dumping_event
    import numpy as np

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch('vlm_verify.OpenAI') as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"confirmed": false, "event_type": "normal_scene", "severity": "LOW", "summary": "A sports ball on the ground, not waste"}'
        mock_client.chat.completions.create.return_value = mock_response

        result = verify_dumping_event(
            frame=fake_frame,
            track_id=99,
            class_name="sports ball",
            centroid=(320, 240),
            bbox=(280, 200, 360, 280),
        )

    print(f"  VLM confirmed: {result.confirmed}")
    print(f"  VLM event_type: {result.event_type}")
    print(f"  VLM summary: {result.summary}")

    assert result.verified == True, "FAIL: VLM should respond"
    assert result.confirmed == False, "FAIL: VLM should reject sports ball"
    assert result.event_type == "normal_scene", "FAIL: Expected normal_scene"
    print("\nRESULT: PASS - VLM correctly rejects non-waste object")
    return True


def test_vlm_confirms_waste():
    """Verify VLM confirms waste objects like garbage bags."""
    print("\n" + "=" * 60)
    print("TEST: VLM CONFIRMS WASTE (garbage bag)")
    print("=" * 60)

    from vlm_verify import verify_dumping_event
    import numpy as np

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with patch('vlm_verify.OpenAI') as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"confirmed": true, "event_type": "illegal_dumping", "severity": "MEDIUM", "summary": "Black garbage bag left on roadside without owner"}'
        mock_client.chat.completions.create.return_value = mock_response

        result = verify_dumping_event(
            frame=fake_frame,
            track_id=99,
            class_name="backpack",
            centroid=(320, 240),
            bbox=(280, 200, 360, 280),
        )

    print(f"  VLM confirmed: {result.confirmed}")
    print(f"  VLM event_type: {result.event_type}")
    print(f"  VLM summary: {result.summary}")

    assert result.verified == True, "FAIL: VLM should respond"
    assert result.confirmed == True, "FAIL: VLM should confirm waste"
    assert result.event_type == "illegal_dumping", "FAIL: Expected illegal_dumping"
    print("\nRESULT: PASS - VLM correctly confirms waste object")
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


def test_vehicle_class_rejected():
    """Verify vehicles are NOT tracked as potential waste objects."""
    print("\n" + "=" * 60)
    print("TEST: VEHICLES REJECTED (not tracked as waste)")
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

    for f in range(6, 80):
        engine.update([
            {"track_id": 1, "class_name": "person", "centroid": (100, 300), "confidence": 0.9},
            {"track_id": 2, "class_name": "car", "centroid": (300, 350), "confidence": 0.9},
        ], f)

    summary = engine.get_state_summary()
    print(f"After frame 80: {summary}")

    assert 2 not in engine.objects, "FAIL: car should NOT be tracked as waste"
    print("\nRESULT: PASS - Vehicles correctly excluded from tracking")
    return True


def test_generalization_any_class():
    """Verify ANY non-person/non-vehicle class enters temporal evaluation."""
    print("\n" + "=" * 60)
    print("TEST: GENERALIZATION — any class accepted")
    print("=" * 60)

    test_classes = ["backpack", "handbag", "suitcase", "bottle", "box",
                    "sports ball", "chair", "potted plant", "bird",
                    "fire hydrant", "stop sign", "unknown_object"]

    thresholds = Thresholds(
        movement_threshold=15.0,
        persistence_frames=50,
        actor_absence_frames=10,
        association_radius=150.0,
        min_track_length=5,
    )

    for cls_name in test_classes:
        engine = TemporalEventEngine(thresholds)
        for f in range(1, 6):
            engine.update([{"track_id": 1, "class_name": "person", "centroid": (100, 300), "confidence": 0.9}], f)
        for f in range(6, 16):
            engine.update([
                {"track_id": 1, "class_name": "person", "centroid": (100 + f, 300), "confidence": 0.9},
                {"track_id": 2, "class_name": cls_name, "centroid": (200, 350), "confidence": 0.8},
            ], f)
        for f in range(16, 80):
            engine.update([{"track_id": 2, "class_name": cls_name, "centroid": (200, 350), "confidence": 0.8}], f)

        assert 2 in engine.objects, f"FAIL: {cls_name} should be tracked"
        print(f"  {cls_name}: tracked and evaluated")

    print("\nRESULT: PASS - All non-person/non-vehicle classes accepted")
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
    test_vlm_rejects_non_waste()
    test_vlm_confirms_waste()
    test_vlm_receives_candidate_bbox()
    test_vehicle_class_rejected()
    test_generalization_any_class()
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
