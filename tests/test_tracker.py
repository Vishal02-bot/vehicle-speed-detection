"""Tests for unique vehicle tracking record aggregation."""

from src.detector import Detection
from src.tracker import TrailManager


def _detection(track_id: int, frame_index: int, class_id: int = 2) -> Detection:
    return Detection(
        track_id=track_id,
        class_id=class_id,
        class_name={2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}[class_id],
        confidence=0.9,
        x1=10,
        y1=20,
        x2=30,
        y2=40,
        cx=20,
        cy=30,
        bx=20,
        by=40,
        frame_index=frame_index,
    )


def test_tracker_keeps_ids_across_frames_and_counts_unique_vehicles():
    tracker = TrailManager()
    tracker.update(
        [
            _detection(7, 0),
            _detection(8, 0, class_id=3),
            _detection(9, 0, class_id=5),
            _detection(10, 0, class_id=7),
        ]
    )
    tracker.update([_detection(7, 1)])

    assert tracker.total_unique_vehicles == 4
    assert tracker.count_by_class() == {
        "car": 1,
        "motorcycle": 1,
        "bus": 1,
        "truck": 1,
    }
    assert tracker.get(7).position_history == [(0, 20, 40), (1, 20, 40)]


def test_tracker_ignores_detections_without_a_tracking_id():
    tracker = TrailManager()
    tracker.update([_detection(-1, 0)])

    assert tracker.total_unique_vehicles == 0
