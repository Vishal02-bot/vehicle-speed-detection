"""Tests for COCO vehicle class filtering and detections without track IDs."""

from types import SimpleNamespace

import numpy as np

import config
from src.detector import Detection, VehicleDetector
from src.video_processor import VideoProcessor


class _FakeTensor:
    def __init__(self, values):
        self.values = values

    def int(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class _FakeBoxes:
    def __init__(self, track_ids):
        self.id = None if track_ids is None else _FakeTensor(track_ids)
        self.cls = _FakeTensor([1, 2, 3, 5, 7])
        self.conf = _FakeTensor([0.91, 0.92, 0.93, 0.94, 0.95])
        self.xyxy = _FakeTensor(
            [
                [0, 0, 10, 10],
                [10, 10, 30, 30],
                [30, 30, 50, 50],
                [50, 50, 70, 70],
                [70, 70, 90, 90],
            ]
        )


class _FakeModel:
    def __init__(self, track_ids):
        self.boxes = _FakeBoxes(track_ids)
        self.track_options = None

    def track(self, **kwargs):
        self.track_options = kwargs
        return [SimpleNamespace(boxes=self.boxes)]


def _detector(track_ids):
    detector = VehicleDetector.__new__(VehicleDetector)
    detector.confidence = 0.4
    detector._model = _FakeModel(track_ids)
    return detector


def test_detector_filters_to_supported_coco_motor_vehicle_ids():
    detector = _detector([10, 11, 12, 13, 14])

    detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert config.VEHICLE_CLASS_IDS == [2, 3, 5, 7]
    assert [(item.class_id, item.class_name, item.track_id) for item in detections] == [
        (2, "car", 11),
        (3, "motorcycle", 12),
        (5, "bus", 13),
        (7, "truck", 14),
    ]
    assert [item.confidence for item in detections] == [0.92, 0.93, 0.94, 0.95]
    assert detector._model.track_options["classes"] == [2, 3, 5, 7]
    assert "Motorcycle #12" in VideoProcessor._build_label(detections[1], None)


def test_detector_keeps_boxes_when_tracker_has_not_assigned_ids():
    detector = _detector(None)

    detections = detector.detect(np.zeros((100, 100, 3), dtype=np.uint8))

    assert [item.class_name for item in detections] == [
        "car",
        "motorcycle",
        "bus",
        "truck",
    ]
    assert all(item.track_id == -1 for item in detections)

    motorcycle = detections[1]
    label = VideoProcessor._build_label(motorcycle, None)
    assert "Motorcycle" in label
    assert "93%" in label
    assert "#" not in label


def test_untracked_detection_label_does_not_show_a_fake_track_number():
    detection = Detection(
        track_id=-1,
        class_id=7,
        class_name="truck",
        confidence=0.876,
        x1=1,
        y1=2,
        x2=30,
        y2=40,
        cx=15,
        cy=21,
        bx=15,
        by=40,
    )

    label = VideoProcessor._build_label(detection, None)

    assert label == "Truck  88%"
