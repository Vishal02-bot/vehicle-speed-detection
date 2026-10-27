"""
tests/test_distance_estimator.py — Unit tests for distance calculation logic.
"""

import pytest

from src.calibration import MeasurementLines, HomographyCalibration
from src.distance_estimator import DistanceEstimator, V2VDistanceMeasurement
from src.utils import pixel_distance


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _FakeDet:
    def __init__(
        self,
        track_id: int,
        bx: int,
        by: int,
        frame_index: int = 0,
        class_name: str = "car",
        y1: int = 0,
        y2: int = 50,
    ):
        self.track_id = track_id
        self.bx = bx
        self.by = by
        self.frame_index = frame_index
        self.class_name = class_name
        self.y1 = y1
        self.y2 = y2


# ---------------------------------------------------------------------------
# Pixel distance utility
# ---------------------------------------------------------------------------

class TestPixelDistance:
    def test_horizontal_distance(self):
        assert pixel_distance((0, 0), (10, 0)) == pytest.approx(10.0)

    def test_vertical_distance(self):
        assert pixel_distance((0, 0), (0, 10)) == pytest.approx(10.0)

    def test_diagonal_distance_3_4_5(self):
        assert pixel_distance((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_same_point(self):
        assert pixel_distance((5, 5), (5, 5)) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Distance estimator — no calibration
# ---------------------------------------------------------------------------

class TestDistanceEstimatorNoCalibration:
    def test_pixel_distance_returned_when_no_calibration(self):
        est = DistanceEstimator()
        det_a = _FakeDet(1, bx=100, by=200)
        det_b = _FakeDet(2, bx=200, by=200)
        v2v, _ = est.update([det_a, det_b], frame_index=0)
        assert len(v2v) >= 1
        assert v2v[0].distance_m is None   # no metric without calibration
        assert v2v[0].is_metric is False

    def test_note_explains_missing_calibration(self):
        est = DistanceEstimator()
        det_a = _FakeDet(1, bx=50, by=100)
        det_b = _FakeDet(2, bx=250, by=100)
        v2v, _ = est.update([det_a, det_b], frame_index=0)
        assert len(v2v) >= 1
        assert "calibration" in v2v[0].note.lower() or "unavailable" in v2v[0].note.lower()

    def test_single_vehicle_produces_no_v2v(self):
        est = DistanceEstimator()
        det_a = _FakeDet(1, bx=100, by=200)
        v2v, _ = est.update([det_a], frame_index=0)
        assert v2v == []

    def test_empty_frame_produces_no_measurements(self):
        est = DistanceEstimator()
        v2v, cam = est.update([], frame_index=0)
        assert v2v == []
        assert cam == []

    def test_untracked_detections_are_ignored(self):
        est = DistanceEstimator()
        detections = [
            _FakeDet(-1, bx=50, by=100),
            _FakeDet(-1, bx=250, by=100, class_name="truck"),
        ]

        v2v, camera = est.update(detections, frame_index=0)

        assert v2v == []
        assert camera == []


# ---------------------------------------------------------------------------
# Distance estimator — pixel-scale fallback (measurement lines)
# ---------------------------------------------------------------------------

class TestDistanceEstimatorPixelScale:
    def test_metric_distance_with_line_calibration(self):
        lines = MeasurementLines(
            line1_y=100,
            line2_y=300,   # 200 pixel span = 20 metres → 0.1 m/px
            real_distance_m=20.0,
        )
        est = DistanceEstimator(lines=lines)
        # Two vehicles 100 pixels apart horizontally
        det_a = _FakeDet(1, bx=50, by=200)
        det_b = _FakeDet(2, bx=150, by=200)
        v2v, _ = est.update([det_a, det_b], frame_index=0)
        assert len(v2v) == 1
        m = v2v[0]
        assert m.is_metric is True
        assert m.distance_m is not None
        assert m.distance_m > 0

    def test_metric_distance_is_approximate_note(self):
        lines = MeasurementLines(line1_y=100, line2_y=300, real_distance_m=20.0)
        est = DistanceEstimator(lines=lines)
        det_a = _FakeDet(1, bx=50, by=200)
        det_b = _FakeDet(2, bx=150, by=200)
        v2v, _ = est.update([det_a, det_b], frame_index=0)
        assert "approximate" in v2v[0].note.lower() or v2v[0].note == ""
        assert "perspective" in v2v[0].note.lower()
        from src.analytics import distances_to_dataframe

        frame = distances_to_dataframe(v2v)
        assert frame.loc[0, "estimate_type"] == "Approximate calibrated estimate"


# ---------------------------------------------------------------------------
# Minimum pixel distance filter
# ---------------------------------------------------------------------------

class TestMinimumPixelDistance:
    def test_overlapping_vehicles_skipped(self):
        """Vehicles very close together (< MIN_PIXEL_DISTANCE) should be skipped."""
        est = DistanceEstimator()
        det_a = _FakeDet(1, bx=100, by=100)
        det_b = _FakeDet(2, bx=101, by=100)   # only 1 px apart
        v2v, _ = est.update([det_a, det_b], frame_index=0)
        assert v2v == []


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------

class TestDistanceEstimatorReset:
    def test_reset_clears_measurements(self):
        lines = MeasurementLines(line1_y=100, line2_y=300, real_distance_m=20.0)
        est = DistanceEstimator(lines=lines)
        det_a = _FakeDet(1, bx=50, by=200)
        det_b = _FakeDet(2, bx=250, by=200)
        est.update([det_a, det_b], frame_index=0)
        assert len(est.v2v_measurements) > 0
        est.reset()
        assert len(est.v2v_measurements) == 0
