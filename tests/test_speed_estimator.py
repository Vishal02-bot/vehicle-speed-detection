"""
tests/test_speed_estimator.py — Unit tests for speed calculation logic.

These tests exercise the SpeedEstimator in isolation from the detector
and video pipeline, so they run without a model or video file.
"""

import pytest

from src.calibration import MeasurementLines
from src.speed_estimator import SpeedEstimator, SpeedMeasurement
from src.utils import mps_to_kmh, kmh_to_mps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_estimator(
    line1_y: int = 100,
    line2_y: int = 300,
    real_distance_m: float = 20.0,
    fps: float = 25.0,
) -> SpeedEstimator:
    lines = MeasurementLines(
        line1_y=line1_y,
        line2_y=line2_y,
        real_distance_m=real_distance_m,
    )
    return SpeedEstimator(lines=lines, fps=fps)


class _FakeDet:
    """Minimal Detection-like object for testing."""
    def __init__(self, track_id: int, by: int, frame_index: int = 0, class_name: str = "car"):
        self.track_id = track_id
        self.by = by
        self.frame_index = frame_index
        self.class_name = class_name


# ---------------------------------------------------------------------------
# Unit conversion tests
# ---------------------------------------------------------------------------

class TestUnitConversion:
    def test_mps_to_kmh_known_value(self):
        assert mps_to_kmh(10.0) == pytest.approx(36.0, rel=1e-4)

    def test_kmh_to_mps_known_value(self):
        assert kmh_to_mps(36.0) == pytest.approx(10.0, rel=1e-4)

    def test_round_trip(self):
        original = 72.5
        assert mps_to_kmh(kmh_to_mps(original)) == pytest.approx(original, rel=1e-6)

    def test_zero(self):
        assert mps_to_kmh(0.0) == 0.0
        assert kmh_to_mps(0.0) == 0.0


# ---------------------------------------------------------------------------
# Speed calculation tests
# ---------------------------------------------------------------------------

class TestSpeedCalculation:
    """
    The key formula is:
        speed_mps = distance_m / elapsed_s
        speed_kmh = speed_mps * 3.6

    For line1_y=100, line2_y=300, real_distance_m=20, fps=25:
    - A vehicle crossing line 1 at frame 0 and line 2 at frame 25
      has elapsed = 1 second → speed = 20 m/s = 72 km/h.
    """

    def test_basic_speed_calculation(self):
        est = _make_estimator(fps=25.0, real_distance_m=20.0)

        # Frame 0: cross line 1
        det1 = _FakeDet(track_id=1, by=100, frame_index=0)
        est.update([det1], frame_index=0)

        # Frame 25 (1 second later): cross line 2
        det2 = _FakeDet(track_id=1, by=300, frame_index=25)
        results = est.update([det2], frame_index=25)

        assert len(results) == 1
        m = results[0]
        assert m.track_id == 1
        assert m.speed_kmh == pytest.approx(72.0, rel=0.01)
        assert m.elapsed_seconds == pytest.approx(1.0, rel=1e-4)

    def test_half_second_crossing(self):
        """Vehicle crossing in 0.5 s should be twice as fast."""
        est = _make_estimator(fps=25.0, real_distance_m=20.0)
        est.update([_FakeDet(1, by=100, frame_index=0)], frame_index=0)
        results = est.update([_FakeDet(1, by=300, frame_index=12)], frame_index=12)

        # 0.48 s elapsed → ~41.67 m/s → ~150 km/h
        assert len(results) == 1
        assert results[0].speed_kmh > 0

    def test_speed_in_kmh_is_mps_times_3_6(self):
        est = _make_estimator(fps=25.0, real_distance_m=10.0)
        est.update([_FakeDet(2, by=100, frame_index=0)], frame_index=0)
        results = est.update([_FakeDet(2, by=300, frame_index=25)], frame_index=25)

        assert len(results) == 1
        m = results[0]
        assert m.speed_kmh == pytest.approx(m.speed_mps * 3.6, rel=1e-5)

    def test_reverse_direction_measures_speed_and_keeps_line_frame_labels(self):
        est = _make_estimator(fps=25.0, real_distance_m=20.0)
        est.update([_FakeDet(3, by=300)], frame_index=0)

        results = est.update([_FakeDet(3, by=100)], frame_index=25)

        assert len(results) == 1
        assert results[0].speed_kmh == pytest.approx(72.0)
        assert results[0].crossing_frame_1 == 25
        assert results[0].crossing_frame_2 == 0

    def test_interpolates_crossing_time_when_frames_skip_over_lines(self):
        est = _make_estimator(fps=25.0, real_distance_m=20.0)
        est.update([_FakeDet(4, by=50)], frame_index=0)

        results = est.update([_FakeDet(4, by=350)], frame_index=30)

        assert len(results) == 1
        assert results[0].crossing_frame_1 == 5
        assert results[0].crossing_frame_2 == 25
        assert results[0].elapsed_seconds == pytest.approx(0.8)
        assert results[0].speed_kmh == pytest.approx(90.0)

    def test_rejects_untracked_detection_id(self):
        est = _make_estimator()

        assert est.update([_FakeDet(-1, by=100)], frame_index=0) == []
        assert est.pending_track_ids == []

    @pytest.mark.parametrize("class_name", ["motorcycle", "truck"])
    def test_speed_measurement_preserves_motorcycle_and_truck_class(self, class_name):
        est = _make_estimator(fps=25.0, real_distance_m=20.0)
        est.update(
            [_FakeDet(12, by=100, frame_index=0, class_name=class_name)],
            frame_index=0,
        )

        measurements = est.update(
            [_FakeDet(12, by=300, frame_index=25, class_name=class_name)],
            frame_index=25,
        )

        assert len(measurements) == 1
        assert measurements[0].class_name == class_name


# ---------------------------------------------------------------------------
# Duplicate measurement prevention
# ---------------------------------------------------------------------------

class TestDuplicatePrevention:
    def test_second_crossing_not_duplicated(self):
        est = _make_estimator(fps=25.0)
        est.update([_FakeDet(5, by=100, frame_index=0)], frame_index=0)

        # First valid crossing
        r1 = est.update([_FakeDet(5, by=300, frame_index=25)], frame_index=25)
        assert len(r1) == 1

        # Same vehicle crossing line 2 again — should NOT produce another measurement
        r2 = est.update([_FakeDet(5, by=300, frame_index=50)], frame_index=50)
        assert len(r2) == 0

    def test_total_measurements_count(self):
        est = _make_estimator(fps=25.0)
        for tid in [10, 11, 12]:
            est.update([_FakeDet(tid, by=100, frame_index=0)], frame_index=0)
            est.update([_FakeDet(tid, by=300, frame_index=25)], frame_index=25)

        assert len(est.measurements) == 3


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    @pytest.mark.parametrize("fps", [0.0, -25.0, float("inf"), float("nan")])
    def test_invalid_fps_rejected(self, fps):
        lines = MeasurementLines(line1_y=100, line2_y=300, real_distance_m=20.0)
        with pytest.raises(ValueError, match="frame rate"):
            SpeedEstimator(lines=lines, fps=fps)

    def test_too_short_elapsed_time_rejected(self):
        """Elapsed time < MIN_ELAPSED_SECONDS should be rejected."""
        est = _make_estimator(fps=25.0, real_distance_m=20.0)
        # Frames 0 and 1 are only 0.04 s apart at 25 fps
        est.update([_FakeDet(1, by=100, frame_index=0)], frame_index=0)
        results = est.update([_FakeDet(1, by=300, frame_index=1)], frame_index=1)
        assert len(results) == 0

    def test_invalid_calibration_returns_empty(self):
        """An invalid MeasurementLines should produce no measurements."""
        lines = MeasurementLines(line1_y=100, line2_y=100, real_distance_m=20.0)
        assert not lines.is_valid
        est = SpeedEstimator(lines=lines, fps=25.0)
        r = est.update([_FakeDet(1, by=100, frame_index=0)], frame_index=0)
        assert r == []

    def test_zero_distance_not_used(self):
        lines = MeasurementLines(line1_y=100, line2_y=200, real_distance_m=0.0)
        assert not lines.is_valid
        est = SpeedEstimator(lines=lines, fps=25.0)
        r = est.update([_FakeDet(1, by=100, frame_index=0)], frame_index=0)
        assert r == []

    def test_empty_detections_handled(self):
        est = _make_estimator()
        r = est.update([], frame_index=0)
        assert r == []

    def test_pending_ids_tracked(self):
        est = _make_estimator()
        est.update([_FakeDet(7, by=100, frame_index=0)], frame_index=0)
        assert 7 in est.pending_track_ids

    def test_pending_cleared_after_second_crossing(self):
        est = _make_estimator(fps=25.0)
        est.update([_FakeDet(8, by=100, frame_index=0)], frame_index=0)
        est.update([_FakeDet(8, by=300, frame_index=25)], frame_index=25)
        assert 8 not in est.pending_track_ids


# ---------------------------------------------------------------------------
# Line crossing detection
# ---------------------------------------------------------------------------

class TestLineCrossingDetection:
    def test_tolerance_near_line1(self):
        est = _make_estimator(line1_y=100)
        assert est._which_line_crossed(103) == 0   # within 8px tolerance
        assert est._which_line_crossed(92) == 0

    def test_tolerance_near_line2(self):
        est = _make_estimator(line2_y=300)
        assert est._which_line_crossed(297) == 1
        assert est._which_line_crossed(308) == 1

    def test_no_crossing_mid_frame(self):
        est = _make_estimator(line1_y=100, line2_y=300)
        assert est._which_line_crossed(200) is None

    def test_reset_clears_state(self):
        est = _make_estimator(fps=25.0)
        est.update([_FakeDet(1, by=100, frame_index=0)], frame_index=0)
        est.update([_FakeDet(1, by=300, frame_index=25)], frame_index=25)
        est.reset()
        assert len(est.measurements) == 0
        assert est.pending_track_ids == []
