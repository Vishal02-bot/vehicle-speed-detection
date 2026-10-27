"""
tests/test_calibration.py — Unit tests for calibration validation logic.
"""

import pytest

from src.calibration import (
    MeasurementLines,
    HomographyCalibration,
    validate_measurement_lines,
    validate_homography_points,
    compute_pixel_scale,
)


# ---------------------------------------------------------------------------
# MeasurementLines
# ---------------------------------------------------------------------------

class TestMeasurementLines:
    def test_valid_lines(self):
        lines = MeasurementLines(line1_y=100, line2_y=300, real_distance_m=20.0)
        assert lines.is_valid is True

    def test_same_y_is_invalid(self):
        lines = MeasurementLines(line1_y=200, line2_y=200, real_distance_m=20.0)
        assert lines.is_valid is False

    def test_zero_distance_is_invalid(self):
        lines = MeasurementLines(line1_y=100, line2_y=300, real_distance_m=0.0)
        assert lines.is_valid is False

    def test_negative_distance_is_invalid(self):
        lines = MeasurementLines(line1_y=100, line2_y=300, real_distance_m=-5.0)
        assert lines.is_valid is False

    def test_lines_too_close_is_invalid(self):
        lines = MeasurementLines(line1_y=100, line2_y=102, real_distance_m=20.0)
        assert lines.is_valid is False

    def test_top_and_bottom_y(self):
        lines = MeasurementLines(line1_y=300, line2_y=100, real_distance_m=20.0)
        assert lines.top_y == 100
        assert lines.bottom_y == 300


# ---------------------------------------------------------------------------
# validate_measurement_lines
# ---------------------------------------------------------------------------

class TestValidateMeasurementLines:
    def test_valid_returns_true(self):
        lines = MeasurementLines(line1_y=50, line2_y=250, real_distance_m=15.0)
        ok, msg = validate_measurement_lines(lines)
        assert ok is True

    def test_same_y_returns_false(self):
        lines = MeasurementLines(line1_y=200, line2_y=200, real_distance_m=10.0)
        ok, _ = validate_measurement_lines(lines)
        assert ok is False

    def test_zero_distance_returns_false(self):
        lines = MeasurementLines(line1_y=100, line2_y=200, real_distance_m=0.0)
        ok, msg = validate_measurement_lines(lines)
        assert ok is False
        assert "positive" in msg.lower()

    def test_too_large_distance_returns_false(self):
        lines = MeasurementLines(line1_y=100, line2_y=400, real_distance_m=999.0)
        ok, _ = validate_measurement_lines(lines)
        assert ok is False

    def test_lines_too_close_returns_false(self):
        lines = MeasurementLines(line1_y=100, line2_y=103, real_distance_m=5.0)
        ok, _ = validate_measurement_lines(lines)
        assert ok is False


# ---------------------------------------------------------------------------
# HomographyCalibration
# ---------------------------------------------------------------------------

class TestHomographyCalibration:
    def _valid_hom(self) -> HomographyCalibration:
        # A simple rectangle in both pixel and world coords
        src = [(0, 0), (200, 0), (200, 200), (0, 200)]
        dst = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        hom = HomographyCalibration(src_points=src, dst_points=dst)
        return hom

    def test_valid_config_is_valid(self):
        hom = self._valid_hom()
        assert hom.is_valid is True

    def test_wrong_point_count_is_invalid(self):
        hom = HomographyCalibration(
            src_points=[(0, 0), (1, 0)],
            dst_points=[(0.0, 0.0)],
        )
        assert hom.is_valid is False

    def test_compute_matrix_succeeds(self):
        hom = self._valid_hom()
        ok = hom.compute_matrix()
        assert ok is True
        assert hom.has_matrix is True

    def test_pixel_to_world_after_matrix(self):
        hom = self._valid_hom()
        hom.compute_matrix()
        # The centre of the 200×200 square should map near (5, 5) in world coords
        result = hom.pixel_to_world(100, 100)
        assert result is not None
        x_w, y_w = result
        assert x_w == pytest.approx(5.0, abs=0.5)
        assert y_w == pytest.approx(5.0, abs=0.5)

    def test_pixel_to_world_without_matrix_returns_none(self):
        hom = self._valid_hom()  # matrix not computed
        result = hom.pixel_to_world(100, 100)
        assert result is None


# ---------------------------------------------------------------------------
# validate_homography_points
# ---------------------------------------------------------------------------

class TestValidateHomographyPoints:
    def test_valid_four_points(self):
        hom = HomographyCalibration(
            src_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
            dst_points=[(0.0, 0.0), (5.0, 0.0), (5.0, 5.0), (0.0, 5.0)],
        )
        ok, msg = validate_homography_points(hom)
        assert ok is True

    def test_too_few_src_points(self):
        hom = HomographyCalibration(src_points=[(0, 0)], dst_points=[])
        ok, _ = validate_homography_points(hom)
        assert ok is False

    def test_duplicate_src_points(self):
        hom = HomographyCalibration(
            src_points=[(0, 0), (0, 0), (100, 100), (200, 200)],
            dst_points=[(0, 0), (0, 0), (5, 5), (10, 10)],
        )
        ok, _ = validate_homography_points(hom)
        assert ok is False


# ---------------------------------------------------------------------------
# Pixel scale computation
# ---------------------------------------------------------------------------

class TestComputePixelScale:
    def test_correct_scale(self):
        lines = MeasurementLines(line1_y=100, line2_y=300, real_distance_m=20.0)
        scale = compute_pixel_scale(lines)
        # 20 m / 200 px = 0.1 m/px
        assert scale == pytest.approx(0.1, rel=1e-5)

    def test_invalid_lines_return_none(self):
        lines = MeasurementLines(line1_y=100, line2_y=100, real_distance_m=10.0)
        assert compute_pixel_scale(lines) is None
