"""
tests/test_utils.py — Unit tests for shared utility functions.
"""

import pytest

from src.utils import (
    mps_to_kmh,
    kmh_to_mps,
    center_point,
    bottom_center,
    pixel_distance,
    speed_status_label,
    speed_status_color,
    safe_filename,
    new_session_id,
    validate_video_upload,
)
import config


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

class TestCenterPoint:
    def test_simple_box(self):
        assert center_point(0, 0, 100, 100) == (50, 50)

    def test_non_square_box(self):
        assert center_point(10, 20, 110, 60) == (60, 40)

    def test_integer_result(self):
        cx, cy = center_point(0, 0, 101, 101)
        assert isinstance(cx, int)
        assert isinstance(cy, int)


class TestBottomCenter:
    def test_simple_box(self):
        assert bottom_center(0, 0, 100, 100) == (50, 100)

    def test_horizontal_box(self):
        assert bottom_center(10, 30, 110, 60) == (60, 60)


class TestPixelDistance:
    def test_horizontal(self):
        assert pixel_distance((0, 0), (10, 0)) == pytest.approx(10.0)

    def test_vertical(self):
        assert pixel_distance((0, 0), (0, 10)) == pytest.approx(10.0)

    def test_3_4_5(self):
        assert pixel_distance((0, 0), (3, 4)) == pytest.approx(5.0)

    def test_same_point_is_zero(self):
        assert pixel_distance((7, 7), (7, 7)) == pytest.approx(0.0)

    def test_symmetry(self):
        assert pixel_distance((0, 0), (3, 4)) == pixel_distance((3, 4), (0, 0))


# ---------------------------------------------------------------------------
# Speed status labels
# ---------------------------------------------------------------------------

class TestSpeedStatusLabel:
    def setup_method(self):
        self.threshold = 60.0  # km/h

    def test_within_limit(self):
        assert speed_status_label(50.0, self.threshold) == "Within Limit"

    def test_exactly_at_threshold_is_within(self):
        assert speed_status_label(60.0, self.threshold) == "Within Limit"

    def test_near_limit(self):
        # 65 km/h is within 10 km/h of 60 km/h but above it
        assert speed_status_label(65.0, self.threshold) == "Near Limit"

    def test_above_limit(self):
        # 80 km/h is more than 10 km/h above 60 km/h
        assert speed_status_label(80.0, self.threshold) == "Above Limit"

    def test_zero_speed_is_within(self):
        assert speed_status_label(0.0, self.threshold) == "Within Limit"


class TestSpeedStatusColor:
    def test_within_limit_color(self):
        color = speed_status_color("Within Limit")
        assert color.startswith("#")

    def test_above_limit_color(self):
        color = speed_status_color("Above Limit")
        assert color.startswith("#")

    def test_unknown_status_returns_string(self):
        color = speed_status_color("Unknown")
        assert isinstance(color, str)


# ---------------------------------------------------------------------------
# Filename safety
# ---------------------------------------------------------------------------

class TestSafeFilename:
    def test_clean_name_unchanged(self):
        assert safe_filename("my_file-01.csv") == "my_file-01.csv"

    def test_spaces_replaced(self):
        assert " " not in safe_filename("my file name")

    def test_special_chars_replaced(self):
        result = safe_filename("test/output:file?")
        assert "/" not in result
        assert ":" not in result
        assert "?" not in result

    def test_empty_string(self):
        assert safe_filename("") == ""


# ---------------------------------------------------------------------------
# Session ID
# ---------------------------------------------------------------------------

class TestSessionId:
    def test_is_string(self):
        assert isinstance(new_session_id(), str)

    def test_length_is_8(self):
        assert len(new_session_id()) == 8

    def test_uniqueness(self):
        ids = {new_session_id() for _ in range(100)}
        # Very unlikely to have any collision
        assert len(ids) == 100


class TestValidateVideoUpload:
    def test_accepts_supported_nonempty_video(self):
        ok, _ = validate_video_upload("traffic.MP4", 1024)
        assert ok is True

    def test_rejects_unsupported_extension(self):
        ok, message = validate_video_upload("traffic.txt", 1024)
        assert ok is False
        assert "unsupported" in message.lower()

    def test_rejects_empty_video(self):
        ok, message = validate_video_upload("traffic.mp4", 0)
        assert ok is False
        assert "empty" in message.lower()

    def test_rejects_video_over_limit(self):
        ok, message = validate_video_upload("traffic.mp4", 11 * 1024 * 1024, 10)
        assert ok is False
        assert "10 MB" in message
