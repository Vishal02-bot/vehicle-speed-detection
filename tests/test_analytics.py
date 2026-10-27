"""Tests for analytics tables and CSV export."""

import pandas as pd

from src.analytics import detections_to_dataframe, export_csv, speeds_to_dataframe
from src.speed_estimator import SpeedMeasurement


def test_detections_dataframe_renames_columns_and_rounds_confidence():
    frame = detections_to_dataframe(
        [
            {
                "frame_index": 12,
                "track_id": 4,
                "class_name": "car",
                "confidence": 0.87654,
            }
        ]
    )

    assert frame.loc[0, "frame"] == 12
    assert frame.loc[0, "class"] == "car"
    assert frame.loc[0, "confidence"] == 0.877


def test_speed_dataframe_includes_threshold_status():
    measurement = SpeedMeasurement(
        track_id=9,
        class_name="truck",
        speed_kmh=65.0,
        speed_mps=18.0556,
        crossing_frame_1=0,
        crossing_frame_2=25,
        elapsed_seconds=1.0,
        real_distance_m=18.0556,
        timestamp_seconds=1.0,
    )

    frame = speeds_to_dataframe([measurement], speed_threshold=60.0)

    assert frame.loc[0, "track_id"] == 9
    assert frame.loc[0, "status"] == "Near Limit"


def test_csv_export_writes_dataframe(tmp_path):
    destination = tmp_path / "session" / "detections.csv"
    expected = pd.DataFrame([{"track_id": 3, "class": "car"}])

    assert export_csv(expected, destination) is True
    actual = pd.read_csv(destination)
    pd.testing.assert_frame_equal(actual, expected)
