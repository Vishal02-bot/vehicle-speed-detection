"""Tests for frame processing settings and annotated-video output."""

import cv2
import numpy as np
import pytest

from src.video_processor import ProcessingSettings, VideoProcessor


@pytest.mark.parametrize(
    "settings",
    [
        {"confidence": -0.1},
        {"confidence": 1.1},
        {"confidence": float("nan")},
        {"frame_skip": 0},
        {"frame_skip": 1.5},
        {"frame_skip": True},
        {"speed_threshold_kmh": 0},
        {"speed_threshold_kmh": float("inf")},
    ],
)
def test_processing_settings_reject_invalid_values(settings):
    with pytest.raises(ValueError):
        ProcessingSettings(**settings)


def test_pipeline_writes_playable_annotated_video(tmp_path, monkeypatch):
    source_path = tmp_path / "source.mp4"
    output_path = tmp_path / "processed.mp4"
    writer = cv2.VideoWriter(
        str(source_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (64, 48),
    )
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV MP4 encoder is unavailable.")
    for shade in (30, 60, 90, 120):
        writer.write(np.full((48, 64, 3), shade, dtype=np.uint8))
    writer.release()

    class EmptyDetector:
        def __init__(self, **_kwargs):
            pass

        def detect(self, _frame, frame_index):
            return []

    monkeypatch.setattr("src.video_processor.VehicleDetector", EmptyDetector)
    processor = VideoProcessor(
        source_path,
        ProcessingSettings(frame_skip=2, output_path=output_path),
    )

    processed_frames = list(processor.process_video())

    assert [frame.frame_index for frame in processed_frames] == [0, 2]
    assert output_path.is_file()
    capture = cv2.VideoCapture(str(output_path))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
    ok, frame = capture.read()
    capture.release()
    assert ok
    assert frame.shape[:2] == (48, 64)
