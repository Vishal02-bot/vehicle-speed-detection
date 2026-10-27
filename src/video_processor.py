"""
src/video_processor.py — Frame-by-frame video processing pipeline.

This module ties together the detector, tracker, speed estimator,
and distance estimator into a single generator-based pipeline.

Processing is incremental: each call to process_video() yields one
ProcessedFrame at a time so the Streamlit UI can update a progress
bar without loading the entire video into memory.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional

import cv2
import numpy as np

import config
from src.calibration import MeasurementLines, HomographyCalibration
from src.detector import VehicleDetector, Detection
from src.distance_estimator import DistanceEstimator, V2VDistanceMeasurement
from src.speed_estimator import SpeedEstimator, SpeedMeasurement
from src.tracker import TrailManager
from src.utils import (
    get_logger,
    bgr_color_for_class,
    draw_bounding_box,
    draw_center_point,
    draw_trail,
    draw_measurement_line,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-frame output
# ---------------------------------------------------------------------------

@dataclass
class ProcessedFrame:
    """Everything the UI needs from one processed video frame."""
    frame_index: int
    annotated_frame: np.ndarray          # BGR frame with overlays drawn
    detections: List[Detection]          # detections in this frame
    new_speeds: List[SpeedMeasurement]   # speeds completed in this frame
    new_distances: List[V2VDistanceMeasurement]  # v2v distances this frame
    progress: float                      # 0.0 – 1.0 completion


# ---------------------------------------------------------------------------
# Processing settings
# ---------------------------------------------------------------------------

@dataclass
class ProcessingSettings:
    """User-configurable parameters for a single processing run."""
    confidence: float = config.DEFAULT_CONFIDENCE
    frame_skip: int = config.DEFAULT_FRAME_SKIP
    lines: Optional[MeasurementLines] = None
    homography: Optional[HomographyCalibration] = None
    focal_length_px: Optional[float] = None
    reference_vehicle_height_m: Optional[float] = None
    speed_threshold_kmh: float = float(config.DEFAULT_SPEED_THRESHOLD)
    draw_trails: bool = True
    draw_lines: bool = True
    output_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection confidence must be between 0 and 1.")
        if (
            isinstance(self.frame_skip, bool)
            or not isinstance(self.frame_skip, int)
            or self.frame_skip < 1
        ):
            raise ValueError("Frame skip must be a positive integer.")
        if (
            not math.isfinite(self.speed_threshold_kmh)
            or self.speed_threshold_kmh <= 0
        ):
            raise ValueError("Speed threshold must be a positive finite number.")


# ---------------------------------------------------------------------------
# Main processor
# ---------------------------------------------------------------------------

class VideoProcessor:
    """
    Orchestrates the full detection-tracking-estimation pipeline.

    Parameters
    ----------
    video_path : str | Path
        Path to the source video file.
    settings : ProcessingSettings
        User-defined configuration for this run.
    """

    def __init__(
        self,
        video_path: str | Path,
        settings: ProcessingSettings,
    ) -> None:
        self.video_path = str(video_path)
        self.settings = settings

        # Lazy-loaded resources
        self._detector: Optional[VehicleDetector] = None
        self._trail_manager: Optional[TrailManager] = None
        self._speed_estimator: Optional[SpeedEstimator] = None
        self._distance_estimator: Optional[DistanceEstimator] = None

        # Video metadata (populated on first open)
        self.fps: float = 25.0
        self.total_frames: int = 0
        self.frame_width: int = 0
        self.frame_height: int = 0
        self._output_writer: Optional[cv2.VideoWriter] = None

    # ------------------------------------------------------------------
    # Pipeline entry point
    # ------------------------------------------------------------------

    def process_video(self) -> Generator[ProcessedFrame, None, None]:
        """
        Generator that processes the video frame by frame.

        Yields one ProcessedFrame per processed frame.  The caller
        (typically the Streamlit app) iterates this to update UI state.

        Raises RuntimeError if the video cannot be opened or the
        detector fails to load.
        """
        cap = self._open_video()
        if cap is None:
            raise RuntimeError(
                f"Could not open video: '{self.video_path}'.  "
                "The file may be corrupted or use an unsupported codec."
            )

        try:
            self._init_pipeline()
            self._open_output_video()
            yield from self._frame_loop(cap)
        finally:
            if self._output_writer is not None:
                self._output_writer.release()
                self._output_writer = None
            cap.release()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _open_video(self) -> Optional[cv2.VideoCapture]:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if math.isfinite(fps) and fps > 0 else 25.0
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return cap

    def _open_output_video(self) -> None:
        """Create an MP4 writer when the caller requested an annotated video."""
        if self.settings.output_path is None:
            return

        output_path = Path(self.settings.output_path)
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise RuntimeError("The input video has invalid frame dimensions.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_fps = self.fps / self.settings.frame_skip
        self._output_writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (self.frame_width, self.frame_height),
        )
        if not self._output_writer.isOpened():
            self._output_writer.release()
            self._output_writer = None
            raise RuntimeError(
                f"OpenCV could not create the annotated video at '{output_path}'. "
                "Check that the MP4 video encoder is available."
            )

    def _init_pipeline(self) -> None:
        """Create all processing components with the current settings."""
        self._detector = VehicleDetector(confidence=self.settings.confidence)

        self._trail_manager = TrailManager()

        if self.settings.lines and self.settings.lines.is_valid:
            self._speed_estimator = SpeedEstimator(
                lines=self.settings.lines,
                fps=self.fps,
            )
        else:
            self._speed_estimator = None

        self._distance_estimator = DistanceEstimator(
            homography=self.settings.homography,
            lines=self.settings.lines,
            focal_length_px=self.settings.focal_length_px,
            reference_vehicle_height_m=self.settings.reference_vehicle_height_m,
        )

    # ------------------------------------------------------------------
    # Frame loop
    # ------------------------------------------------------------------

    def _frame_loop(
        self, cap: cv2.VideoCapture
    ) -> Generator[ProcessedFrame, None, None]:
        """Iterate through all frames and yield ProcessedFrame objects."""
        frame_index = 0
        processed_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Optional frame skipping to speed up processing
            if frame_index % max(self.settings.frame_skip, 1) != 0:
                frame_index += 1
                continue

            # --- Detection + tracking ---
            detections = self._detector.detect(frame, frame_index=frame_index)
            self._trail_manager.update(detections)
            tracked_detections = [
                detection for detection in detections if detection.track_id >= 0
            ]

            # --- Speed estimation ---
            new_speeds: List[SpeedMeasurement] = []
            if self._speed_estimator is not None:
                new_speeds = self._speed_estimator.update(
                    tracked_detections, frame_index
                )

            # --- Distance estimation ---
            new_v2v, _ = self._distance_estimator.update(
                tracked_detections, frame_index
            )

            # --- Annotation ---
            annotated = self._annotate_frame(frame, detections, frame_index)
            if self._output_writer is not None:
                self._output_writer.write(annotated)

            # --- Progress ---
            progress = (
                frame_index / self.total_frames
                if self.total_frames > 0
                else 0.0
            )

            yield ProcessedFrame(
                frame_index=frame_index,
                annotated_frame=annotated,
                detections=detections,
                new_speeds=new_speeds,
                new_distances=new_v2v,
                progress=min(progress, 1.0),
            )

            frame_index += 1
            processed_count += 1

        logger.info(
            "Processing complete: %d frames processed (total frames: %d).",
            processed_count,
            self.total_frames,
        )

    # ------------------------------------------------------------------
    # Frame annotation
    # ------------------------------------------------------------------

    def _annotate_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        frame_index: int,
    ) -> np.ndarray:
        """Draw all overlays onto a copy of the frame and return it."""
        annotated = frame.copy()

        # Draw measurement lines
        if self.settings.draw_lines and self.settings.lines and self.settings.lines.is_valid:
            draw_measurement_line(annotated, self.settings.lines.line1_y, "Line 1")
            draw_measurement_line(
                annotated,
                self.settings.lines.line2_y,
                "Line 2",
                color=(80, 255, 180),
            )

        # Draw each detection
        for det in detections:
            color = bgr_color_for_class(det.class_id)
            record = self._trail_manager.get(det.track_id)

            # Trail
            if self.settings.draw_trails and record:
                draw_trail(annotated, record.trail_list, color)

            # Bounding box + label
            speed_m = None
            if self._speed_estimator:
                speed_m = self._speed_estimator.get_speed_for_track(det.track_id)

            label = self._build_label(det, speed_m)
            draw_bounding_box(
                annotated,
                det.x1,
                det.y1,
                det.x2,
                det.y2,
                color,
                label,
            )
            draw_center_point(annotated, det.bx, det.by, color)

        # Frame counter overlay (top-right)
        cv2.putText(
            annotated,
            f"Frame {frame_index}",
            (annotated.shape[1] - 120, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
            cv2.LINE_AA,
        )

        return annotated

    @staticmethod
    def _build_label(det: Detection, speed_m: Optional[SpeedMeasurement]) -> str:
        """Build a compact, readable overlay label for one detection."""
        label = det.class_name.capitalize()
        if det.track_id >= 0:
            label += f" #{det.track_id}"
        parts = [label, f"{det.confidence:.0%}"]
        if speed_m:
            parts.append(f"Est. {speed_m.speed_kmh:.0f} km/h")
        return "  ".join(parts)

    # ------------------------------------------------------------------
    # Convenience properties (available after processing)
    # ------------------------------------------------------------------

    @property
    def all_speeds(self) -> List[SpeedMeasurement]:
        if self._speed_estimator:
            return self._speed_estimator.measurements
        return []

    @property
    def all_distances(self) -> List[V2VDistanceMeasurement]:
        if self._distance_estimator:
            return self._distance_estimator.v2v_measurements
        return []

    @property
    def trail_manager(self) -> Optional[TrailManager]:
        return self._trail_manager
