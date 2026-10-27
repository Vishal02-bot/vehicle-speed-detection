"""
src/speed_estimator.py — Vehicle speed estimation.

Algorithm
---------
1. Two horizontal measurement lines are drawn across the road in the
   calibration step.  The user records the real-world distance between
   those lines (e.g. 20 metres).

2. For every tracked vehicle, we note the precise timestamp (video time,
   derived from frame index ÷ FPS) when the vehicle's bottom-centre
   point crosses each line.

3. Speed is then:
       speed_mps = real_distance_m / elapsed_seconds
       speed_kmh = speed_mps × 3.6

4. Only one speed measurement per vehicle per line-pair is recorded
   (duplicate prevention).  Measurements with unrealistic values or
   very short elapsed times are rejected.

This module is intentionally separate from the UI and from distance
estimation so it can be unit-tested independently.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import config
from src.calibration import MeasurementLines
from src.utils import get_logger, mps_to_kmh

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SpeedMeasurement:
    """One completed speed measurement for a tracked vehicle."""
    track_id: int
    class_name: str
    speed_kmh: float
    speed_mps: float
    crossing_frame_1: int           # frame index of first line crossing
    crossing_frame_2: int           # frame index of second line crossing
    elapsed_seconds: float
    real_distance_m: float
    timestamp_seconds: float = 0.0  # video time of second crossing (seconds)


@dataclass
class LineCrossingEvent:
    """Records the moment a vehicle crosses one measurement line."""
    track_id: int
    line_index: int     # 0 = first line, 1 = second line
    frame_index: int
    video_time_s: float


# ---------------------------------------------------------------------------
# Speed estimator
# ---------------------------------------------------------------------------

class SpeedEstimator:
    """
    Estimates vehicle speed from line-crossing timestamps.

    Usage
    -----
    estimator = SpeedEstimator(lines, fps)
    for each frame:
        new_measurements = estimator.update(detections, frame_index)
    all_measurements = estimator.measurements
    """

    def __init__(
        self,
        lines: MeasurementLines,
        fps: float,
    ) -> None:
        """
        Parameters
        ----------
        lines : MeasurementLines
            Calibrated measurement-line positions and real-world distance.
        fps : float
            Video frames per second (used to convert frame index → time).
        """
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video frame rate must be a positive finite number.")
        self.lines = lines
        self.fps = fps

        # Pending first-line crossings waiting for their second crossing.
        # Maps track_id → LineCrossingEvent
        self._pending: Dict[int, LineCrossingEvent] = {}

        # Completed measurements
        self._measurements: List[SpeedMeasurement] = []

        # Set of track IDs that have already produced a measurement
        # (prevents the same vehicle triggering duplicate recordings)
        self._measured_ids: set = set()

        # Crossing proximity tolerance in pixels — how close to a line
        # the bottom-centre must be to count as a crossing
        self._tolerance_px: int = 8
        self._last_positions: Dict[int, Tuple[int, int]] = {}

    # ------------------------------------------------------------------
    # Main update method
    # ------------------------------------------------------------------

    def update(
        self,
        detections,          # List[Detection] from detector
        frame_index: int,
        track_class_map: Optional[Dict[int, str]] = None,
    ) -> List[SpeedMeasurement]:
        """
        Process detections from a single frame and return any new speed
        measurements that became available in this frame.

        Parameters
        ----------
        detections : List[Detection]
            Vehicle detections for the current frame.
        frame_index : int
            Index of the current frame in the video.
        track_class_map : dict, optional
            Mapping of track_id → class_name (used when class_name is
            missing from the detection).

        Returns
        -------
        List[SpeedMeasurement]
            Speed measurements completed during this call.
        """
        if not self.lines.is_valid:
            return []

        new_measurements: List[SpeedMeasurement] = []

        for det in detections:
            tid = det.track_id
            if tid < 0:
                continue
            by = det.by   # bottom-centre Y coordinate
            class_name = det.class_name
            if track_class_map and tid in track_class_map:
                class_name = track_class_map[tid]

            previous = self._last_positions.get(tid)
            self._last_positions[tid] = (frame_index, by)
            events = self._crossed_lines(tid, previous, frame_index, by)
            for event in events:
                if tid in self._measured_ids:
                    break

                pending = self._pending.get(tid)
                if pending is None or pending.line_index == event.line_index:
                    self._pending[tid] = event
                    continue

                self._pending.pop(tid, None)
                measurement = self._compute_speed(
                    tid=tid,
                    class_name=class_name,
                    first_event=pending,
                    second_event=event,
                )
                if measurement is None:
                    self._pending[tid] = event
                    continue

                self._measurements.append(measurement)
                self._measured_ids.add(tid)
                new_measurements.append(measurement)
                logger.info(
                    "Speed measured for track %d: %.1f km/h",
                    tid,
                    measurement.speed_kmh,
                )

        self._expire_pending(frame_index)

        return new_measurements

    def _crossed_lines(
        self,
        track_id: int,
        previous: Optional[Tuple[int, int]],
        frame_index: int,
        current_y: int,
    ) -> List[LineCrossingEvent]:
        """Find line crossings, interpolating between observed frame positions."""
        lines = (self.lines.line1_y, self.lines.line2_y)
        if previous is None:
            line_index = self._which_line_crossed(current_y)
            if line_index is None:
                return []
            return [
                LineCrossingEvent(
                    track_id=track_id,
                    line_index=line_index,
                    frame_index=frame_index,
                    video_time_s=frame_index / self.fps,
                )
            ]

        previous_frame, previous_y = previous
        if frame_index <= previous_frame or current_y == previous_y:
            return []

        crossings: List[LineCrossingEvent] = []
        for line_index, line_y in enumerate(lines):
            crossed = (
                previous_y < line_y <= current_y
                or previous_y > line_y >= current_y
            )
            entered_tolerance = (
                abs(current_y - line_y) <= self._tolerance_px
                and abs(previous_y - line_y) > self._tolerance_px
            )
            if not crossed and not entered_tolerance:
                continue

            if crossed:
                fraction = (line_y - previous_y) / (current_y - previous_y)
                crossing_frame = previous_frame + (
                    frame_index - previous_frame
                ) * fraction
            else:
                crossing_frame = float(frame_index)

            crossings.append(
                LineCrossingEvent(
                    track_id=track_id,
                    line_index=line_index,
                    frame_index=round(crossing_frame),
                    video_time_s=crossing_frame / self.fps,
                )
            )

        crossings.sort(key=lambda event: event.video_time_s)
        return crossings

    def _expire_pending(self, frame_index: int) -> None:
        """Discard unfinished crossings after a long gap or invalid frame order."""
        max_age_frames = max(round(self.fps * 120), 1)
        expired_ids = [
            track_id
            for track_id, event in self._pending.items()
            if frame_index < event.frame_index
            or frame_index - event.frame_index > max_age_frames
        ]
        for track_id in expired_ids:
            self._pending.pop(track_id, None)
            self._last_positions.pop(track_id, None)

    # ------------------------------------------------------------------
    # Speed calculation
    # ------------------------------------------------------------------

    def _compute_speed(
        self,
        tid: int,
        class_name: str,
        first_event: LineCrossingEvent,
        second_event: LineCrossingEvent,
    ) -> Optional[SpeedMeasurement]:
        """
        Calculate speed from two crossing events.

        Returns a SpeedMeasurement or None if the inputs are invalid.
        """
        elapsed = abs(second_event.video_time_s - first_event.video_time_s)

        if elapsed < config.MIN_ELAPSED_SECONDS:
            logger.warning(
                "Track %d: elapsed time %.4fs is too short — measurement rejected.",
                tid,
                elapsed,
            )
            return None

        if self.lines.real_distance_m <= 0:
            logger.error("Real-world distance is not positive — cannot compute speed.")
            return None

        speed_mps = self.lines.real_distance_m / elapsed
        speed_kmh = mps_to_kmh(speed_mps)

        if speed_kmh < config.MIN_SPEED_KMH:
            logger.debug(
                "Track %d speed %.1f km/h below minimum — skipping.", tid, speed_kmh
            )
            return None

        if speed_kmh > config.MAX_SPEED_KMH:
            logger.warning(
                "Track %d speed %.1f km/h exceeds maximum — measurement rejected.",
                tid,
                speed_kmh,
            )
            return None

        crossing_frames = {
            first_event.line_index: first_event.frame_index,
            second_event.line_index: second_event.frame_index,
        }
        return SpeedMeasurement(
            track_id=tid,
            class_name=class_name,
            speed_kmh=round(speed_kmh, 2),
            speed_mps=round(speed_mps, 4),
            crossing_frame_1=crossing_frames[0],
            crossing_frame_2=crossing_frames[1],
            elapsed_seconds=round(elapsed, 4),
            real_distance_m=self.lines.real_distance_m,
            timestamp_seconds=round(
                max(first_event.video_time_s, second_event.video_time_s), 2
            ),
        )

    # ------------------------------------------------------------------
    # Line-crossing detection
    # ------------------------------------------------------------------

    def _which_line_crossed(self, by: int) -> Optional[int]:
        """
        Determine whether pixel Y *by* is near line 1 (returns 0),
        line 2 (returns 1), or neither (returns None).
        """
        tol = self._tolerance_px
        if abs(by - self.lines.line1_y) <= tol:
            return 0
        if abs(by - self.lines.line2_y) <= tol:
            return 1
        return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def measurements(self) -> List[SpeedMeasurement]:
        """All completed speed measurements."""
        return list(self._measurements)

    @property
    def pending_track_ids(self) -> List[int]:
        """Track IDs that have crossed line 1 but not yet line 2."""
        return list(self._pending.keys())

    def get_speed_for_track(self, track_id: int) -> Optional[SpeedMeasurement]:
        """Return the speed measurement for a specific track ID, if available."""
        for m in reversed(self._measurements):
            if m.track_id == track_id:
                return m
        return None

    def reset(self) -> None:
        """Clear all state (call between video sessions)."""
        self._pending.clear()
        self._last_positions.clear()
        self._measurements.clear()
        self._measured_ids.clear()
        logger.debug("SpeedEstimator reset.")
