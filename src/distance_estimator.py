"""
src/distance_estimator.py — Vehicle-to-vehicle distance estimation.

Two distinct measurements are handled here:

1. Vehicle-to-vehicle distance (v2v)
   The distance between the bottom-centre (ground contact) points of
   two nearby vehicles, mapped to real-world metres using the homography
   calibration if available, or pixel scale if not.

2. Distance from camera
   A rough depth estimate derived from the apparent size of a bounding
   box versus a known reference object height.  This only works when the
   user supplies a reference vehicle height and has a roughly flat road.
   It is marked as approximate and is optional.

Important: This module does NOT pretend that pixel distance equals metres.
If no calibration is available, metric estimates are clearly flagged as
unavailable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

import config
from src.calibration import HomographyCalibration, MeasurementLines, compute_pixel_scale
from src.utils import get_logger, pixel_distance

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class V2VDistanceMeasurement:
    """
    Estimated distance between two tracked vehicles.

    All distances are in real-world metres when calibration is available,
    or None (with a reason) when calibration is missing or insufficient.
    """
    track_id_a: int
    track_id_b: int
    class_name_a: str
    class_name_b: str
    distance_m: Optional[float]           # None if metric unavailable
    distance_px: float                    # always available (pixel distance)
    frame_index: int
    is_metric: bool = False               # True only when calibration is used
    note: str = ""                        # explains why metric is unavailable


@dataclass
class CameraDistanceMeasurement:
    """
    Rough distance from the camera to a single vehicle.

    Based on pinhole camera model:
        distance = (known_height_m × focal_length_px) / bbox_height_px

    This requires a known reference height and is marked as approximate.
    """
    track_id: int
    class_name: str
    distance_m: float
    bbox_height_px: int
    reference_height_m: float
    focal_length_px: float
    frame_index: int
    note: str = "Approximate — assumes flat road and known vehicle height."


# ---------------------------------------------------------------------------
# Distance estimator
# ---------------------------------------------------------------------------

class DistanceEstimator:
    """
    Estimates vehicle-to-vehicle and camera distances.

    Usage
    -----
    estimator = DistanceEstimator(homography=hom, lines=lines)
    v2v, cam = estimator.update(detections, frame_index)
    """

    def __init__(
        self,
        homography: Optional[HomographyCalibration] = None,
        lines: Optional[MeasurementLines] = None,
        focal_length_px: Optional[float] = None,
        reference_vehicle_height_m: Optional[float] = None,
    ) -> None:
        """
        Parameters
        ----------
        homography : HomographyCalibration, optional
            Four-point calibration for projecting pixels to ground-plane metres.
        lines : MeasurementLines, optional
            Two-line calibration; used as a fallback pixel scale for v2v
            distance when no homography is available.
        focal_length_px : float, optional
            Camera focal length in pixels.  Required for camera distance.
        reference_vehicle_height_m : float, optional
            Known real-world height of a reference vehicle (e.g. car ≈ 1.5 m).
        """
        self._hom = homography
        self._lines = lines
        self._focal_length_px = focal_length_px
        self._ref_height_m = reference_vehicle_height_m

        # Pre-compute pixel scale for fallback
        self._pixel_scale: Optional[float] = None
        if lines is not None and lines.is_valid:
            self._pixel_scale = compute_pixel_scale(lines)

        # Track the last known position + class per track ID
        self._last_positions: Dict[int, Tuple[int, int]] = {}
        self._last_class: Dict[int, str] = {}
        self._last_bbox_height: Dict[int, int] = {}

        # Accumulated measurements
        self._v2v_measurements: List[V2VDistanceMeasurement] = []
        self._cam_measurements: List[CameraDistanceMeasurement] = []

    # ------------------------------------------------------------------
    # Main update
    # ------------------------------------------------------------------

    def update(
        self,
        detections,
        frame_index: int,
    ) -> Tuple[List[V2VDistanceMeasurement], List[CameraDistanceMeasurement]]:
        """
        Process one frame of detections and return new distance measurements.

        Returns
        -------
        (v2v_list, camera_list) — new measurements from this frame.
        """
        new_v2v: List[V2VDistanceMeasurement] = []
        new_cam: List[CameraDistanceMeasurement] = []

        # Update position map from current detections
        current_ids = []
        for det in detections:
            if det.track_id < 0:
                continue
            self._last_positions[det.track_id] = (det.bx, det.by)
            self._last_class[det.track_id] = det.class_name
            self._last_bbox_height[det.track_id] = det.y2 - det.y1
            current_ids.append(det.track_id)

        # --- Vehicle-to-vehicle distances ---
        pairs_done = set()
        for i, tid_a in enumerate(current_ids):
            for tid_b in current_ids[i + 1:]:
                pair_key = (min(tid_a, tid_b), max(tid_a, tid_b))
                if pair_key in pairs_done:
                    continue
                pairs_done.add(pair_key)

                pos_a = self._last_positions[tid_a]
                pos_b = self._last_positions[tid_b]

                px_dist = pixel_distance(pos_a, pos_b)
                if px_dist < config.MIN_PIXEL_DISTANCE:
                    continue   # too close / overlapping; skip

                m = self._v2v_distance(
                    tid_a=tid_a,
                    tid_b=tid_b,
                    pos_a=pos_a,
                    pos_b=pos_b,
                    px_dist=px_dist,
                    frame_index=frame_index,
                )
                new_v2v.append(m)
                self._v2v_measurements.append(m)

        # --- Camera distances (optional) ---
        if self._focal_length_px and self._ref_height_m:
            for tid in current_ids:
                bbox_h = self._last_bbox_height.get(tid, 0)
                if bbox_h <= 0:
                    continue
                cam_m = self._camera_distance(tid, bbox_h, frame_index)
                if cam_m is not None:
                    new_cam.append(cam_m)
                    self._cam_measurements.append(cam_m)

        return new_v2v, new_cam

    # ------------------------------------------------------------------
    # Calculation helpers
    # ------------------------------------------------------------------

    def _v2v_distance(
        self,
        tid_a: int,
        tid_b: int,
        pos_a: Tuple[int, int],
        pos_b: Tuple[int, int],
        px_dist: float,
        frame_index: int,
    ) -> V2VDistanceMeasurement:
        """Compute metric or pixel vehicle-to-vehicle distance."""
        dist_m: Optional[float] = None
        is_metric = False
        note = ""

        # Attempt homography-based metric distance
        if self._hom is not None and self._hom.has_matrix:
            world_a = self._hom.pixel_to_world(*pos_a)
            world_b = self._hom.pixel_to_world(*pos_b)
            if world_a and world_b:
                dx = world_a[0] - world_b[0]
                dy = world_a[1] - world_b[1]
                dist_m = float(np.sqrt(dx * dx + dy * dy))
                is_metric = True
            else:
                note = "Homography transform failed for one or both points."

        # Fallback: pixel-scale from measurement lines
        elif self._pixel_scale is not None:
            dist_m = px_dist * self._pixel_scale
            is_metric = True
            note = (
                "Approximate — applies the vertical scale across the full "
                "2D pixel distance; perspective is uncorrected."
            )

        else:
            note = (
                "Metric distance unavailable.  "
                "Configure homography calibration or measurement lines "
                "to get real-world distance in metres."
            )

        return V2VDistanceMeasurement(
            track_id_a=tid_a,
            track_id_b=tid_b,
            class_name_a=self._last_class.get(tid_a, "vehicle"),
            class_name_b=self._last_class.get(tid_b, "vehicle"),
            distance_m=round(dist_m, 2) if dist_m is not None else None,
            distance_px=round(px_dist, 1),
            frame_index=frame_index,
            is_metric=is_metric,
            note=note,
        )

    def _camera_distance(
        self,
        track_id: int,
        bbox_height_px: int,
        frame_index: int,
    ) -> Optional[CameraDistanceMeasurement]:
        """
        Estimate distance from camera using the pinhole model.

            distance = (real_height × focal_length) / apparent_height_px
        """
        try:
            distance_m = (self._ref_height_m * self._focal_length_px) / bbox_height_px
        except ZeroDivisionError:
            return None

        if distance_m <= 0 or distance_m > 300:
            return None  # implausible result

        return CameraDistanceMeasurement(
            track_id=track_id,
            class_name=self._last_class.get(track_id, "vehicle"),
            distance_m=round(distance_m, 2),
            bbox_height_px=bbox_height_px,
            reference_height_m=self._ref_height_m,
            focal_length_px=self._focal_length_px,
            frame_index=frame_index,
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def v2v_measurements(self) -> List[V2VDistanceMeasurement]:
        return list(self._v2v_measurements)

    @property
    def camera_measurements(self) -> List[CameraDistanceMeasurement]:
        return list(self._cam_measurements)

    @property
    def has_metric_capability(self) -> bool:
        """True if any metric distance estimation is possible."""
        return (
            (self._hom is not None and self._hom.has_matrix)
            or self._pixel_scale is not None
        )

    def reset(self) -> None:
        """Clear all accumulated measurements."""
        self._last_positions.clear()
        self._last_class.clear()
        self._last_bbox_height.clear()
        self._v2v_measurements.clear()
        self._cam_measurements.clear()
        logger.debug("DistanceEstimator reset.")
