"""
src/calibration.py — Road-plane calibration and homography.

Speed estimation requires knowing how far apart two measurement lines
are in real-world metres.  Distance estimation requires mapping pixel
coordinates to real-world ground-plane coordinates.

This module provides:
  - CalibrationConfig  — stores the user's calibration parameters
  - validate_calibration()  — checks that the config is sensible
  - pixel_to_world()  — maps a pixel coordinate to metres using homography
  - build_homography()  — computes the perspective transform matrix
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MeasurementLines:
    """
    Two horizontal lines drawn across the road, used for speed estimation.

    line1_y and line2_y are pixel Y-coordinates in the video frame.
    real_distance_m is the actual distance between those two lines in metres
    (measured on the real road).
    """
    line1_y: int = 0          # pixel Y of the first measurement line
    line2_y: int = 0          # pixel Y of the second measurement line
    real_distance_m: float = 0.0   # real-world distance in metres

    @property
    def is_valid(self) -> bool:
        """True when the lines are distinct and the distance is positive."""
        return (
            self.line1_y != self.line2_y
            and self.real_distance_m > 0.0
            and abs(self.line1_y - self.line2_y) >= 5  # at least 5 px apart
        )

    @property
    def top_y(self) -> int:
        """The smaller (upper) Y value."""
        return min(self.line1_y, self.line2_y)

    @property
    def bottom_y(self) -> int:
        """The larger (lower) Y value."""
        return max(self.line1_y, self.line2_y)


@dataclass
class HomographyCalibration:
    """
    Four point-pairs used to build a perspective homography.

    src_points: four pixel coordinates in the video frame, arranged as
                [top-left, top-right, bottom-right, bottom-left].
    dst_points: the corresponding real-world ground-plane coordinates in
                metres (same ordering).

    When dst_points is not provided the homography cannot be computed and
    metric distance estimation is unavailable.
    """
    src_points: List[Tuple[int, int]] = field(default_factory=list)
    dst_points: List[Tuple[float, float]] = field(default_factory=list)
    _matrix: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def is_valid(self) -> bool:
        return (
            len(self.src_points) == 4
            and len(self.dst_points) == 4
        )

    def compute_matrix(self) -> bool:
        """
        Compute the homography matrix from the stored point pairs.

        Returns True on success, False if the computation fails.
        After a successful call, use pixel_to_world() for coordinate mapping.
        """
        if not self.is_valid:
            logger.warning("Cannot compute homography: need exactly 4 point pairs.")
            return False

        src = np.array(self.src_points, dtype=np.float32)
        dst = np.array(self.dst_points, dtype=np.float32)

        matrix, mask = cv2.findHomography(src, dst)
        if matrix is None:
            logger.error("cv2.findHomography returned None (degenerate point set?).")
            self._matrix = None
            return False

        self._matrix = matrix
        logger.info("Homography matrix computed successfully.")
        return True

    def pixel_to_world(
        self, px: int, py: int
    ) -> Optional[Tuple[float, float]]:
        """
        Map a single pixel (px, py) to real-world ground-plane metres.

        Returns (x_metres, y_metres) or None if the matrix is not computed.
        """
        if self._matrix is None:
            return None

        pt = np.array([[[float(px), float(py)]]], dtype=np.float32)
        world = cv2.perspectiveTransform(pt, self._matrix)
        x_w = float(world[0, 0, 0])
        y_w = float(world[0, 0, 1])
        return (x_w, y_w)

    @property
    def has_matrix(self) -> bool:
        return self._matrix is not None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_measurement_lines(lines: MeasurementLines) -> Tuple[bool, str]:
    """
    Check that measurement-line calibration data is usable.

    Returns (ok: bool, message: str).
    """
    if lines.line1_y == lines.line2_y:
        return False, "The two measurement lines must be at different Y positions."

    if abs(lines.line1_y - lines.line2_y) < 5:
        return False, "The measurement lines are too close together (< 5 pixels)."

    if lines.real_distance_m <= 0.0:
        return False, "Real-world distance must be a positive number of metres."

    if lines.real_distance_m > 500.0:
        return (
            False,
            "Real-world distance seems too large (> 500 m).  "
            "Please double-check your measurement.",
        )

    return True, "Calibration looks valid."


def validate_homography_points(hom: HomographyCalibration) -> Tuple[bool, str]:
    """
    Basic sanity check on the four-point homography calibration.

    Returns (ok: bool, message: str).
    """
    if len(hom.src_points) != 4:
        return False, f"Exactly 4 source points required (got {len(hom.src_points)})."

    if len(hom.dst_points) != 4:
        return False, f"Exactly 4 destination points required (got {len(hom.dst_points)})."

    # Check for duplicate points
    if len(set(hom.src_points)) < 4:
        return False, "Source points must all be distinct."

    return True, "Homography points look valid."


# ---------------------------------------------------------------------------
# Pixel-scale conversion (simple fallback)
# ---------------------------------------------------------------------------

def compute_pixel_scale(lines: MeasurementLines) -> Optional[float]:
    """
    Compute a vertical pixel-to-metre scale factor from measurement lines.

    This is a simplified approach that assumes the road surface is
    approximately flat between the two lines. It is used for speed
    estimation and as an approximate scale for vehicle-to-vehicle distance.
    It does not compensate for perspective distortion.

    Returns metres-per-pixel (float) or None if calibration is invalid.
    """
    if not lines.is_valid:
        return None

    pixel_span = abs(lines.line2_y - lines.line1_y)
    if pixel_span == 0:
        return None

    return lines.real_distance_m / pixel_span
