"""
src/utils.py — Shared helper functions used across modules.

Keeps common utilities in one place so they can be imported and tested
independently from the main pipeline.
"""

import logging
import math
import os
import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import config

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger with the project's standard format."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def new_session_id() -> str:
    """Generate a short unique session identifier (8 hex chars)."""
    return uuid.uuid4().hex[:8]


def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and any parents) if it does not exist.  Return a Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def session_output_dir(session_id: str) -> Path:
    """Return and create the output directory for *session_id*."""
    return ensure_dir(Path(config.OUTPUT_DIR) / session_id)


def validate_video_upload(
    file_name: str,
    file_size_bytes: int,
    max_size_mb: int = config.MAX_UPLOAD_SIZE_MB,
) -> Tuple[bool, str]:
    """Validate an uploaded video's extension and size before saving it."""
    allowed_extensions = {
        f".{extension}" for extension in config.SUPPORTED_VIDEO_EXTENSIONS
    }
    if Path(file_name).suffix.lower() not in allowed_extensions:
        extensions = ", ".join(sorted(allowed_extensions))
        return False, f"Unsupported video format. Use one of: {extensions}."
    if file_size_bytes <= 0:
        return False, "The selected video file is empty."
    if max_size_mb <= 0:
        return False, "The configured maximum upload size must be positive."
    if file_size_bytes > max_size_mb * 1024 * 1024:
        return False, f"Video exceeds the {max_size_mb} MB upload limit."
    return True, "Video upload looks valid."


# ---------------------------------------------------------------------------
# Video metadata
# ---------------------------------------------------------------------------

def read_video_metadata(video_path: str) -> dict:
    """
    Open a video file and return basic metadata.

    Returns a dict with keys:
        width, height, fps, frame_count, duration_seconds, codec.

    Raises FileNotFoundError if the file is missing.
    Raises ValueError if OpenCV cannot open the file.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(
            f"OpenCV could not open '{video_path}'.  "
            "The file may be corrupted or use an unsupported codec."
        )

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            fps = 25.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or frame_count <= 0:
            raise ValueError(
                f"Video '{video_path}' has invalid dimensions or contains no frames."
            )
        duration = frame_count / fps
        fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
    finally:
        cap.release()

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "codec": codec.strip(),
    }


# ---------------------------------------------------------------------------
# Drawing helpers (OpenCV)
# ---------------------------------------------------------------------------

def bgr_color_for_class(class_id: int) -> Tuple[int, int, int]:
    """Return the BGR draw color for a COCO vehicle class ID."""
    return config.CLASS_COLORS_BGR.get(class_id, config.COLOR_DEFAULT_BGR)


def draw_bounding_box(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: Tuple[int, int, int],
    label: str,
    thickness: int = 2,
) -> None:
    """Draw a filled-header bounding box with a readable label onto *frame* in-place."""
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Semi-transparent label background
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(
        label, font, font_scale, font_thickness
    )
    label_y1 = max(y1 - text_h - baseline - 4, 0)
    label_y2 = label_y1 + text_h + baseline + 4

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, label_y1), (x1 + text_w + 6, label_y2), color, -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    cv2.putText(
        frame,
        label,
        (x1 + 3, label_y2 - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA,
    )


def draw_center_point(
    frame: np.ndarray,
    cx: int,
    cy: int,
    color: Tuple[int, int, int],
    radius: int = 4,
) -> None:
    """Draw a small filled circle at the vehicle's center point."""
    cv2.circle(frame, (cx, cy), radius, color, -1)
    cv2.circle(frame, (cx, cy), radius + 1, (255, 255, 255), 1)


def draw_trail(
    frame: np.ndarray,
    trail: list,
    color: Tuple[int, int, int],
) -> None:
    """
    Draw a fading movement trail from a list of (cx, cy) points.

    The most recent point is the brightest; older points fade toward
    the panel background.
    """
    n = len(trail)
    for i in range(1, n):
        alpha = i / n  # 0 (oldest) → 1 (newest)
        pt_color = tuple(int(c * alpha) for c in color)
        cv2.line(frame, trail[i - 1], trail[i], pt_color, 1, cv2.LINE_AA)


def draw_measurement_line(
    frame: np.ndarray,
    y_pos: int,
    label: str,
    color: Tuple[int, int, int] = (80, 200, 255),
) -> None:
    """Draw a horizontal calibration / measurement line across the full frame."""
    width = frame.shape[1]
    cv2.line(frame, (0, y_pos), (width, y_pos), color, 2, cv2.LINE_AA)
    cv2.putText(
        frame,
        label,
        (8, y_pos - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Speed / unit helpers
# ---------------------------------------------------------------------------

def mps_to_kmh(speed_mps: float) -> float:
    """Convert metres per second to kilometres per hour."""
    return speed_mps * 3.6


def kmh_to_mps(speed_kmh: float) -> float:
    """Convert kilometres per hour to metres per second."""
    return speed_kmh / 3.6


def speed_status_label(speed_kmh: float, threshold: float) -> str:
    """
    Return a text status label for the given speed relative to *threshold*.

    Returns one of: "Within Limit", "Near Limit", "Above Limit".
    """
    margin = config.NEAR_THRESHOLD_MARGIN
    if speed_kmh <= threshold:
        return "Within Limit"
    elif speed_kmh <= threshold + margin:
        return "Near Limit"
    else:
        return "Above Limit"


def speed_status_color(status: str) -> str:
    """Return a Streamlit-compatible hex color string for a speed status label."""
    mapping = {
        "Within Limit": "#34EB8C",
        "Near Limit": "#F5B434",
        "Above Limit": "#EB3434",
    }
    return mapping.get(status, "#AAAAAA")


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def center_point(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    """Return the integer centre pixel of a bounding box."""
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def bottom_center(x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int]:
    """
    Return the bottom-centre pixel of a bounding box.

    Using the bottom-centre instead of the box centre gives a better
    approximation of where the vehicle contacts the road surface —
    which matters for distance and speed calculations.
    """
    return ((x1 + x2) // 2, y2)


def pixel_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """Euclidean pixel distance between two (x, y) points."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return float(np.sqrt(dx * dx + dy * dy))


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def safe_filename(name: str) -> str:
    """Replace characters that are unsafe in filenames with underscores."""
    keep = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
    return "".join(c if c in keep else "_" for c in name)


def timestamp_str() -> str:
    """Return the current local time as a compact string for filenames."""
    return time.strftime("%Y%m%d_%H%M%S")
