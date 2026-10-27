"""
src/detector.py — Vehicle detection using Ultralytics YOLO.

Responsibilities:
  - Load the YOLO model (once, with informative error messages).
  - Run inference on a single frame.
  - Filter detections to the vehicle classes defined in config.
  - Return clean Detection objects that the rest of the pipeline can use.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

import config
from src.utils import get_logger, center_point, bottom_center

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """One vehicle detected in a single frame."""
    track_id: int           # unique tracking ID (set by tracker, -1 until tracked)
    class_id: int           # COCO class ID (2=car, 3=motorcycle, 5=bus, 7=truck)
    class_name: str         # human-readable class label
    confidence: float       # detection confidence [0, 1]
    x1: int                 # bounding box top-left x
    y1: int                 # bounding box top-left y
    x2: int                 # bounding box bottom-right x
    y2: int                 # bounding box bottom-right y
    cx: int                 # centre x of bounding box
    cy: int                 # centre y of bounding box
    bx: int                 # bottom-centre x (road contact point)
    by: int                 # bottom-centre y (road contact point)
    frame_index: int = 0    # frame number where this detection occurred


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class VehicleDetector:
    """
    Wraps a YOLO model and exposes a simple ``detect(frame)`` interface.

    Usage
    -----
    detector = VehicleDetector()               # loads default model
    detections = detector.detect(frame, fps=25, frame_index=42)
    """

    def __init__(
        self,
        model_name: str = config.YOLO_MODEL_NAME,
        confidence: float = config.DEFAULT_CONFIDENCE,
    ) -> None:
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Detection confidence must be between 0 and 1.")
        self.model_name = model_name
        self.confidence = confidence
        self._model = None
        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Load the YOLO model.

        The Ultralytics library automatically downloads the weights on
        first use if they are not found locally.  Any failure is caught
        and re-raised with a user-friendly message.
        """
        try:
            from ultralytics import YOLO  # import here to give a clear error if missing

            model_path = Path(config.MODELS_DIR) / self.model_name
            # Use local path if it exists; otherwise let Ultralytics find/download it
            load_path = str(model_path) if model_path.exists() else self.model_name

            logger.info("Loading YOLO model: %s", load_path)
            self._model = YOLO(load_path)
            logger.info("YOLO model loaded successfully.")

        except ImportError:
            raise ImportError(
                "The 'ultralytics' package is not installed.  "
                "Run:  pip install ultralytics"
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLO model '{self.model_name}'.  "
                f"Error: {exc}\n"
                "Make sure the model file exists or that you have an internet "
                "connection so Ultralytics can download the weights."
            ) from exc

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int = 0,
        tracker_config: str = config.TRACKER_TYPE,
    ) -> List[Detection]:
        """
        Run YOLO detection + ByteTrack tracking on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (as returned by cv2.VideoCapture.read).
        frame_index : int
            Frame number in the video (used to populate Detection.frame_index).
        tracker_config : str
            Ultralytics tracker YAML config name.

        Returns
        -------
        List[Detection]
            One Detection per configured vehicle class in this frame. A
            detection without an assigned tracker ID uses track_id=-1 and is
            retained for annotation but excluded from track-based estimates.
            Inference failures raise RuntimeError so the UI can report them.
        """
        if self._model is None:
            logger.error("Model is not loaded; cannot run detection.")
            return []

        if frame is None or frame.size == 0:
            logger.warning("Empty frame received at index %d; skipping.", frame_index)
            return []

        try:
            results = self._model.track(
                source=frame,
                conf=self.confidence,
                classes=config.VEHICLE_CLASS_IDS,
                tracker=tracker_config,
                persist=True,   # keep tracks alive across frames
                verbose=False,
            )
        except Exception as exc:
            logger.exception("Detection failed on frame %d.", frame_index)
            raise RuntimeError(
                f"Vehicle detection failed on frame {frame_index}: {exc}"
            ) from exc

        return self._parse_results(results, frame_index)

    # ------------------------------------------------------------------
    # Result parsing
    # ------------------------------------------------------------------

    def _parse_results(self, results, frame_index: int) -> List[Detection]:
        """Convert Ultralytics result objects into Detection dataclasses."""
        detections: List[Detection] = []

        if not results or results[0].boxes is None:
            return detections

        boxes = results[0].boxes
        class_ids = boxes.cls.int().cpu().tolist()
        track_ids = (
            boxes.id.int().cpu().tolist()
            if boxes.id is not None
            else [-1] * len(class_ids)
        )
        confidences = boxes.conf.float().cpu().tolist()
        xyxy = boxes.xyxy.int().cpu().tolist()

        for tid, cid, conf, (x1, y1, x2, y2) in zip(
            track_ids, class_ids, confidences, xyxy
        ):
            if cid not in config.VEHICLE_CLASS_IDS:
                continue  # filter: only vehicles

            cx, cy = center_point(x1, y1, x2, y2)
            bx, by = bottom_center(x1, y1, x2, y2)

            detections.append(
                Detection(
                    track_id=int(tid),
                    class_id=int(cid),
                    class_name=config.VEHICLE_CLASS_NAMES.get(int(cid), "vehicle"),
                    confidence=round(float(conf), 3),
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    cx=cx,
                    cy=cy,
                    bx=bx,
                    by=by,
                    frame_index=frame_index,
                )
            )

        return detections

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def is_ready(self) -> bool:
        """True if the model was loaded successfully."""
        return self._model is not None
