"""
src/tracker.py — Per-vehicle trail management.

The Ultralytics tracker assigns unique IDs; this module maintains a
lightweight history for each ID so other modules can:
  - draw movement trails on the video
  - query recent positions for speed/distance calculations
  - know which class a given track ID belongs to
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import config
from src.utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Per-vehicle track record
# ---------------------------------------------------------------------------

@dataclass
class TrackRecord:
    """Accumulated information for one tracked vehicle."""
    track_id: int
    class_id: int
    class_name: str

    # Ring-buffer of (cx, cy) centre points — most recent is on the right
    trail: deque = field(
        default_factory=lambda: deque(maxlen=config.TRAIL_LENGTH)
    )

    # Ring-buffer of (bx, by) bottom-centre points (road contact)
    bottom_trail: deque = field(
        default_factory=lambda: deque(maxlen=config.TRAIL_LENGTH)
    )

    # History of (frame_index, bx, by) tuples — used by speed estimator
    position_history: List[Tuple[int, int, int]] = field(default_factory=list)

    # Latest detection confidence
    confidence: float = 0.0

    # Latest bounding-box
    last_bbox: Optional[Tuple[int, int, int, int]] = None   # (x1, y1, x2, y2)

    # Frame index of the most recent observation
    last_seen_frame: int = 0

    def update(
        self,
        cx: int,
        cy: int,
        bx: int,
        by: int,
        confidence: float,
        frame_index: int,
        bbox: Tuple[int, int, int, int],
    ) -> None:
        """Record a new observation for this track."""
        self.trail.append((cx, cy))
        self.bottom_trail.append((bx, by))
        self.position_history.append((frame_index, bx, by))
        self.confidence = confidence
        self.last_bbox = bbox
        self.last_seen_frame = frame_index

    @property
    def current_position(self) -> Optional[Tuple[int, int]]:
        """Bottom-centre of the most recent observation, or None."""
        if self.bottom_trail:
            return self.bottom_trail[-1]
        return None

    @property
    def trail_list(self) -> List[Tuple[int, int]]:
        """Ordered trail as a plain list (oldest → newest)."""
        return list(self.trail)


# ---------------------------------------------------------------------------
# Trail manager
# ---------------------------------------------------------------------------

class TrailManager:
    """
    Maintains a TrackRecord for every unique tracking ID seen so far.

    Call ``update(detections)`` once per frame; all bookkeeping is
    handled automatically.
    """

    def __init__(self) -> None:
        # Maps track_id → TrackRecord
        self._tracks: Dict[int, TrackRecord] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detections) -> None:
        """
        Incorporate a list of Detection objects from the current frame.

        Parameters
        ----------
        detections : List[Detection]
            Output from VehicleDetector.detect().
        """
        for det in detections:
            tid = det.track_id
            if tid < 0:
                logger.warning("Ignoring detection without a tracking ID.")
                continue
            if tid not in self._tracks:
                self._tracks[tid] = TrackRecord(
                    track_id=tid,
                    class_id=det.class_id,
                    class_name=det.class_name,
                )

            record = self._tracks[tid]
            record.update(
                cx=det.cx,
                cy=det.cy,
                bx=det.bx,
                by=det.by,
                confidence=det.confidence,
                frame_index=det.frame_index,
                bbox=(det.x1, det.y1, det.x2, det.y2),
            )

    def get(self, track_id: int) -> Optional[TrackRecord]:
        """Return the TrackRecord for *track_id*, or None if unknown."""
        return self._tracks.get(track_id)

    def all_records(self) -> Dict[int, TrackRecord]:
        """Return all track records (ID → TrackRecord)."""
        return dict(self._tracks)

    def unique_ids(self) -> List[int]:
        """Return all tracking IDs seen so far."""
        return list(self._tracks.keys())

    def count_by_class(self) -> Dict[str, int]:
        """Return a dict mapping class name → unique vehicle count."""
        counts: Dict[str, int] = {}
        for record in self._tracks.values():
            counts[record.class_name] = counts.get(record.class_name, 0) + 1
        return counts

    @property
    def total_unique_vehicles(self) -> int:
        """Total number of unique tracking IDs seen."""
        return len(self._tracks)

    def reset(self) -> None:
        """Clear all track records (call between video sessions)."""
        self._tracks.clear()
        logger.debug("TrailManager reset.")
