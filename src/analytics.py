"""
src/analytics.py — Analytics and report generation.

Converts raw measurement lists into Pandas DataFrames for display
and export, and builds Plotly charts for the dashboard.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import config
from src.speed_estimator import SpeedMeasurement
from src.distance_estimator import V2VDistanceMeasurement
from src.tracker import TrailManager
from src.utils import get_logger, speed_status_label

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# DataFrames
# ---------------------------------------------------------------------------

def detections_to_dataframe(detection_records: list) -> pd.DataFrame:
    """
    Convert a list of raw detection dicts to a Pandas DataFrame.

    Each dict should have: frame_index, track_id, class_name, confidence,
    x1, y1, x2, y2, cx, cy, bx, by.
    """
    if not detection_records:
        return pd.DataFrame(columns=[
            "frame", "track_id", "class", "confidence",
            "x1", "y1", "x2", "y2", "cx", "cy",
        ])

    df = pd.DataFrame(detection_records)
    df = df.rename(columns={
        "frame_index": "frame",
        "class_name": "class",
    })
    # Round confidence for readability
    if "confidence" in df.columns:
        df["confidence"] = df["confidence"].round(3)
    return df


def speeds_to_dataframe(
    measurements: List[SpeedMeasurement],
    speed_threshold: float = float(config.DEFAULT_SPEED_THRESHOLD),
) -> pd.DataFrame:
    """Convert SpeedMeasurement objects to a DataFrame with a status column."""
    if not measurements:
        return pd.DataFrame(columns=[
            "track_id", "class", "speed_kmh", "speed_mps",
            "elapsed_s", "distance_m", "video_time_s", "status",
        ])

    rows = []
    for m in measurements:
        status = speed_status_label(m.speed_kmh, speed_threshold)
        rows.append({
            "track_id": m.track_id,
            "class": m.class_name,
            "speed_kmh": m.speed_kmh,
            "speed_mps": m.speed_mps,
            "elapsed_s": m.elapsed_seconds,
            "distance_m": m.real_distance_m,
            "video_time_s": m.timestamp_seconds,
            "status": status,
        })

    return pd.DataFrame(rows)


def distances_to_dataframe(
    measurements: List[V2VDistanceMeasurement],
) -> pd.DataFrame:
    """Convert V2VDistanceMeasurement objects to a DataFrame."""
    if not measurements:
        return pd.DataFrame(columns=[
            "frame", "track_a", "track_b", "class_a", "class_b",
            "distance_m", "distance_px", "is_metric", "estimate_type", "note",
        ])

    rows = []
    for m in measurements:
        rows.append({
            "frame": m.frame_index,
            "track_a": m.track_id_a,
            "track_b": m.track_id_b,
            "class_a": m.class_name_a,
            "class_b": m.class_name_b,
            "distance_m": m.distance_m,
            "distance_px": m.distance_px,
            "is_metric": m.is_metric,
            "estimate_type": (
                "Approximate calibrated estimate"
                if m.is_metric and "approximate" in m.note.lower()
                else "Homography estimate"
                if m.is_metric
                else "Pixels only; no metric calibration"
            ),
            "note": m.note,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def build_summary(
    trail_manager: Optional[TrailManager],
    speed_df: pd.DataFrame,
    distance_df: pd.DataFrame,
) -> dict:
    """
    Compile session statistics into a plain dict suitable for JSON export.
    """
    counts = trail_manager.count_by_class() if trail_manager else {}
    total = trail_manager.total_unique_vehicles if trail_manager else 0

    speed_stats: dict = {}
    if not speed_df.empty and "speed_kmh" in speed_df.columns:
        speed_stats = {
            "vehicles_with_speed": int(speed_df["track_id"].nunique()),
            "average_speed_kmh": round(float(speed_df["speed_kmh"].mean()), 2),
            "max_speed_kmh": round(float(speed_df["speed_kmh"].max()), 2),
            "min_speed_kmh": round(float(speed_df["speed_kmh"].min()), 2),
        }

    dist_stats: dict = {}
    metric_rows = distance_df[distance_df["is_metric"] == True] if not distance_df.empty else pd.DataFrame()
    if not metric_rows.empty and "distance_m" in metric_rows.columns:
        valid = metric_rows["distance_m"].dropna()
        dist_stats = {
            "v2v_measurements": int(len(valid)),
            "average_v2v_distance_m": round(float(valid.mean()), 2) if len(valid) else None,
            "min_v2v_distance_m": round(float(valid.min()), 2) if len(valid) else None,
        }

    return {
        "total_unique_vehicles": total,
        "vehicles_by_class": counts,
        "speed_statistics": speed_stats,
        "distance_statistics": dist_stats,
    }


# ---------------------------------------------------------------------------
# Plotly charts
# ---------------------------------------------------------------------------

CHART_THEME = "plotly_dark"
PANEL_BG = "#1E2330"
PAPER_BG = "#161B27"


def _base_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(size=14, color="#E0E6F0")),
        paper_bgcolor=PAPER_BG,
        plot_bgcolor=PANEL_BG,
        font=dict(color="#B0BAD0", size=12),
        margin=dict(l=40, r=20, t=50, b=40),
    )


def chart_vehicle_counts(trail_manager: TrailManager) -> Optional[go.Figure]:
    """Bar chart — vehicle count by class."""
    counts = trail_manager.count_by_class()
    if not counts:
        return None

    classes = list(counts.keys())
    values = [counts[c] for c in classes]
    colors = [config.CLASS_COLORS_HEX.get(c, "#6677AA") for c in classes]

    fig = go.Figure(
        go.Bar(
            x=classes,
            y=values,
            marker_color=colors,
            text=values,
            textposition="outside",
        )
    )
    fig.update_layout(
        **_base_layout("Vehicles Detected by Class"),
        xaxis=dict(title="Class", gridcolor="#2A3050"),
        yaxis=dict(title="Count", gridcolor="#2A3050", zeroline=False),
        showlegend=False,
    )
    return fig


def chart_speed_distribution(speed_df: pd.DataFrame) -> Optional[go.Figure]:
    """Histogram — speed distribution across all measurements."""
    if speed_df.empty or "speed_kmh" not in speed_df.columns:
        return None

    fig = px.histogram(
        speed_df,
        x="speed_kmh",
        nbins=20,
        color_discrete_sequence=["#4C8EF5"],
        labels={"speed_kmh": "Speed (km/h)"},
    )
    fig.update_traces(marker_line_width=0.5, marker_line_color="#2A3050")
    fig.update_layout(
        **_base_layout("Speed Distribution"),
        xaxis=dict(title="Speed (km/h)", gridcolor="#2A3050"),
        yaxis=dict(title="Count", gridcolor="#2A3050", zeroline=False),
        showlegend=False,
    )
    return fig


def chart_speed_by_track(speed_df: pd.DataFrame) -> Optional[go.Figure]:
    """Bar chart — estimated speed per vehicle track ID."""
    if speed_df.empty or "speed_kmh" not in speed_df.columns:
        return None

    df = speed_df.copy()
    df["label"] = df["class"].str.capitalize() + " #" + df["track_id"].astype(str)
    colors = [config.CLASS_COLORS_HEX.get(c, "#6677AA") for c in df["class"]]

    fig = go.Figure(
        go.Bar(
            x=df["label"],
            y=df["speed_kmh"],
            marker_color=colors,
            text=df["speed_kmh"].apply(lambda v: f"{v:.1f}"),
            textposition="outside",
        )
    )
    fig.update_layout(
        **_base_layout("Speed by Vehicle ID"),
        xaxis=dict(title="Vehicle", tickangle=-30, gridcolor="#2A3050"),
        yaxis=dict(title="Speed (km/h)", gridcolor="#2A3050", zeroline=False),
        showlegend=False,
    )
    return fig


def chart_speed_status(
    speed_df: pd.DataFrame,
    threshold: float = float(config.DEFAULT_SPEED_THRESHOLD),
) -> Optional[go.Figure]:
    """Pie chart — proportion of vehicles within / near / above threshold."""
    if speed_df.empty or "speed_kmh" not in speed_df.columns:
        return None

    status_counts = speed_df["status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]

    color_map = {
        "Within Limit": "#34EB8C",
        "Near Limit": "#F5B434",
        "Above Limit": "#EB3434",
    }
    colors = [color_map.get(s, "#AAAAAA") for s in status_counts["status"]]

    fig = go.Figure(
        go.Pie(
            labels=status_counts["status"],
            values=status_counts["count"],
            marker_colors=colors,
            hole=0.4,
        )
    )
    fig.update_layout(
        **_base_layout(f"Speed Status (threshold: {threshold:.0f} km/h)"),
        showlegend=True,
    )
    return fig


def chart_v2v_distances(distance_df: pd.DataFrame) -> Optional[go.Figure]:
    """Scatter plot — v2v distance over frame index."""
    metric_df = distance_df[distance_df["is_metric"] == True].dropna(subset=["distance_m"]) \
        if not distance_df.empty else pd.DataFrame()

    if metric_df.empty:
        return None

    metric_df = metric_df.copy()
    metric_df["pair"] = (
        "ID "
        + metric_df["track_a"].astype(str)
        + " ↔ "
        + metric_df["track_b"].astype(str)
    )

    fig = px.scatter(
        metric_df,
        x="frame",
        y="distance_m",
        color="pair",
        labels={"frame": "Frame", "distance_m": "Distance (m)"},
    )
    fig.update_layout(
        **_base_layout("Vehicle-to-Vehicle Distance Over Time"),
        xaxis=dict(title="Frame", gridcolor="#2A3050"),
        yaxis=dict(title="Distance (m)", gridcolor="#2A3050", zeroline=False),
    )
    return fig


# ---------------------------------------------------------------------------
# CSV / JSON export
# ---------------------------------------------------------------------------

def export_csv(df: pd.DataFrame, path: Path) -> bool:
    """Write a DataFrame to CSV.  Returns True on success."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info("Exported CSV: %s", path)
        return True
    except Exception as exc:
        logger.error("CSV export failed (%s): %s", path, exc)
        return False


def export_json(data: dict, path: Path) -> bool:
    """Write a dict to JSON.  Returns True on success."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Exported JSON: %s", path)
        return True
    except Exception as exc:
        logger.error("JSON export failed (%s): %s", path, exc)
        return False
