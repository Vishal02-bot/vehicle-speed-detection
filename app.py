"""
app.py — Vehicle Vision: Speed & Distance Monitor

Streamlit application entry point.

Run with:
    streamlit run app.py

The UI is organised into four pages (Streamlit sidebar tabs):
  1. Dashboard     — session overview and statistics
  2. Analyze Video — upload, calibrate, and process
  3. Results       — charts and data tables
  4. About         — project identity and documentation
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

import config
from src import auth_store
from src.analytics import (
    build_summary,
    chart_speed_distribution,
    chart_speed_by_track,
    chart_speed_status,
    chart_v2v_distances,
    chart_vehicle_counts,
    detections_to_dataframe,
    distances_to_dataframe,
    export_csv,
    export_json,
    speeds_to_dataframe,
)
from src.calibration import (
    HomographyCalibration,
    MeasurementLines,
    validate_measurement_lines,
)
from src.utils import (
    ensure_dir,
    new_session_id,
    read_video_metadata,
    session_output_dir,
    speed_status_label,
    validate_video_upload,
)
from src.video_processor import ProcessingSettings, VideoProcessor

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{config.APP_TITLE} — {config.APP_SUBTITLE}",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark navy design system
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---------- Base ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', system-ui, sans-serif;
}

.main { background-color: #0F1421; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1300px; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background: #131929;
    border-right: 1px solid #232B3E;
}
[data-testid="stSidebar"] * { color: #B0BAD0 !important; }
[data-testid="stSidebarNavItems"] a { border-radius: 6px; margin: 2px 0; }
[data-testid="stSidebarNavItems"] a:hover { background: #1E2B45 !important; }

/* ---------- Metric cards ---------- */
.metric-card {
    background: #1A2133;
    border: 1px solid #232B3E;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-card .label {
    font-size: 0.72rem;
    font-weight: 500;
    color: #6B7A99;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #E0E6F0;
    line-height: 1.1;
}
.metric-card .sub {
    font-size: 0.78rem;
    color: #6B7A99;
    margin-top: 0.2rem;
}

/* ---------- Status badge ---------- */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
}
.badge-ok    { background: #1B3E2E; color: #34EB8C; border: 1px solid #34EB8C44; }
.badge-warn  { background: #3E2E10; color: #F5B434; border: 1px solid #F5B43444; }
.badge-alert { background: #3E1010; color: #EB5434; border: 1px solid #EB543444; }

/* ---------- Section headers ---------- */
.section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #C8D3E8;
    border-left: 3px solid #4C8EF5;
    padding-left: 0.7rem;
    margin: 1.5rem 0 0.8rem 0;
}

/* ---------- Info panel ---------- */
.info-panel {
    background: #161D2E;
    border: 1px solid #232B3E;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    color: #8A98B8;
    font-size: 0.85rem;
    line-height: 1.55;
}

/* ---------- Empty state ---------- */
.empty-state {
    background: #161D2E;
    border: 1px dashed #2A3558;
    border-radius: 10px;
    padding: 2.5rem;
    text-align: center;
    color: #5A6A8A;
}
.empty-state .es-icon { font-size: 2.5rem; margin-bottom: 0.8rem; }
.empty-state .es-title { font-size: 1rem; font-weight: 600; color: #8A9AC0; margin-bottom: 0.4rem; }
.empty-state .es-body  { font-size: 0.85rem; line-height: 1.5; }

/* ---------- Progress bar override ---------- */
.stProgress > div > div > div > div { background-color: #4C8EF5; }

/* ---------- Tables ---------- */
.stDataFrame { border: 1px solid #232B3E; border-radius: 6px; overflow: hidden; }

/* ---------- Buttons ---------- */
.stButton > button {
    background: #4C8EF5;
    color: #ffffff;
    border: none;
    border-radius: 6px;
    font-weight: 500;
    padding: 0.45rem 1.2rem;
    transition: background 0.2s;
}
.stButton > button:hover { background: #3A7AE0; }
.stButton > button[kind="secondary"] { background: #1A2133; color: #B0BAD0; border: 1px solid #232B3E; }

/* ---------- Admin authentication ---------- */
.auth-heading { text-align: center; color: #E0E8FF; margin-bottom: 0.25rem; }
.auth-caption { text-align: center; color: #8A98B8; margin-bottom: 1.5rem; }

/* ---------- File uploader ---------- */
[data-testid="stFileUploader"] {
    border: 1px dashed #2A3558;
    border-radius: 8px;
    background: #161D2E;
}

/* ---------- Headings ---------- */
h1 { color: #E0E8FF !important; font-weight: 700 !important; }
h2 { color: #C8D3E8 !important; font-weight: 600 !important; }
h3 { color: #B0BAD0 !important; font-weight: 500 !important; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state bootstrap
# ---------------------------------------------------------------------------

def _init_state() -> None:
    """Initialise all session state keys with their default values."""
    defaults = {
        "session_id": new_session_id(),
        "video_path": None,
        "video_upload_id": None,
        "video_meta": None,
        "processing_done": False,
        "processing_running": False,
        "detection_records": [],   # list of dicts
        "speed_measurements": [],  # List[SpeedMeasurement]
        "distance_measurements": [],  # List[V2VDistanceMeasurement]
        "trail_manager": None,
        "annotated_frames": [],    # list of np.ndarray (subsampled)
        "lines_config": MeasurementLines(),
        "hom_config": HomographyCalibration(),
        "speed_threshold": float(config.DEFAULT_SPEED_THRESHOLD),
        "confidence": config.DEFAULT_CONFIDENCE,
        "frame_skip": config.DEFAULT_FRAME_SKIP,
        "last_annotated": None,
        "processed_video_path": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

def sidebar() -> str:
    with st.sidebar:
        st.markdown(
            f"""
            <div style="padding: 0.5rem 0 1rem 0; text-align: center;">
                <div style="font-size:1.6rem; margin-bottom:0.2rem;">🚗</div>
                <div style="font-size:1.05rem; font-weight:700; color:#E0E8FF;">
                    {config.APP_TITLE}
                </div>
                <div style="font-size:0.72rem; color:#5A6A8A; margin-top:2px;">
                    {config.APP_SUBTITLE}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()
        st.caption(f"Signed in as Admin · {st.session_state.admin_username}")
        st.button(
            "Log out",
            key="admin_logout",
            on_click=_logout_admin,
            width="stretch",
        )
        st.divider()
        page = st.radio(
            "Navigate",
            ["Dashboard", "Analyze Video", "Results", "About"],
            label_visibility="collapsed",
        )
        st.divider()

        # Session status pill
        if st.session_state.processing_done:
            total = (
                st.session_state.trail_manager.total_unique_vehicles
                if st.session_state.trail_manager
                else 0
            )
            st.markdown(
                f'<div style="font-size:0.78rem;color:#34EB8C;">✔ Session ready — {total} vehicles</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="font-size:0.78rem;color:#5A6A8A;">No video processed yet</div>',
                unsafe_allow_html=True,
            )

    return page


def _logout_admin() -> None:
    """Clear the authenticated user's Streamlit session."""
    st.session_state.clear()


def _render_admin_login() -> None:
    """Render the single Admin sign-in screen; no public account registration."""
    _, center, _ = st.columns([1, 1.25, 1])
    with center:
        with st.container(border=True):
            st.markdown(
                f"""
                <div class="auth-heading">
                    <div style="font-size:2rem;">🚗</div>
                    <h1 style="margin:0;">{config.APP_TITLE}</h1>
                    <div style="color:#8A98B8;">{config.APP_SUBTITLE}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.subheader("Admin Login")
            st.caption("Sign in with the administrator account to continue.")

            if not auth_store.has_admin_account():
                st.warning(
                    "Admin access is not set up yet. Run "
                    "`python -m src.admin_setup` in the project terminal."
                )
                return

            with st.form("admin_login_form", clear_on_submit=True):
                username = st.text_input("Username", key="admin_login_username")
                password = st.text_input(
                    "Password",
                    type="password",
                    key="admin_login_password",
                )
                submitted = st.form_submit_button(
                    "Sign in as Admin",
                    key="admin_login_submit",
                    type="primary",
                    width="stretch",
                )

            if submitted:
                account = auth_store.authenticate_admin(username, password)
                if account is None:
                    st.error("Sign-in failed. Check your credentials or account status.")
                    st.caption(
                        "To check the configured Admin account, run "
                        "`python -m src.admin_setup --diagnose`. "
                        "To set a new password securely, run "
                        "`python -m src.admin_setup --reset-password`."
                    )
                elif account.role != "Admin":
                    st.error("This account is not authorized to access the Admin dashboard.")
                else:
                    st.session_state.admin_username = account.username
                    st.session_state.admin_role = account.role
                    st.rerun()


def _has_authenticated_admin() -> bool:
    """Validate the session identity and current database role on every rerun."""
    username = st.session_state.get("admin_username")
    role = st.session_state.get("admin_role")
    if not username or role != "Admin":
        return False
    return auth_store.is_active_admin(username, role)


# ---------------------------------------------------------------------------
# Helper widgets
# ---------------------------------------------------------------------------

def metric_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {'<div class="sub">' + sub + '</div>' if sub else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def empty_state(icon: str, title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="empty-state">
            <div class="es-icon">{icon}</div>
            <div class="es-title">{title}</div>
            <div class="es-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)


def info_panel(text: str) -> None:
    st.markdown(f'<div class="info-panel">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------

def page_dashboard() -> None:
    st.title("Dashboard")
    st.markdown(
        '<p style="color:#5A6A8A;font-size:0.9rem;margin-top:-0.5rem;">'
        "Session overview — results from the most recent video analysis.</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.processing_done:
        empty_state(
            "📹",
            "No vehicles have been processed yet.",
            "Upload a traffic video on the <b>Analyze Video</b> page and run processing.<br>"
            "Results will appear here once analysis is complete.",
        )
        return

    tm = st.session_state.trail_manager
    speeds = st.session_state.speed_measurements
    counts = tm.count_by_class() if tm else {}

    speed_df = speeds_to_dataframe(speeds, st.session_state.speed_threshold)

    # --- Top row: vehicle counts ---
    section("Vehicle Counts")
    cols = st.columns(6)
    metrics = [
        ("Total Unique Vehicles", str(tm.total_unique_vehicles if tm else 0), ""),
        ("Cars", str(counts.get("car", 0)), ""),
        ("Motorcycles", str(counts.get("motorcycle", 0)), ""),
        ("Buses", str(counts.get("bus", 0)), ""),
        ("Trucks", str(counts.get("truck", 0)), ""),
        (
            "With Speed",
            str(speed_df["track_id"].nunique() if not speed_df.empty else 0),
            "valid measurements",
        ),
    ]
    for col, (lbl, val, sub) in zip(cols, metrics):
        with col:
            metric_card(lbl, val, sub)

    # --- Second row: speed stats ---
    section("Speed Summary")
    scols = st.columns(3)
    if not speed_df.empty and "speed_kmh" in speed_df.columns:
        avg_spd = f"{speed_df['speed_kmh'].mean():.1f} km/h"
        max_spd = f"{speed_df['speed_kmh'].max():.1f} km/h"
        threshold = st.session_state.speed_threshold
        above = len(speed_df[speed_df["speed_kmh"] > threshold])
        above_str = str(above) if above else "0"
    else:
        avg_spd = "—"
        max_spd = "—"
        above_str = "—"

    with scols[0]:
        metric_card("Average Speed", avg_spd)
    with scols[1]:
        metric_card("Highest Speed", max_spd)
    with scols[2]:
        metric_card("Above Threshold", above_str, f">{st.session_state.speed_threshold:.0f} km/h")
    st.caption(
        "Speed values are estimates. Reliability depends on a fixed camera, "
        "accurate road-distance calibration, and tracking quality."
    )

    # --- Latest annotated frame ---
    if st.session_state.last_annotated is not None:
        section("Last Processed Frame Preview")
        frame_rgb = cv2.cvtColor(st.session_state.last_annotated, cv2.COLOR_BGR2RGB)
        st.image(frame_rgb, width="stretch", alt="Last processed frame with vehicle detections")

    # --- Recent speed measurements table ---
    if not speed_df.empty:
        section("Recent Speed Measurements")
        display = speed_df[["track_id", "class", "speed_kmh", "status", "video_time_s"]].copy()
        display.columns = ["Track ID", "Class", "Speed (km/h)", "Status", "Video Time (s)"]
        st.dataframe(display.tail(15), hide_index=True, alt="Recent vehicle speed measurements")


# ---------------------------------------------------------------------------
# Page: Analyze Video
# ---------------------------------------------------------------------------

def page_analyze() -> None:
    st.title("Analyze Video")
    st.markdown(
        '<p style="color:#5A6A8A;font-size:0.9rem;margin-top:-0.5rem;">'
        "Upload a traffic video, configure calibration, and run the analysis.</p>",
        unsafe_allow_html=True,
    )

    # ---- 1. Upload ----
    section("1 — Upload Video")
    uploaded = st.file_uploader(
        "Choose a video file",
        type=list(config.SUPPORTED_VIDEO_EXTENSIONS),
        max_upload_size=config.MAX_UPLOAD_SIZE_MB,
        help=f"MP4 (H.264) is best supported. Maximum size: {config.MAX_UPLOAD_SIZE_MB} MB.",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        upload_ok, upload_message = validate_video_upload(
            uploaded.name,
            uploaded.size,
        )
        if not upload_ok:
            st.error(upload_message)
            return

        # Save to a temp file so OpenCV can read it
        suffix = Path(uploaded.name).suffix.lower()
        if (
            uploaded.file_id != st.session_state.video_upload_id
            or not st.session_state.video_path
            or not os.path.isfile(st.session_state.video_path)
        ):
            if st.session_state.video_upload_id is not None:
                _reset_session()

            try:
                tmp_dir = ensure_dir(Path(config.INPUT_DIR))
                tmp_path = tmp_dir / f"upload_{st.session_state.session_id}{suffix}"
                with open(tmp_path, "wb") as f:
                    f.write(uploaded.getbuffer())
            except OSError as exc:
                st.error(f"Could not save uploaded video: {exc}")
                return

            st.session_state.video_path = str(tmp_path)
            st.session_state.video_upload_id = uploaded.file_id
            st.session_state.video_meta = None  # reset metadata cache
            st.session_state.processing_done = False

        # Read metadata
        if st.session_state.video_meta is None:
            try:
                meta = read_video_metadata(st.session_state.video_path)
                st.session_state.video_meta = meta
            except Exception as exc:
                st.error(f"Could not read video: {exc}")
                st.session_state.video_path = None
                st.session_state.video_upload_id = None
                return

        meta = st.session_state.video_meta
        mcols = st.columns(5)
        with mcols[0]:
            metric_card("Resolution", f"{meta['width']}×{meta['height']}")
        with mcols[1]:
            metric_card("Frame Rate", f"{meta['fps']:.1f} fps")
        with mcols[2]:
            metric_card("Duration", f"{meta['duration_seconds']:.1f} s")
        with mcols[3]:
            metric_card("Frames", str(meta['frame_count']))
        with mcols[4]:
            metric_card("Codec", meta['codec'] or "unknown")

        section("Video preview")
        st.video(
            st.session_state.video_path,
            alt=f"Preview of uploaded video {Path(uploaded.name).name}",
        )

        # First-frame preview is useful for placing calibration lines.
        cap_prev = cv2.VideoCapture(st.session_state.video_path)
        ret, preview_frame = cap_prev.read()
        cap_prev.release()
        if ret:
            preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            st.image(
                preview_rgb,
                width="stretch",
                alt="First frame of uploaded video for calibration",
            )
    else:
        info_panel(
            "Upload a traffic video to begin.  "
            "Supported formats: <b>MP4, AVI, MOV, MKV</b>.  "
            "H.264 encoded MP4 files work most reliably on all platforms."
        )
        return

    # ---- 2. Detection settings ----
    section("2 — Detection Settings")
    scols = st.columns(2)
    with scols[0]:
        confidence = st.slider(
            "Detection confidence threshold",
            min_value=0.1,
            max_value=0.95,
            value=float(st.session_state.confidence),
            step=0.05,
            help="Higher = fewer false positives but may miss some vehicles.",
        )
        st.session_state.confidence = confidence
    with scols[1]:
        frame_skip = st.select_slider(
            "Process every N-th frame",
            options=[1, 2, 3, 4, 5],
            value=st.session_state.frame_skip,
            help="1 = every frame (slowest, most accurate).  2–5 = faster but less temporal resolution.",
        )
        st.session_state.frame_skip = frame_skip

    # ---- 3. Calibration ----
    section("3 — Speed Calibration (Measurement Lines)")

    info_panel(
        "<b>How to calibrate:</b> Identify two visible reference points on the road "
        "(e.g. road markings, lane dividers) and measure the real-world distance between them.  "
        "Set Line 1 and Line 2 to the corresponding pixel Y-positions in the video.  "
        "Then enter the actual distance in metres.  "
        "<br><br>"
        "The speed estimator records the video time when a vehicle's road-contact "
        "point is detected near each line, then applies: "
        "<code>speed = distance ÷ elapsed time</code>."
    )

    frame_h = meta["height"] if meta else 480

    ccols = st.columns(3)
    with ccols[0]:
        line1_y = st.number_input(
            "Line 1 — pixel Y",
            min_value=0,
            max_value=frame_h,
            value=max(int(frame_h * 0.35), 0),
            step=1,
        )
    with ccols[1]:
        line2_y = st.number_input(
            "Line 2 — pixel Y",
            min_value=0,
            max_value=frame_h,
            value=max(int(frame_h * 0.65), 0),
            step=1,
        )
    with ccols[2]:
        real_dist = st.number_input(
            "Real-world distance (metres)",
            min_value=1.0,
            max_value=500.0,
            value=20.0,
            step=0.5,
        )

    lines = MeasurementLines(
        line1_y=int(line1_y),
        line2_y=int(line2_y),
        real_distance_m=float(real_dist),
    )
    cal_ok, cal_msg = validate_measurement_lines(lines)

    if cal_ok:
        st.success(f"✔ Calibration valid — {cal_msg}")
        st.session_state.lines_config = lines
    else:
        st.warning(f"⚠ {cal_msg}  Speed will not be estimated until calibration is valid.")
        st.session_state.lines_config = MeasurementLines()  # invalid sentinel

    # ---- 4. Speed monitoring threshold ----
    section("4 — Speed Monitoring Threshold")

    info_panel(
        "Set a reference speed threshold for monitoring.  "
        "The system will label each measurement as "
        "<b>Within Limit</b>, <b>Near Limit</b>, or <b>Above Limit</b>.  "
        "These are configurable monitoring labels — not legal enforcement decisions."
    )

    threshold = st.slider(
        "Speed threshold (km/h)",
        min_value=10,
        max_value=200,
        value=int(st.session_state.speed_threshold),
        step=5,
    )
    st.session_state.speed_threshold = float(threshold)

    # ---- 5. Run / Reset ----
    section("5 — Run Analysis")
    btn_cols = st.columns([1, 1, 4])

    run_pressed = btn_cols[0].button(
        "▶ Start Processing",
        disabled=st.session_state.processing_running,
    )
    reset_pressed = btn_cols[1].button(
        "↺ Reset Analysis",
        type="secondary",
    )

    if reset_pressed:
        _reset_session()
        st.rerun()

    if run_pressed and not st.session_state.processing_running:
        _run_processing(lines=st.session_state.lines_config)


def _reset_session() -> None:
    """Clear all results and generate a fresh session ID."""
    st.session_state.session_id = new_session_id()
    st.session_state.processing_done = False
    st.session_state.processing_running = False
    st.session_state.detection_records = []
    st.session_state.speed_measurements = []
    st.session_state.distance_measurements = []
    st.session_state.trail_manager = None
    st.session_state.annotated_frames = []
    st.session_state.last_annotated = None
    st.session_state.processed_video_path = None


def _run_processing(lines: MeasurementLines) -> None:
    """Execute the video processing pipeline with live progress updates."""
    video_path = st.session_state.video_path
    if not video_path or not os.path.isfile(video_path):
        st.error("No valid video file found.  Please upload a video first.")
        return

    # Load the detector early so import errors surface before we start
    try:
        from src.detector import VehicleDetector
        from src.tracker import TrailManager
    except ImportError as exc:
        st.error(
            f"Required package not installed: {exc}\n"
            "Run `pip install -r requirements.txt` and restart the app."
        )
        return

    # Reset previous results
    _reset_session()
    try:
        output_dir = session_output_dir(st.session_state.session_id)
        settings = ProcessingSettings(
            confidence=st.session_state.confidence,
            frame_skip=st.session_state.frame_skip,
            lines=lines,
            speed_threshold_kmh=st.session_state.speed_threshold,
            output_path=output_dir / config.PROCESSED_VIDEO,
        )
        processor = VideoProcessor(video_path, settings)
    except (OSError, ValueError) as exc:
        st.error(f"Could not prepare video processing: {exc}")
        return

    st.session_state.processing_running = True

    progress_bar = st.progress(0.0, text="Initialising…")
    status_text = st.empty()
    preview_placeholder = st.empty()

    all_detections: list = []
    all_speeds: list = []
    all_distances: list = []
    last_frame = None
    frame_store: list = []
    sample_interval = max(round(processor.fps * 5), 1)

    try:
        for pf in processor.process_video():
            progress_bar.progress(
                pf.progress,
                text=f"Processing frame {pf.frame_index}…",
            )

            # Accumulate detections
            for det in pf.detections:
                all_detections.append(
                    {
                        "frame_index": det.frame_index,
                        "track_id": det.track_id,
                        "class_name": det.class_name,
                        "confidence": det.confidence,
                        "x1": det.x1,
                        "y1": det.y1,
                        "x2": det.x2,
                        "y2": det.y2,
                        "cx": det.cx,
                        "cy": det.cy,
                    }
                )
            all_speeds.extend(pf.new_speeds)
            all_distances.extend(pf.new_distances)
            last_frame = pf.annotated_frame

            # Keep a bounded, lightweight set of stills for older sessions.
            if pf.frame_index % sample_interval == 0 and len(frame_store) < 24:
                frame = pf.annotated_frame
                if frame.shape[1] > 960:
                    height = round(frame.shape[0] * 960 / frame.shape[1])
                    frame = cv2.resize(frame, (960, height))
                frame_store.append(frame.copy())

            # Live preview (update every 15 frames to limit overhead)
            if pf.frame_index % 15 == 0:
                rgb = cv2.cvtColor(pf.annotated_frame, cv2.COLOR_BGR2RGB)
                preview_placeholder.image(
                    rgb,
                    width="stretch",
                    alt="Live annotated video frame during processing",
                )

            # Speed status update
            if pf.new_speeds:
                for sm in pf.new_speeds:
                    status = speed_status_label(sm.speed_kmh, st.session_state.speed_threshold)
                    status_text.markdown(
                        f"Track {sm.track_id} ({sm.class_name}): "
                        f"**{sm.speed_kmh:.1f} km/h** — {status}"
                    )

    except RuntimeError as exc:
        st.error(f"Processing stopped: {exc}")
        st.session_state.processing_running = False
        return
    except Exception as exc:
        st.error(f"Unexpected error during processing: {exc}")
        logging.exception("Unexpected processing error")
        st.session_state.processing_running = False
        return

    if last_frame is None:
        st.error("Video processing ended without decoding any frames.")
        st.session_state.processing_running = False
        return

    # Store results in session state
    st.session_state.detection_records = all_detections
    st.session_state.speed_measurements = all_speeds
    st.session_state.distance_measurements = all_distances
    st.session_state.trail_manager = processor.trail_manager
    st.session_state.annotated_frames = frame_store
    st.session_state.last_annotated = last_frame
    output_video = output_dir / config.PROCESSED_VIDEO
    if output_video.is_file() and output_video.stat().st_size > 0:
        st.session_state.processed_video_path = str(output_video)
    st.session_state.processing_done = True
    st.session_state.processing_running = False

    progress_bar.progress(1.0, text="Processing completed. Review the results below.")
    if all_detections:
        status_text.success(
            f"Processing completed: {len(all_detections):,} vehicle detections. "
            "Review the results below."
        )
    else:
        status_text.warning(
            "Video processing completed, but no vehicles were detected. "
            "Try a lower confidence threshold or a clearer traffic video."
        )

    # Export to output dir
    _export_results()


def _export_results() -> None:
    """Save CSVs and JSON to the session output directory."""
    sid = st.session_state.session_id
    try:
        out_dir = session_output_dir(sid)
    except OSError as exc:
        logging.exception("Could not create session output directory.")
        st.error(f"Could not create the results folder: {exc}")
        return

    speed_df = speeds_to_dataframe(
        st.session_state.speed_measurements,
        st.session_state.speed_threshold,
    )
    dist_df = distances_to_dataframe(st.session_state.distance_measurements)
    det_df = detections_to_dataframe(st.session_state.detection_records)

    summary = build_summary(
        st.session_state.trail_manager,
        speed_df,
        dist_df,
    )

    exports = {
        config.DETECTIONS_CSV: export_csv(det_df, out_dir / config.DETECTIONS_CSV),
        config.SPEEDS_CSV: export_csv(speed_df, out_dir / config.SPEEDS_CSV),
        config.DISTANCES_CSV: export_csv(dist_df, out_dir / config.DISTANCES_CSV),
        config.SUMMARY_JSON: export_json(summary, out_dir / config.SUMMARY_JSON),
    }
    failed_exports = [name for name, succeeded in exports.items() if not succeeded]
    if failed_exports:
        st.error(
            "Could not save result file(s): "
            + ", ".join(failed_exports)
            + ". The downloadable results are still available in this session."
        )


# ---------------------------------------------------------------------------
# Page: Results
# ---------------------------------------------------------------------------

def page_results() -> None:
    st.title("Results & Analytics")
    st.markdown(
        '<p style="color:#5A6A8A;font-size:0.9rem;margin-top:-0.5rem;">'
        "Charts, data tables, and download options for the current session.</p>",
        unsafe_allow_html=True,
    )

    if not st.session_state.processing_done:
        empty_state(
            "📊",
            "No results yet.",
            "Process a video on the <b>Analyze Video</b> page first.",
        )
        return

    tm = st.session_state.trail_manager
    speeds = st.session_state.speed_measurements
    distances = st.session_state.distance_measurements

    speed_df = speeds_to_dataframe(speeds, st.session_state.speed_threshold)
    dist_df = distances_to_dataframe(distances)
    det_df = detections_to_dataframe(st.session_state.detection_records)
    if not speed_df.empty:
        st.caption(
            "Speed values are estimates based on the configured road-distance "
            "calibration and video timing; they are not enforcement-grade measurements."
        )

    # --- Charts ---
    if st.session_state.processed_video_path:
        processed_path = Path(st.session_state.processed_video_path)
        if processed_path.is_file():
            section("Processed video")
            st.video(
                str(processed_path),
                format="video/mp4",
                alt="Video with YOLO vehicle detection, tracking, and calibration overlays",
            )

    section("Vehicle Count by Class")
    if tm:
        fig = chart_vehicle_counts(tm)
        if fig:
            st.plotly_chart(fig, alt="Vehicle count by class")
        else:
            st.info("No vehicle data to chart.")

    if not speed_df.empty:
        section("Speed Distribution")
        c1, c2 = st.columns(2)
        with c1:
            fig = chart_speed_distribution(speed_df)
            if fig:
                st.plotly_chart(fig, alt="Distribution of measured vehicle speeds")
        with c2:
            fig = chart_speed_status(speed_df, st.session_state.speed_threshold)
            if fig:
                st.plotly_chart(fig, alt="Vehicle speed status relative to the selected threshold")

        section("Speed by Vehicle ID")
        fig = chart_speed_by_track(speed_df)
        if fig:
            st.plotly_chart(fig, alt="Measured speed by vehicle tracking ID")
    else:
        info_panel(
            "No speed measurements were recorded.  "
            "This usually means vehicles did not cross both calibration lines during the video, "
            "or the calibration was not set before processing."
        )

    if not dist_df.empty:
        metric_rows = dist_df[dist_df["is_metric"] == True]
        if not metric_rows.empty:
            st.warning(
                "Distance values are approximate. A single scale from the reference "
                "lines does not correct perspective or depth; monocular video alone "
                "cannot provide reliable exact physical distances."
            )
            section("Vehicle-to-Vehicle Distance Over Time")
            fig = chart_v2v_distances(dist_df)
            if fig:
                st.plotly_chart(fig, alt="Estimated vehicle-to-vehicle distance over time")

    # --- Data Tables ---
    section("Detection Records")
    if det_df.empty:
        st.info("No detections recorded.")
    else:
        class_filter = st.multiselect(
            "Filter by class",
            options=list(det_df["class"].unique()) if "class" in det_df.columns else [],
            default=[],
            key="det_class_filter",
        )
        filtered_det = det_df if not class_filter else det_df[det_df["class"].isin(class_filter)]
        st.dataframe(
            filtered_det.head(500),
            hide_index=True,
            alt="Vehicle detection records",
        )

    if not speed_df.empty:
        section("Speed Measurements")
        st.dataframe(speed_df, hide_index=True, alt="Vehicle speed measurements")

    if not dist_df.empty:
        section("Vehicle-to-Vehicle Distances")
        st.dataframe(
            dist_df.head(300),
            hide_index=True,
            alt="Vehicle-to-vehicle distance estimates",
        )

    # --- Downloads ---
    section("Download Results")
    dl_cols = st.columns(4)

    sid = st.session_state.session_id
    out_dir = session_output_dir(sid)

    def _csv_bytes(df: pd.DataFrame) -> bytes:
        return df.to_csv(index=False).encode("utf-8")

    summary = build_summary(tm, speed_df, dist_df)

    with dl_cols[0]:
        st.download_button(
            "⬇ Detection Records (CSV)",
            data=_csv_bytes(det_df),
            file_name=f"detections_{sid}.csv",
            mime="text/csv",
        )
    with dl_cols[1]:
        st.download_button(
            "⬇ Speed Measurements (CSV)",
            data=_csv_bytes(speed_df),
            file_name=f"speeds_{sid}.csv",
            mime="text/csv",
        )
    with dl_cols[2]:
        st.download_button(
            "⬇ Distances (CSV)",
            data=_csv_bytes(dist_df),
            file_name=f"distances_{sid}.csv",
            mime="text/csv",
        )
    with dl_cols[3]:
        summary_json = json.dumps(summary, indent=2, default=str)
        st.download_button(
            "⬇ Summary (JSON)",
            data=summary_json.encode("utf-8"),
            file_name=f"summary_{sid}.json",
            mime="application/json",
        )

    # --- Annotated frame preview ---
    if st.session_state.annotated_frames:
        section("Annotated Frame Preview")
        frame_count = len(st.session_state.annotated_frames)
        if frame_count > 1:
            idx = st.slider(
                "Select frame sample",
                min_value=0,
                max_value=frame_count - 1,
                value=0,
                key="frame_preview_slider",
            )
        else:
            idx = 0
        frame = st.session_state.annotated_frames[idx]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(rgb, width="stretch", alt="Sample annotated video frame")


# ---------------------------------------------------------------------------
# Page: About
# ---------------------------------------------------------------------------

def page_about() -> None:
    st.title("About This Project")

    pd_info = config.PROJECT_DETAILS

    st.markdown(
        f"""
        <div class="info-panel" style="margin-bottom:1rem;">
            <b>Project Title:</b> {pd_info['project_title']}<br>
            <b>Student:</b> {pd_info['student_name']}<br>
            <b>College:</b> {pd_info['college']}<br>
            <b>Department:</b> {pd_info['department']}<br>
            <b>Academic Year:</b> {pd_info['academic_year']}<br>
            <b>Project Guide:</b> {pd_info['guide']}
        </div>
        """,
        unsafe_allow_html=True,
    )

    info_panel(
        "<i>Update the placeholders above in <code>config.py</code> → "
        "<code>PROJECT_DETAILS</code>.</i>"
    )

    section("What This Application Does")
    st.markdown(
        """
        **Vehicle Vision** analyses road traffic videos using computer vision.
        It detects vehicles, assigns unique tracking IDs across frames, estimates
        vehicle speed from calibrated measurement lines, and measures the distance
        between vehicles on the road.

        The application is designed for academic demonstration and learning.
        It makes honest estimates and clearly flags measurements that are
        approximate or unavailable due to missing calibration.
        """
    )

    section("Technology Stack")
    cols = st.columns(3)
    tech = [
        ("Python 3.11+", "Core language"),
        ("OpenCV", "Video reading and frame annotation"),
        ("Ultralytics YOLO", "Real-time object detection"),
        ("ByteTrack", "Multi-object tracking (built into Ultralytics)"),
        ("Streamlit", "Interactive web interface"),
        ("Pandas / NumPy", "Data handling and calculations"),
        ("Plotly", "Interactive charts"),
        ("Pytest", "Unit testing"),
    ]
    for i, (name, desc) in enumerate(tech):
        with cols[i % 3]:
            st.markdown(
                f'<div class="metric-card" style="text-align:left;margin-bottom:0.5rem;">'
                f'<div style="font-weight:600;color:#C8D3E8;">{name}</div>'
                f'<div style="font-size:0.8rem;color:#6B7A99;">{desc}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

    section("How Speed Is Calculated")
    st.markdown(
        """
        1. The user places two horizontal **measurement lines** on the video at
           known positions — for example, at two road-marking lines that are 20 metres apart.
        2. As each tracked vehicle travels through the frame, the system records
           the video time when its road-contact point (bottom-centre of the
           bounding box) is detected near each line. Frame-based timing is
           approximate and depends on frame rate and frame skipping.
        3. Speed is then:
           ```
           speed_mps = real_distance_metres / elapsed_time_seconds
           speed_kmh = speed_mps × 3.6
           ```
        4. Measurements are rejected if the elapsed time is too short,
           or the resulting speed is outside a plausible range.
        5. Each vehicle produces at most one speed measurement to prevent
           duplicate readings.
        """
    )

    section("How Distance Is Estimated")
    st.markdown(
        """
        **Vehicle-to-vehicle distance** is the distance between the ground-contact
        points of two vehicles on the road plane.

        - The current UI estimates distance by scaling the 2D pixel separation
          with the vertical pixel scale from the measurement lines. It does not
          compensate for perspective or changing scale across the image, so metre
          values are rough estimates, not surveyed distances.
        - The processing code supports a four-point **homography calibration**,
          but the current UI does not expose those calibration controls.
        - If no calibration is available, pixel distances are shown but clearly
          marked as non-metric.

        **Distance from camera** requires a known vehicle height and camera focal
        length.  It is based on the pinhole camera model and is approximate even
        with calibration — it is disabled by default.
        """
    )

    section("Known Limitations")
    st.markdown(
        """
        - Speed estimates assume a **fixed camera** and a calibrated road section.
          Moving cameras or uncalibrated videos will produce unreliable results.
        - The two-line method only captures speed over one segment of road.
          Vehicles that do not cross both lines will not receive a measurement.
        - Distance estimation accuracy depends on calibration quality. The UI
          currently uses a single scale for a perspective scene, so distances
          should be treated as approximate.
        - Detection accuracy depends on video resolution, lighting, and
          the chosen YOLO model size.
        - Processing speed is limited by CPU on machines without a compatible GPU.
        """
    )

    section("Your Customisation Checklist")
    st.markdown(
        """
        Here are practical changes you can make yourself to make this project your own:

        - [ ] Fill in your real name, college, and department in `config.py`
        - [ ] Choose your own default speed threshold and confidence level
        - [ ] Record a short demonstration video showing the app running
        - [ ] Test with a different traffic video and note the calibration values you used
        - [ ] Adjust vehicle label colours in `config.py → CLASS_COLORS_BGR`
        - [ ] Write your own explanation of the calibration process for your demonstration
        - [ ] Run the unit tests and confirm they pass on your machine
        - [ ] Read the README and confirm every step works before your demo
        """
    )


# ---------------------------------------------------------------------------
# Main routing
# ---------------------------------------------------------------------------

def main() -> None:
    if not _has_authenticated_admin():
        st.session_state.pop("admin_username", None)
        st.session_state.pop("admin_role", None)
        _render_admin_login()
        return

    page = sidebar()

    if page == "Dashboard":
        page_dashboard()
    elif page == "Analyze Video":
        page_analyze()
    elif page == "Results":
        page_results()
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()
