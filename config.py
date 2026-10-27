"""
config.py — Central configuration for Vehicle Vision.

All tunable constants live here so you can adjust behavior without
hunting through multiple source files.
"""

# ---------------------------------------------------------------------------
# Application identity
# ---------------------------------------------------------------------------
APP_TITLE = "Vehicle Vision"
APP_SUBTITLE = "Speed & Distance Monitor"
APP_DESCRIPTION = "A Python-based computer vision project for analyzing road traffic videos."

# Student / project details — fill these in yourself
PROJECT_DETAILS = {
    "student_name": "[Your Name]",
    "college": "[Your College / University]",
    "department": "[Your Department]",
    "academic_year": "[Academic Year]",
    "guide": "[Project Guide Name]",
    "project_title": "Vehicle Speed and Distance Detection System",
}

# ---------------------------------------------------------------------------
# Detection settings
# ---------------------------------------------------------------------------
# YOLO model to use.  'yolov8n.pt' is small and fast; use 'yolov8s.pt' or
# 'yolov8m.pt' for better accuracy on a machine with more memory / GPU.
YOLO_MODEL_NAME = "yolov8n.pt"

# COCO class IDs that correspond to vehicles we care about.
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASS_IDS = [2, 3, 5, 7]

# Human-readable labels for those class IDs
VEHICLE_CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Default detection confidence threshold (0–1)
DEFAULT_CONFIDENCE = 0.40

# ---------------------------------------------------------------------------
# Tracking settings
# ---------------------------------------------------------------------------
# ByteTrack parameters (passed to the YOLO tracker)
TRACKER_TYPE = "bytetrack.yaml"  # built-in Ultralytics ByteTrack config
MAX_AGE = 30          # frames to keep a lost track alive
MIN_HITS = 3          # minimum detections before a track is confirmed
IOU_THRESHOLD = 0.3   # IOU threshold for track association

# ---------------------------------------------------------------------------
# Speed estimation
# ---------------------------------------------------------------------------
# Minimum valid speed to record (filters noise from stationary vehicles)
MIN_SPEED_KMH = 1.0

# Maximum plausible road speed; measurements above this are flagged
MAX_SPEED_KMH = 250.0

# Minimum elapsed time between line crossings to accept a measurement (s)
MIN_ELAPSED_SECONDS = 0.1

# ---------------------------------------------------------------------------
# Speed monitoring thresholds (km/h)
# ---------------------------------------------------------------------------
DEFAULT_SPEED_THRESHOLD = 60     # "within limit" if speed ≤ this
NEAR_THRESHOLD_MARGIN = 10       # "near limit" if speed is within this much

# ---------------------------------------------------------------------------
# Distance estimation
# ---------------------------------------------------------------------------
# Minimum pixel distance between vehicle centers to bother estimating
MIN_PIXEL_DISTANCE = 20

# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------
# Accepted upload formats and per-file upload limit.
SUPPORTED_VIDEO_EXTENSIONS = ("mp4", "avi", "mov", "mkv")
MAX_UPLOAD_SIZE_MB = 500

# Maximum dimension (pixels) to resize frames to before processing.
# Smaller → faster; larger → more accurate detections on small objects.
MAX_FRAME_WIDTH = 1280

# Process every Nth frame (1 = every frame, 2 = every other frame, …)
# Higher values speed things up but reduce temporal resolution for speed calc.
DEFAULT_FRAME_SKIP = 1

# Number of frames to keep in the movement trail shown on video
TRAIL_LENGTH = 20

# ---------------------------------------------------------------------------
# Output / export
# ---------------------------------------------------------------------------
INPUT_DIR = "data/input"
OUTPUT_DIR = "data/output"
MODELS_DIR = "data/models"

# Output CSV / JSON filenames (inside OUTPUT_DIR/<session_id>/)
DETECTIONS_CSV = "detections.csv"
SPEEDS_CSV = "speed_measurements.csv"
DISTANCES_CSV = "distance_measurements.csv"
SUMMARY_JSON = "summary.json"
PROCESSED_VIDEO = "processed_video.mp4"

# ---------------------------------------------------------------------------
# UI colors (used in Streamlit and OpenCV overlays)
# ---------------------------------------------------------------------------
# OpenCV uses BGR
COLOR_CAR_BGR = (235, 180, 52)         # warm amber
COLOR_MOTORCYCLE_BGR = (52, 200, 235)  # cyan
COLOR_BUS_BGR = (52, 235, 140)         # green
COLOR_TRUCK_BGR = (200, 52, 235)       # purple
COLOR_DEFAULT_BGR = (200, 200, 200)    # light grey

CLASS_COLORS_BGR = {
    2: COLOR_CAR_BGR,
    3: COLOR_MOTORCYCLE_BGR,
    5: COLOR_BUS_BGR,
    7: COLOR_TRUCK_BGR,
}

# Streamlit accent hex colors per class (used in charts / badges)
CLASS_COLORS_HEX = {
    "car": "#F5B434",
    "motorcycle": "#34C8EB",
    "bus": "#34EB8C",
    "truck": "#C834EB",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL = "INFO"   # DEBUG | INFO | WARNING | ERROR
