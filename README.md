# Vehicle Vision — Speed & Distance Monitor

> A Python-based computer vision project for analysing road traffic videos.

---

## What This Project Does

Vehicle Vision analyses road traffic videos and provides:

- **Vehicle detection** using a real YOLO model
- **Unique vehicle tracking** with ByteTrack (each vehicle keeps the same ID across frames)
- **Speed estimation in km/h** using a two-line calibration method
- **Vehicle-to-vehicle distance estimation in metres** using road-plane mapping
- **Vehicle classification** — cars, motorcycles, buses, trucks
- **Per-detection labels** — class, confidence, and ByteTrack ID when available
- **Speed monitoring** with configurable thresholds
- **Interactive charts and data tables**
- **CSV and JSON exports** for further analysis

---

## Why It Is Useful

Traffic speed measurement and vehicle counting are standard tools in transport engineering and road safety. This project demonstrates how computer vision techniques — detection, tracking, and geometric calibration — can be combined into a practical, self-contained Python application.

It is designed as a college project. It makes honest estimates and clearly labels measurements that are approximate or unavailable due to missing calibration.

---

## Features

| Feature | Details |
|---|---|
| Object detection | YOLOv8n (small, fast, pre-trained on COCO) |
| Tracking | ByteTrack via Ultralytics |
| Speed estimation | Two-line calibration, interpolated frame times, either direction |
| Distance estimation | Homography or pixel-scale fallback |
| User interface | Streamlit with video player, annotated-video playback, charts, and exports |
| Video formats | MP4, AVI, MOV, MKV |
| Exports | CSV (detections, speeds, distances) + JSON summary |
| Tests | Pytest unit tests (no model or video required) |

Vehicle classes use the bundled COCO-pretrained `yolov8n.pt` model: car = class
2, motorcycle (motorbike) = class 3, bus = class 5, and truck = class 7.
COCO bicycle is class 1 and is intentionally not included in the motor-vehicle
filter. No additional model weights are required for motorcycle or truck
detection. If the bundled weights are missing, Ultralytics can download
`yolov8n.pt` on first use when internet access is available.

---

## Technology Stack

- **Python 3.11** — recommended runtime
- **OpenCV** — video reading, frame annotation
- **Ultralytics YOLO** — object detection and ByteTrack
- **Streamlit** — interactive web interface
- **SQLite / Argon2id** — local Admin-only authentication
- **Pandas / NumPy** — data handling and calculations
- **Plotly** — interactive charts
- **Pytest** — unit testing

---

## Folder Structure

```
vehicle-vision/
├── app.py                  ← Streamlit application entry point
├── config.py               ← All configuration constants
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
├── src/
│   ├── __init__.py
│   ├── detector.py         ← YOLO detection wrapper
│   ├── tracker.py          ← Per-vehicle trail management
│   ├── speed_estimator.py  ← Two-line speed calculation
│   ├── distance_estimator.py ← V2V and camera distance
│   ├── calibration.py      ← Calibration data and validation
│   ├── video_processor.py  ← Frame-by-frame pipeline
│   ├── analytics.py        ← DataFrames, charts, exports
│   ├── auth_store.py       ← SQLite Admin accounts and Argon2id verification
│   ├── admin_setup.py      ← Secure, one-time local Admin setup
│   ├── api.py              ← Optional Admin-protected REST API
│   └── utils.py            ← Shared helpers
├── tests/
│   ├── __init__.py
│   ├── test_auth_store.py
│   ├── test_admin_login_ui.py
│   ├── test_api.py
│   ├── test_analytics.py
│   ├── test_speed_estimator.py
│   ├── test_distance_estimator.py
│   ├── test_calibration.py
│   ├── test_tracker.py
│   ├── test_video_processor.py
│   └── test_utils.py
└── data/
    ├── input/              ← Uploaded videos are saved here temporarily
    ├── output/             ← Per-session results (CSV, JSON)
    ├── models/             ← YOLO weights cache (optional)
    └── auth/               ← Local SQLite Admin account database
```

---

## Installation (Windows PowerShell)

### 1. Create and activate a virtual environment

```powershell
cd "C:\Users\vishal anil khatake\vehicle speed detection"
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow it for the current process:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The provided `yolov8n.pt` model is used when available. If it is absent, Ultralytics may download it on first run.

If you already have the virtual environment and do not want to activate it, use:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Create the Admin account (first run only)

```powershell
python -m src.admin_setup
```

Enter the Admin username and password when prompted. Password characters are
hidden in the terminal; use a password of at least 12 characters. The password
is stored only as an Argon2id hash in `data/auth/admin.sqlite3`. There is no
default account or normal User login.

For a forgotten password, reset it from the trusted local terminal:

```powershell
python -m src.admin_setup --reset-password
```

To change both the existing Admin username and password without creating a
duplicate account, run:

```powershell
python -m src.admin_reset
```

The utility asks for the current Admin username and the new username. It hides
both password prompts, creates a unique verified SQLite backup in
`data/auth/backups/`, updates the existing account row, and verifies that the
new credentials belong to an active Admin. It does not create an account when
the current username is missing.

The reset command verifies the named active Admin exists, prompts for the new
password without echoing it, updates only that account's Argon2id hash, clears
its failed-login lockout, and verifies the new credentials against the database.
To check which SQLite database and account status the command is using, run:

```powershell
python -m src.admin_setup --diagnose
```

### 4. Launch the Streamlit app

```powershell
python -m streamlit run app.py
```

Open the URL printed in the terminal (usually `http://localhost:8501`) and sign
in on the Admin Login page. All dashboard pages remain hidden until the active
Admin account is authenticated. Use **Log out** in the sidebar to end the
Streamlit session.

### Run the REST API (optional)

The API reuses the existing video-processing pipeline and runs alongside the
Streamlit UI as a separate process:

```powershell
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

- `GET /health` checks that the API is responding.
- `POST /api/v1/jobs` accepts a multipart video upload and optional processing
  settings (`confidence`, `frame_skip`, `speed_threshold_kmh`). To calculate
  speed, provide all three calibration fields: `line1_y`, `line2_y`, and
  `real_distance_m`.
- `GET /api/v1/jobs/{job_id}` returns job status, counts, summary, and download
  links when processing completes.
- `GET /api/v1/jobs/{job_id}/files/{filename}` downloads an exported CSV, JSON
  summary, or annotated MP4.

Video analysis runs as a background job. Job status is held in memory and is
lost if the API process restarts; generated output files remain in
`data/output/<job_id>/`. Job submission, status, and download routes require
HTTP Basic credentials for the active Admin account. The API is bound to
loopback by default; do not expose it to an untrusted network without HTTPS and
additional deployment hardening.

The SQLite database and uploaded/output data are local files. Back them up
securely, restrict machine access, and use HTTPS when accessing the app over a
network. The app has no separate User accounts or User dashboard.

---

## How to Use the Application

### Step 1 — Upload a Video

Go to **Analyze Video** in the sidebar.  
Click the upload area and select an MP4 (or AVI/MOV) traffic video.  
The app shows an in-browser video player, a first-frame view for calibration, and the video resolution, frame rate, duration, and codec. Uploads are limited to 500 MB.

### Step 2 — Configure Calibration

Identify two horizontal reference positions on the road — for example, two lane markings or road lines — and measure the actual distance between them in metres.

Set **Line 1 Y** and **Line 2 Y** to the corresponding pixel positions in the video.  
Enter the **real-world distance** between those positions.

A green confirmation message will appear when the calibration is valid.

### Step 3 — Set the Speed Threshold

Choose a monitoring threshold in km/h.  
The system will label each measurement as:
- **Within Limit** — speed ≤ threshold
- **Near Limit** — speed within 10 km/h above threshold  
- **Above Limit** — speed more than 10 km/h above threshold

These are monitoring labels, not legal enforcement decisions.

### Step 4 — Run Analysis

Click **▶ Start Processing**.  
A live preview updates during processing.  
When complete, go to **Results** for charts, tables, and downloads.

---

## How Speed Is Calculated

Speed is calculated using two horizontal measurement lines placed across the road:

1. The user calibrates two lines (Line 1 and Line 2) in the video and enters the real-world distance between them in metres.
2. For each tracked vehicle, the system records the video time when its road-contact point (bottom-centre of the bounding box) is detected within 8 pixels of each line. This frame-based timing is approximate and depends on video frame rate and frame skipping.
3. Speed is then:

```
speed_mps = real_distance_m / elapsed_time_seconds
speed_kmh = speed_mps × 3.6
```

4. Crossing times are interpolated between observations, which helps when
   processing every Nth frame. Both travel directions are supported. Measurements are rejected if:
   - Elapsed time is less than 0.1 seconds (to filter noise)
   - Speed exceeds 250 km/h or is below 1 km/h
   - The vehicle has already received a measurement (to prevent duplicates)

Displayed speeds are estimates, not enforcement-grade measurements. Their
reliability depends on a fixed camera, accurate road-distance calibration,
frame timing, and tracking quality. Untracked detections remain visible with
their class and confidence, but are excluded from speed and distance
calculations until a tracking ID is available.

---

## How Distance Estimation Works

**Vehicle-to-vehicle distance** is the estimated real-world distance between the ground-contact points of two vehicles.

- The current Streamlit UI estimates distance by scaling the 2D pixel separation with the vertical pixel scale derived from the measurement lines. Results are explicitly labelled **approximate calibrated estimates**. This assumes a single scale across the image and does not compensate for perspective or depth; it is not an exact physical distance measurement.
- The processing code can use a four-point **homography calibration** if supplied programmatically, but the current UI does not expose those calibration controls.
- If no calibration is available, pixel distances are shown but clearly marked as non-metric.

Ordinary monocular video cannot provide reliable exact physical distances without
suitable scene calibration or additional assumptions.

**Distance from camera** is optionally available using the pinhole camera model:
```
distance = (known_height_m × focal_length_px) / apparent_bbox_height_px
```
This requires a known vehicle height and camera focal length, and the result is always approximate.

---

## Running the Unit Tests

```powershell
pytest tests/ -v
```

The tests cover speed calculations in both directions, interpolated crossings,
tracking-record aggregation, annotated-video output, distance calculations,
calibration validation, analytics, and utility functions. They run without a
downloaded YOLO model or sample video.

Expected output: all tests should pass. If any fail, the output will show exactly which assertion failed.

---

## Known Limitations

- Speed estimation requires a **fixed camera**. A moving camera produces unreliable results.
- The two-line method only captures speed over one road segment. Vehicles must be tracked across both lines to receive a measurement.
- Speed estimates depend on line placement, measured road distance, camera angle, perspective, frame rate, frame skipping, and tracking quality. They are estimates, not enforcement-grade measurements.
- Distance estimation accuracy depends directly on calibration quality.
- Detection accuracy depends on video resolution, lighting, and camera angle.
- Processing is CPU-bound on machines without a compatible NVIDIA GPU. Expect 2–10× slower than real-time on CPU.
- Annotated MP4 output is saved in each session's output folder. Streamlit keeps only a bounded set of downscaled stills in memory.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ultralytics` not found | Run `pip install -r requirements.txt` |
| Video won't open | Try re-encoding to H.264 MP4 with HandBrake or FFmpeg |
| No vehicles detected | Lower the confidence threshold (try 0.25) |
| No speed measurements | Check that vehicles cross both lines; check calibration |
| Streamlit rerun loop | Only click Start Processing once; use Reset to clear |
| Execution policy error | `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |

---

## Your Customisation Checklist

These are tasks you should complete yourself before your project demonstration:

- [ ] Fill in your name, college, and department in `config.py → PROJECT_DETAILS`
- [ ] Choose your own default speed threshold in `config.py → DEFAULT_SPEED_THRESHOLD`
- [ ] Test the application with a real traffic video and record the calibration values you used
- [ ] Adjust vehicle label colours in `config.py → CLASS_COLORS_BGR` if you prefer different colours
- [ ] Read through `src/speed_estimator.py` so you can explain the algorithm in your own words
- [ ] Run the unit tests (`pytest tests/ -v`) and confirm they all pass
- [ ] Record a short demonstration video showing the application processing a video
- [ ] Write a short explanation of the calibration process in your own words for your demonstration

---

## For Your College Demonstration

When presenting this project, you can explain:

1. **Why computer vision?** Manual vehicle counting and speed measurement is expensive. Automated systems can analyse recorded video at low cost.
2. **How YOLO works.** It is a neural network that predicts bounding boxes and class probabilities in a single forward pass, making it fast enough for real-time processing.
3. **Why ByteTrack?** Tracking assigns consistent IDs across frames so we can follow the same vehicle through the video and measure how long it takes to travel a known distance.
4. **The measurement method.** You place two lines on the video at positions you measured in real life. Time = frames ÷ fps. Speed = distance ÷ time.
5. **Limitations and future work.** The system uses a fixed camera and simple calibration. Future improvements could include automatic calibration, night-time detection with better models, and real-time stream processing.

---

*This project was built as an academic engineering project. Measurement estimates are for educational demonstration purposes only.*
