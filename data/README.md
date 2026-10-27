# data/README.md

## Data Directories

This folder is created automatically when the application runs.

```
data/
├── input/      Uploaded video files (temporary, session-specific)
├── output/     Per-session results
│   └── <session_id>/
│       ├── detections.csv
│       ├── speed_measurements.csv
│       ├── distance_measurements.csv
│       ├── summary.json
│       └── processed_video.mp4
├── models/     Optional: place YOLO weight files here to avoid re-downloading
│               Example: yolov8n.pt
└── auth/       SQLite database for the single Admin account
                Passwords are stored as Argon2id hashes.
```

The `input/`, `output/`, `models/`, and `auth/` directories are excluded from
Git (see `.gitignore`). Back up `auth/admin.sqlite3` securely; it contains the
Admin account hash and is required to sign in.

Place YOLO model files in `data/models/` if you want to use them offline.
The application will look there first before trying to download from the internet.
