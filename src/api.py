"""REST API for submitting and retrieving vehicle video analyses."""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from uuid import uuid4

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
from src.analytics import (
    build_summary,
    detections_to_dataframe,
    distances_to_dataframe,
    export_csv,
    export_json,
    speeds_to_dataframe,
)
from src.calibration import MeasurementLines, validate_measurement_lines
from src.auth_store import authenticate_admin
from src.utils import (
    ensure_dir,
    read_video_metadata,
    session_output_dir,
    validate_video_upload,
)
from src.video_processor import ProcessingSettings, VideoProcessor

logger = logging.getLogger(__name__)
_HTTP_BASIC = HTTPBasic(auto_error=False)

app = FastAPI(
    title="Vehicle Vision API",
    description="Submit road videos for vehicle detection, tracking, and analysis.",
    version="1.0.0",
)

JobState = Literal["pending", "running", "completed", "failed"]
RESULT_FILES = {
    config.DETECTIONS_CSV,
    config.SPEEDS_CSV,
    config.DISTANCES_CSV,
    config.SUMMARY_JSON,
    config.PROCESSED_VIDEO,
}


class JobStatus(BaseModel):
    job_id: str
    status: JobState
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None
    results: Optional[dict] = None


_JOBS: dict[str, JobStatus] = {}
_JOBS_LOCK = threading.Lock()
_PROCESSING_LOCK = threading.Lock()


def require_admin(
    credentials: Optional[HTTPBasicCredentials] = Depends(_HTTP_BASIC),
) -> None:
    """Require a valid active Admin account for every protected API route."""
    account = (
        authenticate_admin(credentials.username, credentials.password)
        if credentials is not None
        else None
    )
    if account is None or account.role != "Admin":
        raise HTTPException(
            status_code=401,
            detail="Admin authentication required.",
            headers={"WWW-Authenticate": "Basic"},
        )


def _update_job(
    job_id: str,
    status: JobState,
    error: Optional[str] = None,
    results: Optional[dict] = None,
) -> None:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is not None:
            _JOBS[job_id] = JobStatus(
                job_id=job.job_id,
                status=status,
                created_at=job.created_at,
                updated_at=datetime.now(timezone.utc),
                error=error,
                results=results,
            )


def _process_job(
    job_id: str,
    video_path: Path,
    confidence: float,
    frame_skip: int,
    speed_threshold_kmh: float,
    lines: Optional[MeasurementLines],
) -> None:
    """Run the existing processing pipeline and save its standard exports."""
    _update_job(job_id, status="running")
    try:
        with _PROCESSING_LOCK:
            output_dir = session_output_dir(job_id)
            processor = VideoProcessor(
                video_path,
                ProcessingSettings(
                    confidence=confidence,
                    frame_skip=frame_skip,
                    lines=lines,
                    speed_threshold_kmh=speed_threshold_kmh,
                    output_path=output_dir / config.PROCESSED_VIDEO,
                ),
            )

            detection_records = []
            for processed_frame in processor.process_video():
                detection_records.extend(
                    {
                        "frame_index": detection.frame_index,
                        "track_id": detection.track_id,
                        "class_name": detection.class_name,
                        "confidence": detection.confidence,
                        "x1": detection.x1,
                        "y1": detection.y1,
                        "x2": detection.x2,
                        "y2": detection.y2,
                        "cx": detection.cx,
                        "cy": detection.cy,
                        "bx": detection.bx,
                        "by": detection.by,
                    }
                    for detection in processed_frame.detections
                )

            speed_df = speeds_to_dataframe(
                processor.all_speeds,
                speed_threshold=speed_threshold_kmh,
            )
            distance_df = distances_to_dataframe(processor.all_distances)
            detection_df = detections_to_dataframe(detection_records)
            summary = build_summary(
                processor.trail_manager,
                speed_df,
                distance_df,
            )

            exports = (
                (detection_df, output_dir / config.DETECTIONS_CSV),
                (speed_df, output_dir / config.SPEEDS_CSV),
                (distance_df, output_dir / config.DISTANCES_CSV),
            )
            failed_exports = [
                path.name for dataframe, path in exports
                if not export_csv(dataframe, path)
            ]
            if not export_json(summary, output_dir / config.SUMMARY_JSON):
                failed_exports.append(config.SUMMARY_JSON)
            if failed_exports:
                raise OSError(
                    "Could not save result file(s): " + ", ".join(failed_exports)
                )

            video_output = output_dir / config.PROCESSED_VIDEO
            result_files = {
                filename: f"/api/v1/jobs/{job_id}/files/{filename}"
                for filename in sorted(RESULT_FILES)
                if (output_dir / filename).is_file()
                and (output_dir / filename).stat().st_size > 0
            }
            results = {
                "detections_count": len(detection_records),
                "speed_measurements_count": len(processor.all_speeds),
                "distance_measurements_count": len(processor.all_distances),
                "summary": summary,
                "files": result_files,
            }
            if video_output.name not in result_files:
                logger.warning("Job %s completed without an annotated video.", job_id)

        _update_job(job_id, status="completed", results=results)
    except Exception as exc:
        logger.exception("Video processing failed for API job %s.", job_id)
        _update_job(job_id, status="failed", error=str(exc))
    finally:
        try:
            video_path.unlink(missing_ok=True)
        except OSError:
            logger.exception("Could not remove uploaded video for job %s.", job_id)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/jobs", response_model=JobStatus, status_code=202)
async def create_job(
    background_tasks: BackgroundTasks,
    _admin: None = Depends(require_admin),
    file: UploadFile = File(...),
    confidence: float = Form(config.DEFAULT_CONFIDENCE),
    frame_skip: int = Form(config.DEFAULT_FRAME_SKIP),
    speed_threshold_kmh: float = Form(float(config.DEFAULT_SPEED_THRESHOLD)),
    line1_y: Optional[int] = Form(None),
    line2_y: Optional[int] = Form(None),
    real_distance_m: Optional[float] = Form(None),
) -> JobStatus:
    """Upload a video and queue it for processing."""
    filename = file.filename or ""
    is_valid, message = validate_video_upload(filename, 1)
    if not is_valid:
        await file.close()
        raise HTTPException(status_code=415, detail=message)

    try:
        ProcessingSettings(
            confidence=confidence,
            frame_skip=frame_skip,
            speed_threshold_kmh=speed_threshold_kmh,
        )
    except ValueError as exc:
        await file.close()
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    calibration_values = (line1_y, line2_y, real_distance_m)
    lines: Optional[MeasurementLines] = None
    if any(value is not None for value in calibration_values):
        if any(value is None for value in calibration_values):
            await file.close()
            raise HTTPException(
                status_code=422,
                detail="Provide line1_y, line2_y, and real_distance_m together.",
            )
        lines = MeasurementLines(
            line1_y=line1_y,
            line2_y=line2_y,
            real_distance_m=real_distance_m,
        )
        valid_lines, calibration_message = validate_measurement_lines(lines)
        if not valid_lines:
            await file.close()
            raise HTTPException(status_code=422, detail=calibration_message)

    job_id = uuid4().hex
    input_path = ensure_dir(config.INPUT_DIR) / (
        f"{job_id}{Path(filename).suffix.lower()}"
    )
    upload_limit = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    size = 0

    try:
        with input_path.open("xb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > upload_limit:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video exceeds the {config.MAX_UPLOAD_SIZE_MB} MB upload limit.",
                    )
                destination.write(chunk)

        is_valid, message = validate_video_upload(filename, size)
        if not is_valid:
            status_code = 413 if size > upload_limit else 415
            raise HTTPException(status_code=status_code, detail=message)

        try:
            metadata = read_video_metadata(str(input_path))
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if lines and any(
            not 0 <= line_y < metadata["height"]
            for line_y in (lines.line1_y, lines.line2_y)
        ):
            raise HTTPException(
                status_code=422,
                detail="Measurement line coordinates must be inside the video frame.",
            )

        now = datetime.now(timezone.utc)
        job = JobStatus(
            job_id=job_id,
            status="pending",
            created_at=now,
            updated_at=now,
        )
        with _JOBS_LOCK:
            _JOBS[job_id] = job

        background_tasks.add_task(
            _process_job,
            job_id,
            input_path,
            confidence,
            frame_skip,
            speed_threshold_kmh,
            lines,
        )
        return job
    except Exception:
        input_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
def get_job(job_id: str, _admin: None = Depends(require_admin)) -> JobStatus:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/api/v1/jobs/{job_id}/files/{filename}")
def get_job_file(
    job_id: str,
    filename: str,
    _admin: None = Depends(require_admin),
) -> FileResponse:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.status != "completed":
        raise HTTPException(status_code=409, detail="Job results are not ready.")
    if filename not in RESULT_FILES:
        raise HTTPException(status_code=404, detail="Result file not found.")

    output_path = session_output_dir(job_id) / filename
    if not output_path.is_file():
        raise HTTPException(status_code=404, detail="Result file not found.")
    return FileResponse(output_path, filename=filename)
