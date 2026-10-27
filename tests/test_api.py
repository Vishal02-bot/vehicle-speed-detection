"""Tests for REST API job submission and result export."""

import json
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient

import config
from src.auth_store import create_admin_account
import src.api as api


@pytest.fixture
def api_client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "INPUT_DIR", str(tmp_path / "input"))
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("VEHICLE_AUTH_DB", str(tmp_path / "auth.sqlite3"))
    create_admin_account("admin-user", "correct-horse-battery")
    monkeypatch.setattr(
        api,
        "read_video_metadata",
        lambda _path: {"width": 640, "height": 480, "fps": 25.0},
    )

    def finish_job(job_id, video_path, *_args):
        output_dir = api.session_output_dir(job_id)
        summary_path = output_dir / config.SUMMARY_JSON
        summary_path.write_text('{"total_unique_vehicles": 0}', encoding="utf-8")
        api._update_job(
            job_id,
            status="completed",
            results={"files": {config.SUMMARY_JSON: "available"}},
        )
        video_path.unlink(missing_ok=True)

    monkeypatch.setattr(api, "_process_job", finish_job)
    with api._JOBS_LOCK:
        api._JOBS.clear()

    with TestClient(api.app) as client:
        client.auth = ("admin-user", "correct-horse-battery")
        yield client

    with api._JOBS_LOCK:
        api._JOBS.clear()


def test_health_endpoint(api_client):
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_job_routes_reject_missing_and_invalid_admin_credentials(api_client):
    with TestClient(api.app) as unauthenticated_client:
        missing_credentials = unauthenticated_client.get(
            "/api/v1/jobs/not-a-job"
        )
        invalid_credentials = unauthenticated_client.get(
            "/api/v1/jobs/not-a-job",
            auth=("admin-user", "incorrect-password"),
        )

    assert missing_credentials.status_code == 401
    assert missing_credentials.headers["www-authenticate"] == "Basic"
    assert invalid_credentials.status_code == 401


def test_upload_creates_background_job_and_allows_result_download(api_client):
    response = api_client.post(
        "/api/v1/jobs",
        files={"file": ("traffic.mp4", b"video-data", "video/mp4")},
        data={"confidence": "0.4", "frame_skip": "1"},
    )

    assert response.status_code == 202
    job = response.json()
    assert job["status"] == "pending"

    status_response = api_client.get(f"/api/v1/jobs/{job['job_id']}")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "completed"

    file_response = api_client.get(
        f"/api/v1/jobs/{job['job_id']}/files/{config.SUMMARY_JSON}"
    )
    assert file_response.status_code == 200
    assert json.loads(file_response.text)["total_unique_vehicles"] == 0


def test_upload_rejects_unsupported_file_extension(api_client):
    response = api_client.post(
        "/api/v1/jobs",
        files={"file": ("notes.txt", b"not a video", "text/plain")},
    )

    assert response.status_code == 415


def test_upload_rejects_incomplete_calibration(api_client):
    response = api_client.post(
        "/api/v1/jobs",
        files={"file": ("traffic.mp4", b"video-data", "video/mp4")},
        data={"line1_y": "100"},
    )

    assert response.status_code == 422
    assert "provide line1_y" in response.json()["detail"].lower()


def test_processing_worker_exports_existing_analytics(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUT_DIR", str(tmp_path / "output"))
    video_path = tmp_path / "uploaded.mp4"
    video_path.write_bytes(b"video")

    class EmptyProcessor:
        def __init__(self, _video_path, settings):
            self.settings = settings
            self.trail_manager = None
            self.all_speeds = []
            self.all_distances = []

        def process_video(self):
            self.settings.output_path.write_bytes(b"annotated-video")
            return iter(())

    monkeypatch.setattr(api, "VideoProcessor", EmptyProcessor)
    job_id = "worker-test"
    now = datetime.now(timezone.utc)
    with api._JOBS_LOCK:
        api._JOBS[job_id] = api.JobStatus(
            job_id=job_id,
            status="pending",
            created_at=now,
            updated_at=now,
        )

    api._process_job(job_id, video_path, 0.4, 1, 60.0, None)

    job = api.get_job(job_id)
    assert job.status == "completed"
    assert job.results["detections_count"] == 0
    output_dir = tmp_path / "output" / job_id
    assert (output_dir / config.DETECTIONS_CSV).is_file()
    assert (output_dir / config.SPEEDS_CSV).is_file()
    assert (output_dir / config.DISTANCES_CSV).is_file()
    assert json.loads((output_dir / config.SUMMARY_JSON).read_text())[
        "total_unique_vehicles"
    ] == 0
    assert (output_dir / config.PROCESSED_VIDEO).read_bytes() == b"annotated-video"
    assert not video_path.exists()

    with api._JOBS_LOCK:
        api._JOBS.clear()
