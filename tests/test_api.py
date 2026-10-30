import time
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from radar_platform.api import app
from radar_platform.jobs import job_manager
from radar_platform.models import default_request


def test_health_and_presets() -> None:
    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 200
        response = client.get("/api/v1/presets")
        assert response.status_code == 200
        assert "high_altitude_missile_10ghz" in response.json()
        assert "clean_high_altitude_demo_10ghz" in response.json()
        terrain = client.get("/api/v1/terrain/ideal-flat")
        assert terrain.status_code == 200
        terrain_data = terrain.json()
        assert terrain_data["name"] == "理想平地 DEM"
        assert all(value == 0.0 for row in terrain_data["elevation_m"] for value in row)


def test_simulation_job_and_binary_frame(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RADAR_RUNTIME_ROOT", str(tmp_path / "runtime"))
    request = default_request().model_copy(deep=True)
    request.sequence.frame_count = 1
    with TestClient(app) as client:
        response = client.post("/api/v1/simulations", json=request.model_dump(mode="json"))
        assert response.status_code == 202
        job_id = response.json()["job_id"]
        deadline = time.monotonic() + 10.0
        state = None
        while time.monotonic() < deadline:
            state = client.get(f"/api/v1/simulations/{job_id}").json()
            if state["status"] in {"complete", "failed"}:
                break
            time.sleep(0.05)
        assert state is not None
        assert state["status"] == "complete"
        frame = client.get(f"/api/v1/simulations/{job_id}/frames/0")
        assert frame.status_code == 200
        rows = int(frame.headers["X-Rows"])
        cols = int(frame.headers["X-Cols"])
        array = np.frombuffer(frame.content, dtype="<f4").reshape(rows, cols)
        assert array.shape == (64, cols)
        assert np.isfinite(array).all()
        with client.websocket_connect(f"/api/v1/simulations/{job_id}/events") as websocket:
            snapshot = websocket.receive_json()
            terminal = websocket.receive_json()
            assert snapshot["type"] == "snapshot"
            assert len(snapshot["job"]["frames"]) == 1
            assert terminal["type"] == "job_completed"
    job_manager.clear()


def test_invalid_request_returns_422() -> None:
    request = default_request().model_dump(mode="json")
    request["radar"]["pulse_width_s"] = 1.0
    with TestClient(app) as client:
        response = client.post("/api/v1/simulations", json=request)
        assert response.status_code == 422


def test_ramp_fallback_api(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RADAR_DATA_ROOT", str(tmp_path / "datasets"))
    monkeypatch.setenv("RADAR_RUNTIME_ROOT", str(tmp_path / "runtime"))
    with TestClient(app) as client:
        status = client.get("/api/v1/datasets/ramp/status")
        assert status.status_code == 200
        assert status.json()["is_real_available"] is False
        processed = client.post("/api/v1/datasets/ramp/process", json={})
        assert processed.status_code == 200
        metadata = processed.json()
        assert metadata["is_real"] is False
        rd = client.get(f"/api/v1/datasets/ramp/results/{metadata['result_id']}/rd")
        assert rd.status_code == 200
        assert int(rd.headers["X-Rows"]) == 255
        assert int(rd.headers["X-Cols"]) == 128
