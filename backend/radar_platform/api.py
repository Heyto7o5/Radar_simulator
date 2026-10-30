from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import __version__
from .jobs import frame_bytes, job_manager
from .models import SimulationCreated, SimulationRequest, clean_demo_request, default_request
from .physics import derived_metrics, request_warnings
from .surface_api import router as surface_router
from .clutter_api import router as clutter_router
from .background_api import router as background_router
from .ground_api import router as ground_router
from .array_api import router as array_router
from .unified.api import router as scene_router
from .workbench import router as workbench_router
from .comparison_ui import router as comparison_router
from .ramp_dataset import (
    RAMP_CONFIG,
    dataset_status,
    get_result,
    list_frames,
    list_sequences,
    process_frame,
    result_array,
)


logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(
    title="Radar RD Platform API",
    version=__version__,
    description="Verifiable high-altitude RD simulation and open ADC validation demo.",
)
app.include_router(surface_router)
app.include_router(clutter_router)
app.include_router(background_router)
app.include_router(ground_router)
app.include_router(array_router)
app.include_router(scene_router)
app.include_router(workbench_router)
from .workbench_lab import router as lab_router
app.include_router(lab_router)
app.include_router(comparison_router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_origin_regex=r"^https?://([a-zA-Z0-9.-]+|\[[0-9a-fA-F:]+\]):5173$",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Rows", "X-Cols", "X-Frame-Index", "X-Dtype", "X-Order", "X-Shape", "X-Reference-Power"],
)


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get('/api/v1/capabilities')
def platform_capabilities():
    from .capabilities import capabilities
    return capabilities()


@app.get("/api/v1/presets")
def presets() -> dict:
    clean_request = clean_demo_request()
    high_speed_request = default_request()
    return {
        "clean_high_altitude_demo_10ghz": {
            "name": "高空近垂直干净演示",
            "description": "90 m/s，无 Doppler 模糊，突出已完成的主链路和解析验证。",
            "request": clean_request.model_dump(mode="json"),
            "derived_metrics": derived_metrics(clean_request.radar).model_dump(mode="json"),
            "warnings": request_warnings(clean_request),
        },
        "high_altitude_missile_10ghz": {
            "name": "1200 m/s 高速近垂直平台",
            "description": "保留真实单 PRF 折叠，用于高速运动和后续多 PRF 扩展。",
            "request": high_speed_request.model_dump(mode="json"),
            "derived_metrics": derived_metrics(high_speed_request.radar).model_dump(mode="json"),
            "warnings": request_warnings(high_speed_request),
        }
    }


@app.get("/api/v1/terrain/ideal-flat")
def ideal_flat_terrain(
    x_min_m: float = Query(-1000.0, ge=-20_000.0, le=20_000.0),
    x_max_m: float = Query(2000.0, ge=-20_000.0, le=20_000.0),
    y_min_m: float = Query(-1000.0, ge=-20_000.0, le=20_000.0),
    y_max_m: float = Query(1000.0, ge=-20_000.0, le=20_000.0),
    resolution_m: float = Query(50.0, ge=10.0, le=500.0),
) -> dict:
    if x_max_m <= x_min_m or y_max_m <= y_min_m:
        raise HTTPException(status_code=422, detail="terrain maximum bounds must exceed minimum bounds")
    x_axis = np.arange(x_min_m, x_max_m + resolution_m * 0.5, resolution_m)
    y_axis = np.arange(y_min_m, y_max_m + resolution_m * 0.5, resolution_m)
    if x_axis.size * y_axis.size > 20_000:
        raise HTTPException(status_code=422, detail="terrain grid exceeds 20,000 cells")
    elevation = np.zeros((y_axis.size, x_axis.size), dtype=np.float32)
    return {
        "id": "ideal_flat",
        "name": "理想平地 DEM",
        "coordinate_frame": "NED horizontal plane / elevation up",
        "resolution_m": resolution_m,
        "x_m": x_axis.tolist(),
        "y_m": y_axis.tolist(),
        "elevation_m": elevation.tolist(),
        "is_physical_scattering_calibrated": False,
    }


@app.post("/api/v1/simulations", response_model=SimulationCreated, status_code=202)
async def create_simulation(request: SimulationRequest) -> SimulationCreated:
    record = job_manager.create(request)
    return SimulationCreated(
        job_id=record.job_id,
        status=record.status,
        derived_metrics=record.derived,
        warnings=record.warnings,
    )


@app.get("/api/v1/simulations/{job_id}")
def get_simulation(job_id: str) -> dict:
    try:
        return job_manager.get(job_id).public().model_dump(mode="json")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/v1/simulations/{job_id}")
async def cancel_simulation(job_id: str) -> dict:
    try:
        record = await job_manager.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return record.public().model_dump(mode="json")


@app.get("/api/v1/simulations/{job_id}/frames/{frame_index}")
def get_simulation_frame(job_id: str, frame_index: int) -> Response:
    try:
        record = job_manager.get(job_id)
        payload, shape = frame_bytes(record, frame_index)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="frame not ready") from exc
    return Response(
        content=payload,
        media_type="application/octet-stream",
        headers={
            "X-Rows": str(shape[0]),
            "X-Cols": str(shape[1]),
            "X-Frame-Index": str(frame_index),
            "X-Dtype": "float32",
            "X-Order": "C",
        },
    )


@app.websocket("/api/v1/simulations/{job_id}/events")
async def simulation_events(websocket: WebSocket, job_id: str) -> None:
    try:
        record = job_manager.get(job_id)
    except KeyError:
        await websocket.close(code=4404, reason="unknown job")
        return
    await websocket.accept()
    queue = job_manager.subscribe(job_id)
    await websocket.send_json(
        {"type": "snapshot", "job": record.public().model_dump(mode="json")}
    )
    if record.status in {"complete", "failed", "cancelled"}:
        await websocket.send_json(
            {
                "type": f"job_{record.status if record.status != 'complete' else 'completed'}",
                "job_id": job_id,
                "error": record.error,
            }
        )
        return
    try:
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=20.0)
                await websocket.send_json(event)
                if event["type"] in {"job_completed", "job_failed", "job_cancelled"}:
                    return
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "heartbeat", "job_id": job_id})
    except WebSocketDisconnect:
        pass
    finally:
        job_manager.unsubscribe(job_id, queue)


class RampProcessRequest(BaseModel):
    frame_id: str | None = None


@app.get("/api/v1/datasets")
def datasets() -> dict:
    return {
        "datasets": [
            {
                "id": "ramp_adc",
                "name": "77GHz AWR1843 Raw ADC",
                "status": dataset_status(),
            }
        ]
    }


@app.get("/api/v1/datasets/ramp/status")
def ramp_status() -> dict:
    return {**dataset_status(), "radar_config": RAMP_CONFIG.__dict__}


@app.get("/api/v1/datasets/ramp/sequences")
def ramp_sequences() -> dict:
    return {"sequences": list_sequences()}


@app.get("/api/v1/datasets/ramp/frames")
def ramp_frames(limit: int = Query(50, ge=1, le=200)) -> dict:
    return {"frames": list_frames(limit)}


@app.post("/api/v1/datasets/ramp/process")
async def ramp_process(request: RampProcessRequest) -> dict:
    result = await asyncio.to_thread(process_frame, request.frame_id)
    return result.metadata


@app.get("/api/v1/datasets/ramp/results/{result_id}")
def ramp_result(result_id: str) -> dict:
    try:
        return get_result(result_id).metadata
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown result") from exc


@app.get("/api/v1/datasets/ramp/results/{result_id}/rd")
def ramp_result_rd(
    result_id: str,
    view: str = Query("measured", pattern="^(measured|ideal)$"),
    channel: str = Query("combined"),
) -> Response:
    try:
        array = np.ascontiguousarray(result_array(result_id, view, channel), dtype="<f4")
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown result") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid channel: {channel}") from exc
    return Response(
        content=array.tobytes(order="C"),
        media_type="application/octet-stream",
        headers={
            "X-Rows": str(array.shape[0]),
            "X-Cols": str(array.shape[1]),
            "X-Dtype": "float32",
            "X-Order": "C",
        },
    )


@app.get("/api/v1/datasets/ramp/results/{result_id}/camera")
def ramp_result_camera(result_id: str) -> FileResponse:
    try:
        result = get_result(result_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown result") from exc
    if result.camera is None or not result.camera.exists():
        raise HTTPException(status_code=404, detail="camera image unavailable")
    return FileResponse(result.camera)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if (FRONTEND_DIST / "assets").is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="frontend-assets")


@app.get("/{full_path:path}", include_in_schema=False)
def frontend_application(full_path: str) -> FileResponse:
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API route not found")
    if not FRONTEND_DIST.is_dir():
        raise HTTPException(status_code=503, detail="frontend build is unavailable; run npm run build")
    requested = (FRONTEND_DIST / full_path).resolve()
    if FRONTEND_DIST.resolve() in requested.parents and requested.is_file():
        return FileResponse(requested)
    return FileResponse(
        FRONTEND_DIST / "index.html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )
