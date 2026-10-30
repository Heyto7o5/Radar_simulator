"""Small finite-surface SFCW jobs, independent of measured-data adapters."""
from __future__ import annotations
import asyncio
import json
import os
import uuid
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from .unified.sfcw import SurfaceSimulationRequest, SFCWSession

router=APIRouter(prefix="/api/v1/surface-simulations",tags=["finite-surface SFCW"])


def run_surface_request(request: SurfaceSimulationRequest) -> dict:
    frame=SFCWSession(request).frame(0)
    signal=frame.signal
    path_counts=frame.metadata['path_counts']
    method=frame.metadata['method']
    job_id=uuid.uuid4().hex
    root=Path(os.environ.get("RADAR_RUNTIME_ROOT",".runtime"))/"surface_jobs"/job_id
    root.mkdir(parents=True,exist_ok=False)
    # Binary protocol explicitly little endian complex64; C-order [tx,rx,f].
    signal.astype("<c8").tofile(root/"spectrum.c64")
    (root/"request.json").write_text(request.model_dump_json(indent=2))
    summary={"job_id":job_id,"status":"complete","shape":list(signal.shape),
        "dtype":"complex64_le","axes":["tx","rx","frequency"],"order":"C",
        "frequencies_hz":request.frequencies_hz,"coordinate_frame":"NED_m",
        "path_counts":path_counts,
        "has_doppler_axis":False,"absolute_power_calibrated":False,
        "method":method,
        "spectrum_url":f"/api/v1/surface-simulations/{job_id}/spectrum"}
    (root/"metadata.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary


@router.post("")
async def create_surface_simulation(request: SurfaceSimulationRequest) -> dict:
    return await asyncio.to_thread(run_surface_request,request)


@router.get("/{job_id}/spectrum")
def spectrum(job_id: str) -> FileResponse:
    try:
        canonical=uuid.UUID(job_id).hex
    except ValueError as exc:
        raise HTTPException(404,"unknown surface job") from exc
    root=Path(os.environ.get("RADAR_RUNTIME_ROOT",".runtime"))/"surface_jobs"/canonical
    path=root/"spectrum.c64"
    if not path.exists():raise HTTPException(404,"unknown surface job")
    meta=json.loads((root/"metadata.json").read_text())
    return FileResponse(path,media_type="application/octet-stream",headers={
        "X-Shape":",".join(map(str,meta["shape"])),"X-Dtype":"complex64_le","X-Order":"C"})
