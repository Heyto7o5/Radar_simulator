from __future__ import annotations

import asyncio
import logging
import json
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .models import DerivedMetrics, FrameMetadata, JobPublic, SimulationRequest
from .physics import derived_metrics, request_warnings
from .simulator import save_frame, simulate_frame
from .unified.legacy_config import convert_legacy
from .unified.session import SimulationSession
from .unified.engine import SimulationCancelled


logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runtime_root() -> Path:
    configured = os.environ.get("RADAR_RUNTIME_ROOT", ".runtime")
    return Path(configured).expanduser().resolve()


@dataclass
class JobRecord:
    job_id: str
    request: SimulationRequest
    derived: DerivedMetrics
    warnings: list[str]
    status: str = "queued"
    progress: float = 0.0
    error: str | None = None
    created_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    completed_at: str | None = None
    frames: list[FrameMetadata] = field(default_factory=list)
    frame_paths: dict[int, Path] = field(default_factory=dict)
    complex_paths: dict[int, Path] = field(default_factory=dict)
    cancelled: bool = False
    subscribers: set[asyncio.Queue] = field(default_factory=set)

    def public(self) -> JobPublic:
        return JobPublic(
            job_id=self.job_id,
            status=self.status,
            progress=self.progress,
            error=self.error,
            created_at=self.created_at,
            started_at=self.started_at,
            completed_at=self.completed_at,
            derived_metrics=self.derived,
            warnings=self.warnings,
            frames=self.frames,
        )


class JobManager:
    def __init__(self) -> None:
        self.jobs: dict[str, JobRecord] = {}
        self.tasks: dict[str, asyncio.Task] = {}

    def create(self, request: SimulationRequest) -> JobRecord:
        job_id = uuid.uuid4().hex
        record = JobRecord(
            job_id=job_id,
            request=request,
            derived=derived_metrics(request.radar),
            warnings=request_warnings(request),
        )
        self.jobs[job_id] = record
        self.tasks[job_id] = asyncio.create_task(self._run(record))
        return record

    def get(self, job_id: str) -> JobRecord:
        try:
            return self.jobs[job_id]
        except KeyError as exc:
            raise KeyError(f"unknown job: {job_id}") from exc

    async def _notify(self, record: JobRecord, payload: dict[str, Any]) -> None:
        dead: list[asyncio.Queue] = []
        for queue in tuple(record.subscribers):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                dead.append(queue)
        for queue in dead:
            record.subscribers.discard(queue)

    async def _run(self, record: JobRecord) -> None:
        job_dir = runtime_root() / "jobs" / record.job_id
        try:
            job_dir.mkdir(parents=True,exist_ok=True)
            (job_dir / "request.json").write_text(record.request.model_dump_json(indent=2))
            common, migration_notes=convert_legacy(record.request)
            session=SimulationSession(common)
            (job_dir / 'simulation_config.json').write_text(common.model_dump_json(indent=2))
            (job_dir / "run_manifest.json").write_text(json.dumps({
                "job_id": record.job_id,"data_origin":"ideal_simulation","coordinate_frame":"NED_m",
                "simulation_engine":"unified","simulation_config":"simulation_config.json",
                "compatibility_notes":migration_notes,
                "velocity_convention":"positive closing","target_motion":"linear translation",
                "calibration":{"source":"none","fitted_parameters":[]},
                "analytic_validation":"truth-assisted local peaks and true Doppler correction; not independent detection",
                "blind_detection":"global RD nonmaximum suppression, no truth input; apparent Doppler correction only",
                "waveform":"pulsed LFM","processing":"matched filter, Hann slow-time FFT",
                "clutter":"world-fixed finite demonstration patch; not calibrated land scattering"
            },ensure_ascii=False,indent=2)+"\n")
            record.status = "running"
            record.started_at = _utc_now()
            await self._notify(record, {"type": "job_started", "job_id": record.job_id})
            total = record.request.sequence.frame_count
            for frame_index in range(total):
                if record.cancelled:
                    record.status = "cancelled"
                    record.completed_at = _utc_now()
                    await self._notify(
                        record, {"type": "job_cancelled", "job_id": record.job_id}
                    )
                    return
                result = await asyncio.to_thread(simulate_frame, record.request, frame_index,
                                                 session=session,cancelled=lambda:record.cancelled)
                if record.cancelled:raise SimulationCancelled()
                display_path, complex_path = await asyncio.to_thread(save_frame, result, job_dir)
                record.frames.append(result.metadata)
                record.frame_paths[frame_index] = display_path
                record.complex_paths[frame_index] = complex_path
                record.progress = (frame_index + 1) / total
                await self._notify(
                    record,
                    {
                        "type": "frame_ready",
                        "job_id": record.job_id,
                        "frame_index": frame_index,
                        "progress": record.progress,
                    },
                )
                await self._notify(
                    record,
                    {
                        "type": "progress",
                        "job_id": record.job_id,
                        "progress": record.progress,
                    },
                )
            record.status = "complete"
            record.completed_at = _utc_now()
            await self._notify(
                record, {"type": "job_completed", "job_id": record.job_id}
            )
        except SimulationCancelled:
            record.status='cancelled';record.completed_at=_utc_now()
            await self._notify(record,{'type':'job_cancelled','job_id':record.job_id})
        except asyncio.CancelledError:
            record.status = "cancelled"
            record.completed_at = _utc_now()
            raise
        except Exception as exc:  # pragma: no cover - exercised through API integration
            logger.exception("simulation job %s failed", record.job_id)
            record.status = "failed"
            record.error = str(exc)
            record.completed_at = _utc_now()
            await self._notify(
                record,
                {"type": "job_failed", "job_id": record.job_id, "error": record.error},
            )

    async def cancel(self, job_id: str) -> JobRecord:
        record = self.get(job_id)
        record.cancelled = True
        if record.status == "queued":
            record.status = "cancelled"
            record.completed_at = _utc_now()
        return record

    def subscribe(self, job_id: str) -> asyncio.Queue:
        record = self.get(job_id)
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        record.subscribers.add(queue)
        return queue

    def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        if job_id in self.jobs:
            self.jobs[job_id].subscribers.discard(queue)

    def clear(self) -> None:
        for task in self.tasks.values():
            if not task.done():
                task.cancel()
        self.tasks.clear()
        self.jobs.clear()
        root = runtime_root() / "jobs"
        if root.exists():
            shutil.rmtree(root)


job_manager = JobManager()


def frame_bytes(record: JobRecord, frame_index: int) -> tuple[bytes, tuple[int, int]]:
    path = record.frame_paths.get(frame_index)
    if path is None or not path.exists():
        raise FileNotFoundError(frame_index)
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    contiguous = np.ascontiguousarray(array, dtype="<f4")
    return contiguous.tobytes(order="C"), contiguous.shape
