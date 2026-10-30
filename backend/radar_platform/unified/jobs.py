"""Small CPU job runner for the shared scene pipeline, without a database or service startup."""
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import uuid

from .schema import Simulation
from .engine import SimulationCancelled
from .session import session_from_config
from .sfcw import SurfaceSimulationRequest
from .background import BackgroundRequest, ClutterSimulationRequest
from .separable_if import SeparableIFRequest
from .preflight import preflight  # Compatibility import for existing callers.
from .storage import write_json

logger=logging.getLogger(__name__)


@dataclass
class Job:
    id: str
    request: Simulation | SurfaceSimulationRequest | BackgroundRequest | ClutterSimulationRequest | SeparableIFRequest
    root: Path
    estimate: dict
    frame_count: int
    data_origin: str
    status: str='queued'
    frames: list=field(default_factory=list)
    error: str|None=None
    cancelled: bool=False
    subscribers: set=field(default_factory=set)
    created_at: str=field(default_factory=lambda:datetime.now(timezone.utc).isoformat())

    def public(self):
        return {'job_id':self.id,'name':self.request.name,'status':self.status,'error':self.error,
                'progress':len(self.frames)/self.frame_count,'frames':self.frames,
                'frame_count':self.frame_count,'created_at':self.created_at,'estimate':self.estimate,
                'data_origin':self.data_origin,'is_real':False}


class Manager:
    def __init__(self):
        self.jobs={}
        self.tasks={}

    def create(self,request):
        if sum(j.status in ('queued','running') for j in self.jobs.values())>=2:
            raise RuntimeError('two scene jobs already active; cancel or wait')
        session=session_from_config(request.model_dump(mode='json'))
        job_id=uuid.uuid4().hex
        root=Path(os.environ.get('RADAR_RUNTIME_ROOT','.runtime')).resolve()/'scene_jobs'/job_id
        job=Job(job_id,session.request,root,session.estimate,session.frame_count,session.data_origin)
        from .provenance import source_snapshot
        job._source_archive,job._source_manifest=source_snapshot()
        self.jobs[job_id]=job
        self.tasks[job_id]=asyncio.create_task(self.run(job,session))
        return job

    def get(self,job_id):
        if job_id not in self.jobs:
            from ..workbench_archive import recover
            self.jobs[job_id]=recover(job_id)
        return self.jobs[job_id]

    def emit(self,job,kind,**extra):
        event={'type':kind,'job_id':job.id,**extra}
        for queue in tuple(job.subscribers):
            if queue.full():
                # Client gets a current snapshot rather than silently losing its terminal state.
                while not queue.empty():queue.get_nowait()
                queue.put_nowait({'type':'snapshot','job':job.public()})
            queue.put_nowait(event)

    async def run(self,job,session):
        try:
            await asyncio.to_thread(session.save_inputs,job.root)
            if hasattr(job,'_source_archive'):
                await asyncio.to_thread((job.root/'source.zip').write_bytes,job._source_archive)
                write_json(job.root/'source_manifest.json',job._source_manifest)
                del job._source_archive
            job.status='running'
            write_json(job.root/'job.json',job.public())
            self.emit(job,'job_started')
            for index in range(session.frame_count):
                if job.cancelled:break
                frame=await asyncio.to_thread(session.frame,index,cancelled=lambda:job.cancelled)
                if job.cancelled:break
                meta=await asyncio.to_thread(session.save_frame,job.root,frame)
                job.frames.append(meta)
                self.emit(job,'frame_ready',frame_index=index,progress=len(job.frames)/session.frame_count)
                write_json(job.root/'job.json',job.public())
            job.status='cancelled' if job.cancelled else 'complete'
            self.emit(job,'job_cancelled' if job.cancelled else 'job_completed')
        except SimulationCancelled:
            job.cancelled=True;job.status='cancelled'
            self.emit(job,'job_cancelled')
        except asyncio.CancelledError:
            job.cancelled=True;job.status='cancelled'
            self.emit(job,'job_cancelled')
            raise
        except Exception as exc:
            logger.exception('unified scene job %s failed',job.id)
            job.status='failed';job.error=str(exc)
            self.emit(job,'job_failed',error=job.error)
        finally:
            if job.root.exists():write_json(job.root/'job.json',job.public())
            self.tasks.pop(job.id,None)

    def cancel(self,job_id):
        job=self.get(job_id)
        if job.status in ('queued','running'):job.cancelled=True
        return job


manager=Manager()
