"""A single scene API independent of waveform and dataset adapters."""
import asyncio
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .schema import Simulation
from .sfcw import SurfaceSimulationRequest
from .background import BackgroundRequest, ClutterSimulationRequest
from .separable_if import SeparableIFRequest
from .jobs import manager
from .presets import presets

router=APIRouter(prefix='/api/v1',tags=['unified scene simulation'])


@router.post('/scene-check')
def check_configuration(request: dict):
    from ..configuration import inspect_configuration
    return inspect_configuration(request)


@router.post('/scene-motion-check')
def check_motion(request: dict):
    from .applicability import inspect_configuration
    from .motion_check import motion_error_report
    report=inspect_configuration(request)
    if report['status']=='unsupported':return report
    try:
        report['motion_check']=motion_error_report(Simulation.model_validate(request))
    except ValueError as exc:
        report['status']='unsupported';report['motion_error']=str(exc)
        return report
    if report['motion_check']['status']=='risk':report['status']='risk'
    return report


@router.get('/scene-presets')
def get_presets():
    return presets()


@router.post('/scene-simulations',status_code=202)
async def create(request: Simulation | SurfaceSimulationRequest | BackgroundRequest | ClutterSimulationRequest | SeparableIFRequest):
    try:return manager.create(request).public()
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    except RuntimeError as exc:raise HTTPException(429,str(exc)) from exc


def get_job(job_id):
    try:return manager.get(job_id)
    except KeyError as exc:raise HTTPException(404,'unknown scene job') from exc


@router.get('/scene-simulations/{job_id}')
def status(job_id: str):
    return get_job(job_id).public()


@router.delete('/scene-simulations/{job_id}')
def cancel(job_id: str):
    get_job(job_id)
    return manager.cancel(job_id).public()


@router.get('/scene-simulations/{job_id}/frames/{frame_index}/data/{product}')
def data(job_id: str,frame_index: int,product: str):
    job=get_job(job_id)
    if not 0<=frame_index<len(job.frames):raise HTTPException(404,'frame not ready')
    manifest=job.frames[frame_index]['products']
    if product not in manifest:raise HTTPException(404,'product not saved')
    info=manifest[product]
    return FileResponse(job.root/f'frame_{frame_index:04d}'/info['file'],media_type='application/octet-stream',headers={
        'X-Shape':','.join(map(str,info['shape'])),'X-Dtype':info['dtype'],'X-Order':'C','X-Frame-Index':str(frame_index)})


@router.websocket('/scene-simulations/{job_id}/events')
async def events(websocket: WebSocket,job_id: str):
    try:job=manager.get(job_id)
    except KeyError:
        await websocket.close(code=4404,reason='unknown scene job');return
    await websocket.accept()
    queue=asyncio.Queue(maxsize=32)
    job.subscribers.add(queue)
    try:
        await websocket.send_json({'type':'snapshot','job':job.public()})
        if job.status in ('complete','failed','cancelled'):
            await websocket.send_json({'type':{'complete':'job_completed','failed':'job_failed','cancelled':'job_cancelled'}[job.status],
                                       'job_id':job.id,'error':job.error})
            return
        while True:
            try:message=await asyncio.wait_for(queue.get(),timeout=20.)
            except asyncio.TimeoutError:message={'type':'heartbeat','job_id':job.id}
            await websocket.send_json(message)
            if message['type'] in ('job_completed','job_failed','job_cancelled'):break
    except WebSocketDisconnect:
        pass
    finally:
        job.subscribers.discard(queue)
