"""Generate statistical range-gated IQ through the platform, without input observations."""
import asyncio,json,os,uuid
from pathlib import Path
import numpy as np
from fastapi import APIRouter,HTTPException
from fastapi.responses import FileResponse

router=APIRouter(prefix='/api/v1/clutter-simulations',tags=['statistical clutter'])


from .unified.background import ClutterSimulationRequest, BackgroundSession


def run_clutter_request(request: ClutterSimulationRequest):
    frame=BackgroundSession(request).frame(0)
    iq,rd=frame.iq,frame.rd
    f=np.asarray(frame.metadata['axes']['raw_slow_frequency_hz'])
    job_id=uuid.uuid4().hex
    root=Path(os.environ.get('RADAR_RUNTIME_ROOT','.runtime'))/'clutter_jobs'/job_id
    root.mkdir(parents=True,exist_ok=False)
    iq.astype('<c8').tofile(root/'iq.c64');rd[0].astype('<c8').tofile(root/'rd.c64')
    (root/'request.json').write_text(request.model_dump_json(indent=2))
    meta={'job_id':job_id,'status':'complete','iq_shape':list(iq.shape),'rd_shape':list(rd[0].shape),
        'doppler_hz':f.tolist(),'range_near_edge_m':list(request.model.range_near_edge_m),
        'dtype':'complex64_le','order':'C','source':'synthetic statistical clutter; not a recreated sea-surface realization',
        'iq_url':f'/api/v1/clutter-simulations/{job_id}/iq','rd_url':f'/api/v1/clutter-simulations/{job_id}/rd'}
    (root/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    return meta


@router.post('')
async def create(request:ClutterSimulationRequest):return await asyncio.to_thread(run_clutter_request,request)


@router.get('/{job_id}/{view}')
def download(job_id:str,view:str):
    if view not in ('iq','rd'):raise HTTPException(404,'unknown view')
    try:job_id=uuid.UUID(job_id).hex
    except ValueError as exc:raise HTTPException(404,'unknown job') from exc
    root=Path(os.environ.get('RADAR_RUNTIME_ROOT','.runtime'))/'clutter_jobs'/job_id
    if not (root/'metadata.json').exists():raise HTTPException(404,'unknown job')
    meta=json.loads((root/'metadata.json').read_text())
    return FileResponse(root/f'{view}.c64',media_type='application/octet-stream',headers={
        'X-Dtype':'complex64_le','X-Shape':','.join(map(str,meta[f'{view}_shape'])),'X-Order':'C'})
