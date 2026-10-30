"""Standalone background synthesis at the declared range-gated IQ stage."""
import asyncio
import json
import os
import uuid
from pathlib import Path
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

router=APIRouter(prefix='/api/v1/background-simulations',tags=['range-gated backgrounds'])

from .unified.background import BackgroundRequest, BackgroundSession

def run(request):
    frame=BackgroundSession(request).frame(0)
    iq,rd=frame.iq,frame.rd
    frequency=np.asarray(frame.metadata['axes']['raw_slow_frequency_hz'])
    key=uuid.uuid4().hex
    folder=Path(os.environ.get('RADAR_RUNTIME_ROOT','.runtime')).resolve()/'background_jobs'/key
    folder.mkdir(parents=True,exist_ok=False)
    iq.astype('<c8').tofile(folder/'iq.c64');rd.astype('<c8').tofile(folder/'rd.c64')
    (folder/'request.json').write_text(request.model_dump_json(indent=2))
    meta={'job_id':key,'status':'complete','is_real':False,'family':request.model.family,
        'domain':'range_gated_iq','iq_shape':list(iq.shape),'rd_shape':list(rd.shape),
        'range_m':list(request.model.range_m),'doppler_hz':frequency.tolist(),
        'dtype':'complex64_le','order':'C','receiver_noise_figure_calibrated':False,
        'limits':['not raw fast-time ADC','not an exact sea/ground geometry replay',
                  'do not add this output directly to raw ADC','stationary batch; no multi-call phase continuity promise'],
        'iq_url':f'/api/v1/background-simulations/{key}/iq','rd_url':f'/api/v1/background-simulations/{key}/rd'}
    (folder/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2))
    return meta

@router.post('')
async def create(request: BackgroundRequest):
    return await asyncio.to_thread(run,request)

@router.get('/{job_id}/{view}')
def download(job_id:str,view:str):
    if view not in ['iq','rd']:raise HTTPException(404,'unknown product')
    try:key=uuid.UUID(job_id).hex
    except ValueError as exc:raise HTTPException(404,'unknown job') from exc
    folder=Path(os.environ.get('RADAR_RUNTIME_ROOT','.runtime')).resolve()/'background_jobs'/key
    if not (folder/'metadata.json').exists():raise HTTPException(404,'unknown job')
    meta=json.loads((folder/'metadata.json').read_text())
    return FileResponse(folder/(view+'.c64'),media_type='application/octet-stream',
        headers={'X-Dtype':'complex64_le','X-Shape':','.join(map(str,meta[view+'_shape'])),'X-Order':'C'})
