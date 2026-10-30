"""UI acquisition planning, geometry preview and lightweight display products.

This module does not change the scattering model. Auto mode resolves receive
sampling from the submitted geometry and shows the exact resulting configuration.
"""
from copy import deepcopy
from functools import lru_cache
import math
from pathlib import Path
import re

import numpy as np
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response
from pydantic import Field

from .models import C0
from .unified.schema import StrictModel, Simulation, Scene, Platform, Sequence, Array, PulseLFM, FMCW
from .unified.scene import compile_scene
from .unified.geometry import build_layout
from .unified.jobs import manager, preflight
from .unified.presets import high_altitude

router=APIRouter(prefix='/api/v1/workbench',tags=['simulation workbench'])


class PreviewRequest(StrictModel):
    request: dict
    auto_acquisition: bool=True


def presets():
    result={}
    for kind in ['pulsed_lfm','fmcw']:
        request=high_altitude(kind).model_dump(mode='json')
        request['name']='高空组合 · '+('LFM' if kind=='pulsed_lfm' else 'FMCW')+'（交互参数）'
        request['array'].update(layout='ula',num_tx=2,num_rx=4)
        request['processing'].update(azimuth_deg=list(range(-20,21,2)),elevation_deg=[0.],save_adc=False)
        request['sequence'].update(frame_count=3,frame_interval_s=.1)
        request['scene']['objects'][0]['divisions']=2
        w=request['waveform'];w['loops']=32
        if kind=='pulsed_lfm':w.update(bandwidth_hz=10e6,sample_rate_hz=20e6,samples=512)
        else:w.update(samples=512,sample_rate_hz=10e6,slope_hz_s=10e6/(512/10e6))
        result['mixed_'+kind]=request
    points=deepcopy(result['mixed_pulsed_lfm']);points['name']='点目标 · 脉冲 LFM'
    points['scene']['terrain']=None
    points['scene']['objects']=[{'kind':'point','id':f'T{i+1}','position_ned_m':pos,
        'velocity_ned_mps':[-10,8,0] if i==1 else [0,0,0],'amplitude':1.}
        for i,pos in enumerate([[550,-120,0],[850,0,0],[1100,160,0]])]
    choices={'mixed_pulsed_lfm':result['mixed_pulsed_lfm'],'points_pulsed_lfm':points,'mixed_fmcw':result['mixed_fmcw']}
    for kind in ['pulsed_lfm','fmcw']:
        standard=high_altitude(kind).model_dump(mode='json')
        standard['name']='高空组合 · '+('LFM' if kind=='pulsed_lfm' else 'FMCW')+'（标准参数）'
        standard['processing']['save_adc']=False
        choices['standard_'+kind]=standard
    return choices


def positive(value,name):
    v=float(value)
    if not math.isfinite(v) or v<=0:raise ValueError(f'{name} 必须为有限正数')
    return v


def resolve_request(draft: dict, auto_acquisition: bool):
    raw=deepcopy(draft);notes=[]
    if not auto_acquisition:return Simulation.model_validate(raw),notes
    scene=Scene.model_validate(raw.get('scene',{}));platform=Platform.model_validate(raw.get('platform',{}))
    sequence=Sequence.model_validate(raw.get('sequence',{}));array=Array.model_validate(raw.get('array',{}))
    compiled=compile_scene(scene)
    kind=raw.get('waveform',{}).get('kind','pulsed_lfm')
    if kind not in ('pulsed_lfm','fmcw'):raise ValueError('不支持的波形')
    w=(PulseLFM() if kind=='pulsed_lfm' else FMCW()).model_dump()
    w.update(raw.get('waveform',{}))
    slot=positive(w['slot_period_s'],'发射间隔');loops=positive(w['loops'],'CPI 循环数')
    period=array.loop_period_s or array.num_tx*slot
    if sequence.frame_count>1 and sequence.frame_interval_s<loops*period:
        sequence=sequence.model_copy(update={'frame_interval_s':loops*period})
        notes.append('帧间隔已增大到一个完整 CPI，避免采集时间重叠')
    times=[sequence.start_time_s,sequence.start_time_s+(sequence.frame_count-1)*sequence.frame_interval_s+loops*period]
    radii=[]
    for t in times:
        target=compiled.positions_ned_m+compiled.velocities_ned_mps*t
        radar=np.asarray(platform.position_ned_m)+np.asarray(platform.velocity_ned_mps)*t
        radii.append(np.linalg.norm(target-radar,axis=1))
    delta=compiled.positions_ned_m-np.asarray(platform.position_ned_m)
    velocity=compiled.velocities_ned_mps-np.asarray(platform.velocity_ned_mps)
    squared_speed=np.sum(velocity**2,axis=1)
    closest=np.clip(np.divide(-np.sum(delta*velocity,axis=1),squared_speed,
        out=np.full(len(delta),times[0]),where=squared_speed>0),*times)
    rmin=float(np.min(np.linalg.norm(delta+velocity*closest[:,None],axis=1)));rmax=float(np.max(radii))
    if rmin<1e-3:raise ValueError('目标或地形与雷达位置相交')
    fs=positive(w['sample_rate_hz'],'采样率')
    if kind=='pulsed_lfm':
        bandwidth=positive(w['bandwidth_hz'],'带宽');pulse=positive(w['pulse_width_s'],'脉宽')
        fs=max(fs,1.25*bandwidth)
        guard=max(4/fs,1e-6)
        start=max(0.,2*rmin/C0-guard)
        duration=2*(rmax-rmin)/C0+pulse+2*guard
        samples=2**math.ceil(math.log2(max(8,duration*fs)))
        if start+samples/fs>slot:
            # A power-of-two gate is an optimization, not a reason to reject a fitting physical gate.
            samples=int(math.ceil(duration*fs))
        if start+samples/fs>slot:
            raise ValueError('当前 PRF 的接收时间不足以覆盖场景，请降低 PRF 或缩小距离范围')
        w.update(adc_start_s=start,samples=samples,sample_rate_hz=fs)
    else:
        original_samples=positive(w['samples'],'采样点数')
        bandwidth=positive(w['slope_hz_s'],'调频斜率')*original_samples/fs
        samples=2**math.ceil(math.log2(max(128,original_samples,1.25*2*bandwidth*rmax/C0)))
        start=2*rmax/C0+max(4/fs,2e-6)
        available=slot-start-2e-6
        if available<=0:raise ValueError('FMCW 发射间隔不足以接收场景回波，请增加发射间隔')
        fs=max(fs,math.ceil(samples/available/1e6)*1e6)
        duration=samples/fs
        w.update(adc_start_s=start,samples=samples,sample_rate_hz=fs,
                 slope_hz_s=bandwidth/duration,ramp_s=start+duration)
    resolution=C0/(2*bandwidth)
    raw['waveform']=w
    raw.setdefault('processing',{})['range_limits_m']=[max(0.,rmin-4*resolution),rmax+4*resolution]
    raw['sequence']=sequence.model_dump()
    notes.append('接收窗、采样点数和显示距离按当前场景自动计算；实际参数已随任务保存')
    return Simulation.model_validate(raw),notes


def preview(body: PreviewRequest):
    request,notes=resolve_request(body.request,body.auto_acquisition)
    scene,estimate=preflight(request)
    layout=build_layout(request.array,request.waveform,request.platform)
    # A bounded geometry preview; the entire compiled scene still drives simulation.
    stride=max(1,math.ceil(len(scene.amplitudes)/1500));indices=np.arange(0,len(scene.amplitudes),stride)
    vertices=scene.positions_ned_m[indices]
    t=request.sequence.start_time_s
    if request.platform.trajectory is not None:
        from .unified.trajectory import TrajectoryEvaluator
        pos,vel,basis,_=TrajectoryEvaluator(request.platform.trajectory).at(t)
    else:
        pos=np.asarray(request.platform.position_ned_m)+t*np.asarray(request.platform.velocity_ned_mps)
        vel=np.asarray(request.platform.velocity_ned_mps);basis=layout.basis_ned
    state={'time_s':t,'platform_position_ned_m':pos.tolist(),'platform_velocity_ned_mps':vel.tolist(),
        'basis_ned':basis.tolist(),'coverage':None,'pattern':None,'array':layout.metadata(),
        'objects':[{'id':o.id,'position_ned_m':(np.asarray(o.position_ned_m)+t*np.asarray(o.velocity_ned_mps)).tolist()} for o in request.scene.objects]}
    return {'request':request.model_dump(mode='json'),'notes':notes,'estimate':estimate,'state':state,
        'geometry':{'positions_ned_m':vertices.tolist(),'object_index':scene.object_index[indices].tolist(),
                    'objects':scene.objects,'terrain':scene.terrain,'preview_stride':stride,'scatterer_count':len(scene.amplitudes)},
        'array':layout.metadata(),
        'derived':{'carrier_hz':request.waveform.carrier_hz,'range_resolution_m':C0/(2*request.waveform.bandwidth_hz),
            'max_unambiguous_velocity_mps':request.waveform.wavelength_m/(4*layout.loop_period_s),
            'velocity_cell_mps':request.waveform.wavelength_m/(2*request.waveform.loops*layout.loop_period_s),
            'samples':request.waveform.samples,'loops':request.waveform.loops,'channels':request.array.num_tx*request.array.num_rx}}


@router.get('/presets')
def get_presets():return presets()


@router.post('/preview')
def scene_preview(body: PreviewRequest):
    try:return preview(body)
    except (ValueError,TypeError,KeyError) as exc:raise HTTPException(422,str(exc)) from exc


def read_product(job,index,name):
    if not 0<=index<len(job.frames):raise ValueError('帧尚未就绪')
    info=job.frames[index]['products'][name]
    return np.memmap(job.root/f'frame_{index:04d}'/info['file'],mode='r',dtype=info['dtype'],shape=tuple(info['shape']))


def parse_channel(channel,rx_count,tx_count):
    if channel=='combined':return None
    match=re.fullmatch(r'tx(\d+)_rx(\d+)',channel)
    if not match:raise ValueError('无效通道')
    tx,rx=map(int,match.groups())
    if tx>=tx_count or rx>=rx_count:raise ValueError('通道超出阵列范围')
    return rx,tx


@lru_cache(maxsize=128)
def reference_peak(path,dtype,shape):
    data=np.memmap(path,mode='r',dtype=dtype,shape=shape)
    peak=0.
    for start in range(0,len(data),32):
        chunk=np.asarray(data[start:start+32])
        power=abs(chunk.astype(np.complex128))**2 if np.iscomplexobj(chunk) else chunk
        peak=max(peak,float(np.max(power)))
    return max(peak,1e-30)


def db_response(power,reference,*,floor=-100.):
    db=np.ascontiguousarray(10*np.log10(np.maximum(np.asarray(power,dtype=float)/reference,10**(floor/10))),dtype='<f4')
    if db.ndim!=2 or not np.isfinite(db).all():raise ValueError('显示矩阵无效')
    return Response(db.tobytes(order='C'),media_type='application/octet-stream',headers={
        'X-Rows':str(db.shape[0]),'X-Cols':str(db.shape[1]),'X-Dtype':'float32','X-Order':'C',
        'X-Reference-Power':str(reference)})


def scene_view(job,index,view,channel='combined',elevation_index=0,range_index=None,doppler_index=None):
    meta=job.frames[index] if 0<=index<len(job.frames) else None
    if meta is None:raise ValueError('帧尚未就绪')
    if view=='rd':
        item=read_product(job,index,'rd_channels')
        selected=parse_channel(channel,item.shape[2],item.shape[3])
        power=read_product(job,index,'rd_power') if selected is None else abs(item[:,:,selected[0],selected[1]].astype(np.complex128))**2
        power=power.T;reference_name='rd_power'
    elif view=='ra':
        item=read_product(job,index,'ra_power')
        if not 0<=elevation_index<item.shape[1]:raise ValueError('俯仰切片超出范围')
        power=item[:,elevation_index,:].T;reference_name='ra_power'
    elif view=='angle':
        item=read_product(job,index,'rad')
        if range_index is None or doppler_index is None:
            rd=read_product(job,index,'rd_power');range_index,doppler_index=map(int,np.unravel_index(np.argmax(rd),rd.shape))
        if not 0<=range_index<item.shape[0] or not 0<=doppler_index<item.shape[1]:raise ValueError('角度切片超出范围')
        power=abs(item[range_index,doppler_index].astype(np.complex128))**2;reference_name='rad'
    else:raise ValueError('不支持的显示视图')
    ref=job.frames[0]['products'][reference_name]
    reference=reference_peak(str(job.root/'frame_0000'/ref['file']),ref['dtype'],tuple(ref['shape']))
    return db_response(power,reference)


@router.get('/jobs/{job_id}/request')
def saved_request(job_id: str):
    try:return manager.get(job_id).request.model_dump(mode='json')
    except KeyError as exc:raise HTTPException(404,'任务不存在') from exc


@router.get('/jobs/{job_id}/frames/{index}/view')
def frame_view(job_id: str,index: int,view: str=Query('rd',pattern='^(rd|ra|angle)$'),channel: str='combined',
               elevation_index: int=0,range_index: int|None=None,doppler_index: int|None=None):
    try:return scene_view(manager.get(job_id),index,view,channel,elevation_index,range_index,doppler_index)
    except KeyError as exc:raise HTTPException(404,'任务或结果不存在') from exc
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
