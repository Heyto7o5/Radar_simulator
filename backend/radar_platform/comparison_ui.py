"""Real RAMP observations beside editable inputs to the unified scene simulator.

Camera images are served only for catalogued radar frames, directly from the
local official archive if not extracted. No camera geometry/CAD is invented.
"""
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time
import uuid
import zipfile

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from .ramp_dataset import dataset_root, camera_path, RAMP_SOURCE_URL
from .ramp_alignment import AnnotationTracks, read_ramp_adc, aligned_array
from .ramp_label_priors import relative_class_amplitude as _class_rcs_scale
from .unified.schema import StrictModel, Simulation, Acquisition
from .unified.geometry import build_layout
from .unified.processing import process_adc
from .unified.jobs import manager
from .unified.storage import write_json
from .workbench import db_response, parse_channel, read_product

router=APIRouter(prefix='/api/v1/comparisons/ramp',tags=['real / simulated workspace'])
CLASS_NAMES={0:'行人',2:'汽车',3:'摩托车',5:'公交车',7:'卡车',80:'骑行者'}


@lru_cache(maxsize=4)
def archive_cameras(path,mtime):
    result={}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            parts=Path(name).parts
            if len(parts)==4 and parts[0]=='Automotive' and parts[2]=='images_0' and Path(name).suffix.lower() in ('.jpg','.jpeg','.png'):
                try:key=(parts[1],int(Path(name).stem))
                except ValueError:continue
                if key in result:raise ValueError('配套相机帧编号重复，无法安全匹配')
                result[key]=name
    return result


def camera_source(path,sequence,index,root):
    local=camera_path(path)
    if local is not None and root.resolve() in local.resolve().parents:
        return {'file':str(local),'mime':'image/png' if local.suffix.lower()=='.png' else 'image/jpeg'}
    archive=root/'raw/Automotive.zip'
    if archive.exists():
        member=archive_cameras(str(archive),archive.stat().st_mtime_ns).get((sequence,index))
        if member:return {'archive':str(archive),'member':member,'mime':'image/png' if member.endswith('.png') else 'image/jpeg'}
    return None


_catalog_cache={}


def catalog():
    root=dataset_root()
    if root is None:return {'configured':False,'available':False,'sequences':[],'source_url':RAMP_SOURCE_URL},{}
    key=str(root.resolve());cached=_catalog_cache.get(key)
    if cached and time.monotonic()-cached[0]<20:return cached[1],cached[2]
    entries={};by_frame={}
    # Prefer the richer trajectory subset; retain other locally extracted frames too.
    for base in [root/'alignment_subset_20261005',root/'extracted']:
        if not base.exists():continue
        for path in sorted(base.glob('**/radar_raw_frame/*.mat')):
            if root.resolve() not in path.resolve().parents:continue
            try:index=int(path.stem)
            except ValueError:continue
            sequence=path.parent.parent.name
            identity=(sequence,index)
            if identity in by_frame:continue
            relative=str(path.relative_to(root));frame_id=hashlib.sha256(relative.encode()).hexdigest()[:24]
            camera=camera_source(path,sequence,index,root)
            entries[frame_id]={'id':frame_id,'path':path,'sequence':sequence,'frame_index':index,'camera':camera,
                'name':f'{index:06d}','source_frame':str(Path('Automotive')/sequence/'radar_raw_frame'/path.name)}
            by_frame[identity]=frame_id
            if len(entries)>=2000:break
        if len(entries)>=2000:break
    sequences=[]
    for sequence in sorted({v['sequence'] for v in entries.values()}):
        frames=sorted([v for v in entries.values() if v['sequence']==sequence],key=lambda v:v['frame_index'])
        sequences.append({'id':sequence,'name':sequence,'count':len(frames),'frames':[
            {k:v[k] for k in ['id','name','frame_index']}|{'camera_available':v['camera'] is not None} for v in frames]})
    public={'configured':True,'available':bool(entries),'sequences':sequences,'frames':len(entries),
            'source_url':RAMP_SOURCE_URL,'camera_scope':'official same-frame-name camera; no radar-to-image projection or CAD inferred'}
    _catalog_cache[key]=(time.monotonic(),public,entries)
    return public,entries


def real_configuration():
    a=aligned_array(-1)
    return Acquisition.model_validate({'waveform':{'kind':'fmcw','start_hz':77e9,'slope_hz_s':21e12,'ramp_s':32e-6,
        'slot_period_s':60e-6,'adc_start_s':0.,'samples':128,'loops':255,'sample_rate_hz':4e6},
        'platform':{'position_ned_m':[0,0,0],'velocity_ned_mps':[0,0,0],'boresight_ned':[1,0,0],
                    'right_hint_ned':[0,1,0],'beam_half_angle_deg':89},
        'array':{'layout':'custom','num_tx':2,'num_rx':4,'tx_local_m':a.tx_ned_m.tolist(),'rx_local_m':a.rx_ned_m.tolist(),
                 'tx_offsets_s':a.tx_offsets_s.tolist(),'loop_period_s':a.loop_period_s},
        'processing':{'range_limits_m':[1,24],'range_window':'hamming','doppler_window':'hamming','spatial_window':'hamming',
                      'azimuth_deg':[0.],'elevation_deg':[0.],'tdm_compensation':True,'save_adc':False}})


@lru_cache(maxsize=24)
def tracks(path,mtime):return AnnotationTracks(Path(path))


def fixed_gain(root):
    path=root.parent/'alignment_campaign/20261005/ramp_v3/power_alignment.json'
    if path.exists():
        data=json.loads(path.read_text());gain=float(data['gains']['physical_smooth_roi_gain'])
        if np.isfinite(gain) and gain>0:return gain,'沿用已有固定相对幅度标定；不是绝对 RCS'
    return 1.,'未加载相对幅度标定；仿真幅度由输入指定'


@dataclass
class Record:
    id: str
    frame_id: str
    root: Path
    metadata: dict
    template: dict|None
    camera: dict|None
    reference: float


records=OrderedDict()
_prepared={}


def prepare(frame_id):
    root=dataset_root()
    public,entries=catalog()
    if root is None or frame_id not in entries:raise FileNotFoundError('真实雷达帧不存在；没有使用合成数据替代')
    entry=entries[frame_id];path=entry['path'];key=(str(root.resolve()),frame_id,path.stat().st_mtime_ns)
    cached=_prepared.get(key)
    if cached in records:
        records.move_to_end(cached);return records[cached]
    cfg=real_configuration();adc=read_ramp_adc(path)
    result=process_adc(adc,cfg,build_layout(cfg.array,cfg.waveform,cfg.platform))
    folder=path.parent.parent/'text_labels'
    if folder.exists():
        track=tracks(str(folder),folder.stat().st_mtime_ns)
        labels=track.labels(entry['frame_index'],smooth=False)
        smooth=track.labels(entry['frame_index'],smooth=True)
    else:labels=[];smooth=[]
    gain,gain_description=fixed_gain(root)
    objects=[{'kind':'point','id':l['uid'],'position_ned_m':[l['y_m'],l['x_m'],0.],
        'velocity_ned_mps':l['velocity_ned_mps'],'amplitude':_class_rcs_scale(int(l['class_id']))*np.sqrt(gain)} for l in smooth]
    template=None
    if objects:
        template=Simulation.model_validate({**cfg.model_dump(),'name':f"RAMP {entry['sequence']} / {entry['name']}",
            'scene':{'objects':objects,'amplitude_reference_range_m':10.},'sequence':{'frame_count':1}}).model_dump(mode='json')
    record_id=uuid.uuid4().hex;output=root/'processed'/'workbench'/record_id;output.mkdir(parents=True,exist_ok=False)
    np.save(output/'rd_channels.npy',result.rd_channels,allow_pickle=False)
    np.save(output/'rd_power.npy',result.rd_power,allow_pickle=False)
    reference=max(float(result.rd_power.max()),1e-30)
    for l in labels:l['class_name']=CLASS_NAMES.get(int(l['class_id']),f"类别 {l['class_id']}")
    metadata={'record_id':record_id,'frame_id':frame_id,'source_frame':entry['source_frame'],'sequence':entry['sequence'],
        'frame_index':entry['frame_index'],'is_real':True,'camera_available':entry['camera'] is not None,
        'camera_url':f'/api/v1/comparisons/ramp/records/{record_id}/camera' if entry['camera'] else None,
        'labels':labels,'template':template,'reference_power':reference,'display_reference':'同一实测帧合并通道峰值；两侧及各通道共用',
        'range_m':result.range_m.tolist(),'velocity_mps':result.velocity_mps.tolist(),
        'channels':[f'tx{tx}_rx{rx}' for tx in range(2) for rx in range(4)],
        'source_url':RAMP_SOURCE_URL,'amplitude_note':gain_description,
        'target_note':'公开二维位置标注；无配套真实三维/材料模型。未知速度在默认输入中用 0 占位。',
        'adc_shape':list(adc.shape),'acquisition':cfg.model_dump(mode='json')}
    write_json(output/'metadata.json',metadata)
    record=Record(record_id,frame_id,output,metadata,template,entry['camera'],reference)
    records[record_id]=record;_prepared[key]=record_id
    while len(records)>24:
        old,_=records.popitem(last=False)
        for k,v in list(_prepared.items()):
            if v==old:_prepared.pop(k,None)
    return record


def get_record(record_id):
    if record_id not in records:raise KeyError('对比记录已失效，请重新载入实测帧')
    return records[record_id]


def measured_view(record,channel):
    selected=parse_channel(channel,4,2)
    if selected is None:power=np.load(record.root/'rd_power.npy',mmap_mode='r',allow_pickle=False)
    else:
        data=np.load(record.root/'rd_channels.npy',mmap_mode='r',allow_pickle=False)
        power=abs(data[:,:,selected[0],selected[1]].astype(np.complex128))**2
    return db_response(power.T,record.reference)


def check_request(record,request):
    if request.sequence.frame_count!=1 or request.sequence.start_time_s!=0:raise ValueError('实测对比按当前帧运行，帧数须为 1、起始时间须为 0')
    fixed=record.metadata['acquisition'];current=request.model_dump(mode='json')
    for key in ['waveform','array','platform','processing']:
        if current[key]!=fixed[key]:raise ValueError('实测对比的波形、阵列、平台与处理网格已锁定；请仅修改仿真场景输入')


class PrepareRequest(StrictModel):
    frame_id: str


class SimulateRequest(StrictModel):
    request: Simulation


@router.get('/catalog')
def get_catalog():
    try:return catalog()[0]
    except (OSError,ValueError,zipfile.BadZipFile) as exc:raise HTTPException(422,str(exc)) from exc


@router.post('/prepare')
async def prepare_frame(body: PrepareRequest):
    try:return (await asyncio.to_thread(prepare,body.frame_id)).metadata
    except FileNotFoundError as exc:raise HTTPException(404,str(exc)) from exc
    except (ValueError,OSError) as exc:raise HTTPException(422,str(exc)) from exc


@router.get('/records/{record_id}/camera')
def camera(record_id: str):
    try:source=get_record(record_id).camera
    except KeyError as exc:raise HTTPException(404,str(exc)) from exc
    if not source:raise HTTPException(404,'此帧没有同名配套相机图像')
    if 'file' in source:
        path=Path(source['file'])
        if not path.exists():raise HTTPException(404,'配套相机文件不可用')
        if path.stat().st_size>16_000_000:raise HTTPException(422,'相机图像过大')
        content=path.read_bytes()
    else:
        with zipfile.ZipFile(source['archive']) as archive:
            if archive.getinfo(source['member']).file_size>16_000_000:raise HTTPException(422,'相机图像过大')
            content=archive.read(source['member'])
    return Response(content,media_type=source['mime'],headers={'Cache-Control':'private, max-age=3600'})


@router.get('/records/{record_id}/rd')
def real_rd(record_id: str,channel: str='combined'):
    try:return measured_view(get_record(record_id),channel)
    except KeyError as exc:raise HTTPException(404,str(exc)) from exc
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc


@router.post('/records/{record_id}/simulate',status_code=202)
async def simulate(record_id: str,body: SimulateRequest):
    try:
        record=get_record(record_id);check_request(record,body.request)
        job=manager.create(body.request)
        # Attach immutable pairing to prevent querying a job against unrelated real data.
        job.comparison_record_id=record_id
        return job.public()
    except KeyError as exc:raise HTTPException(404,str(exc)) from exc
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
    except RuntimeError as exc:raise HTTPException(429,str(exc)) from exc


@router.get('/records/{record_id}/simulations/{job_id}/rd')
def simulated_rd(record_id: str,job_id: str,channel: str='combined'):
    try:
        record=get_record(record_id);job=manager.get(job_id)
        if getattr(job,'comparison_record_id',None)!=record_id:raise ValueError('该仿真不属于当前实测帧')
        channels=read_product(job,0,'rd_channels');selected=parse_channel(channel,4,2)
        power=read_product(job,0,'rd_power') if selected is None else abs(channels[:,:,selected[0],selected[1]].astype(np.complex128))**2
        return db_response(power.T,record.reference)
    except KeyError as exc:raise HTTPException(404,str(exc)) from exc
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc
