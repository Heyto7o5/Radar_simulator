"""Presentation and experiment services. Physics remains in the shared engine."""
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
import zipfile

import numpy as np
from fastapi import APIRouter,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel,Field
from starlette.background import BackgroundTask

from .workbench import presets,preview,PreviewRequest,read_product
from .workbench_archive import runtime,PROJECT,DEMO_PATHS
from .unified.jobs import manager
from .unified.geometry import build_layout
from .unified.trajectory import TrajectoryEvaluator
from .unified.radiation import directions,element_voltage,aperture_voltage
from .unified.storage import write_json

router=APIRouter(prefix='/api/v1/workbench',tags=['experiments'])


def job_or_404(job_id):
    try:return manager.get(job_id)
    except KeyError as exc:raise HTTPException(404,'实验不存在或已不可访问') from exc


def signature(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


@router.get('/capabilities')
def capabilities():
    from .unified.schema import Simulation
    return {'schema':Simulation.model_json_schema(),'modules':['platform','waveform','array','terrain','objects','visibility','radiation','receiver_noise','range_background'],
        'products':['rd','ra','angle'],'coordinate_frame':'NED_m','display_frame':'East,Up,-North',
        'limits':['静态扫频请使用其独立产品接口；本工作台编辑动态 LFM/FMCW 场景。','方向图按单位驱动归一化，不是绝对增益标定。']}


def case_specs():
    return [
        dict(id='points',title='点目标与距离多普勒',category='01 · 基础',preset='points_pulsed_lfm',light=True,description='调整目标位置与速度，观察距离与表观接近速度。',steps=['检查平台与目标参数','运行后查看 RD 与坐标读数','选中目标并展开剖面'],boundary='几何点目标与相对幅度；不代表语义目标散射。'),
        dict(id='terrain',title='地形与杂波结构',category='02 · 地形',preset='mixed_pulsed_lfm',light=True,description='查看程序地形、运动平台与混合对象共同形成的回波。',steps=['切换高程与线框','运行并播放已完成帧','复制到 A/B 比较地形开关'],boundary='程序地形；随机散射相位。'),
        dict(id='occlusion',title='照射、遮挡与有效回波',category='03 · 可见性',config='high_altitude_ridge_visibility.json',light=False,description='查看后端累计照射、无遮挡与有效回波次数。',steps=['检查遮挡预算','运行后切换覆盖图层','在实验对比中比较关闭遮挡的配置'],boundary='不透明三角面，按 Tx 时隙计算；不模拟衍射与穿透。'),
        dict(id='array',title='阵列与角度',category='04 · 阵列',config='lab_fmcw_mimo.json',light=False,description='观察 Tx/Rx 几何、距离方位与 TDM 补偿。',steps=['展开局部阵列','运行并查看距离方位','选择 RD 单元查看角度切片'],boundary='理想阵列，无互耦或绝对增益标定。'),
        dict(id='scan',title='电子波束扫描',category='05 · 扫描',config='phased_scan_10ghz.json',light=False,description='按仿真时钟查看发射方向图与扫描时序。',steps=['检查扫描表及 CPI 内切换风险','运行并切换帧','对照方向图和回波'],boundary='子孔径远场窄带近似；归一化图形不表示探测距离。'),
        dict(id='dem',title='真实 DEM 与地物分类',category='06 · 实际地形',archive_id='demo-land-cover',light=False,available=(DEMO_PATHS['demo-land-cover']/'run.json').exists(),description='浏览已有真实地形实验及显式散射参数，支持重新打开配置。',steps=['打开已保存结果','切换高程与地物分类','查看来源和散射参数假设'],boundary='真实 DSM/分类不是实测 RD；散射系数是敏感性实验输入。')]


@router.get('/cases')
def cases():
    specs=case_specs()
    # Exact effective-config matches only; a similarly named edited run is not an original preset result.
    candidates=[]
    for summary in list_experiments():
        if summary['status']=='complete':
            job=manager.get(summary['id'])
            candidates.append((summary,signature(job.request.model_dump(mode='json'))))
    for spec in specs:
        if 'archive_id' in spec:continue
        expected=signature(case_request(spec['id']))
        match=next((item for item,digest in candidates if digest==expected),None)
        if match:spec.update(archive_id=match['id'],available=True)
    return specs


@router.get('/cases/{case_id}/request')
def case_request(case_id:str):
    case=next((c for c in case_specs() if c['id']==case_id),None)
    if case is None:raise HTTPException(404,'案例不存在')
    if 'preset' in case:
        # This is a documented acquisition proposal, shown before submission.
        return preview(PreviewRequest(request=presets()[case['preset']],auto_acquisition=True))['request']
    if 'config' in case:
        from .unified.schema import Simulation
        return Simulation.model_validate_json((PROJECT/'configs/unified'/case['config']).read_text()).model_dump(mode='json')
    return job_or_404(case['archive_id']).request.model_dump(mode='json')


class SaveExperiment(BaseModel):
    title:str=Field(min_length=1,max_length=160)
    display:dict=Field(default_factory=dict)


@router.get('/experiments')
def list_experiments():
    ids=set(manager.jobs)|set(DEMO_PATHS)
    base=runtime()/'scene_jobs'
    if base.exists():ids.update(p.parent.name for p in base.glob('*/job.json'))
    result=[]
    for job_id in ids:
        try:
            job=manager.get(job_id)
            from .unified.schema import Simulation
            if not isinstance(job.request,Simulation):continue
            label_path=runtime()/'experiments'/(job_id+'.json')
            saved=json.loads(label_path.read_text()) if label_path.is_file() else {}
            result.append({'id':job_id,'name':saved.get('title',job.request.name),'status':job.status,
                'frames':len(job.frames),'frame_count':job.frame_count,'created_at':job.created_at,
                'saved':bool(saved),'archived':job_id in DEMO_PATHS,'error':job.error})
        except (KeyError,ValueError,OSError):continue
    return sorted(result,key=lambda x:x['created_at'],reverse=True)


@router.post('/experiments/{job_id}/save')
def save_experiment(job_id:str,body:SaveExperiment):
    job=job_or_404(job_id)
    record={'id':job_id,'title':body.title,'display':body.display,'config_hash':signature(job.request.model_dump(mode='json')),
        'saved_at':datetime.now(timezone.utc).isoformat(),'status':job.status}
    if len(json.dumps(record))>100_000:raise HTTPException(422,'显示设置过大')
    root=runtime()/'experiments';root.mkdir(parents=True,exist_ok=True)
    write_json(root/(job_id+'.json'),record)
    return record


@router.get('/experiments/{job_id}')
def experiment(job_id:str):
    job=job_or_404(job_id);path=runtime()/'experiments'/(job_id+'.json')
    return {'job':job.public(),'request':job.request.model_dump(mode='json'),
        'config_hash':signature(job.request.model_dump(mode='json')),'saved':json.loads(path.read_text()) if path.exists() else None}


@router.get('/jobs/{job_id}/geometry')
def geometry(job_id:str):
    job=job_or_404(job_id)
    if not (job.root/'scatterers.npz').exists():raise HTTPException(404,'几何尚未保存')
    with np.load(job.root/'scatterers.npz',allow_pickle=False) as scatter:
        stride=max(1,math.ceil(len(scatter['position_ned_m'])/4000));idx=np.arange(0,len(scatter['position_ned_m']),stride)
        result={'positions_ned_m':scatter['position_ned_m'][idx].tolist(),'velocities_ned_mps':scatter['velocity_ned_mps'][idx].tolist(),
            'object_index':scatter['object_index'][idx].tolist(),'preview_stride':stride,'scatterer_count':len(scatter['position_ned_m'])}
        if 'land_cover_labels' in scatter and len(scatter['land_cover_labels'])==len(scatter['position_ned_m']):result['land_cover_labels']=scatter['land_cover_labels'][idx].tolist()
    scene=json.loads((job.root/'scene.json').read_text());result['objects']=scene['objects']
    result['terrain']=json.loads((job.root/'terrain.json').read_text()) if (job.root/'terrain.json').exists() else None
    return result


@router.get('/jobs/{job_id}/frames/{index}/auxiliary')
def auxiliary(job_id:str,index:int):
    job=job_or_404(job_id)
    if not 0<=index<len(job.frames):raise HTTPException(404,'帧尚未完成')
    req=job.request;meta=job.frames[index];t=meta['frame_start_s'];layout=build_layout(req.array,req.waveform,req.platform)
    if req.platform.trajectory:
        pos,vel,basis,_=TrajectoryEvaluator(req.platform.trajectory).at(t)
    else:
        pos=np.asarray(req.platform.position_ned_m)+t*np.asarray(req.platform.velocity_ned_mps)
        vel=np.asarray(req.platform.velocity_ned_mps);basis=layout.basis_ned
    coverage=deepcopy(meta.get('visibility',{}).get('coverage'))
    if coverage:
        n=len(coverage['object_index']);stride=max(1,math.ceil(n/4000))
        for key in ['positions_at_frame_start_ned_m','object_index','illuminated_tx_slots','unblocked_channel_slots','sampled_echo_channel_slots']:coverage[key]=coverage[key][::stride]
        coverage.update(preview_stride=stride,total_scatterers=n)
    pattern=None
    if req.radiation:
        az,el=np.meshgrid(np.linspace(-180,180,73),np.linspace(-90,90,37))
        unit=directions(az,el);gain=element_voltage(unit[...,0],req.radiation.tx_element).astype(complex)
        aperture=req.radiation.transmit_aperture
        if aperture:gain*=aperture_voltage(unit,np.full(az.shape,t),aperture,req.waveform.carrier_hz)
        power=abs(gain)**2
        if not req.radiation.replace_legacy_cone:power*=unit[...,0]>=np.cos(np.deg2rad(req.platform.beam_half_angle_deg))
        maximum=float(power.max());power=power/max(maximum,1e-30)
        pattern={'directions_ned':(unit@basis.T).reshape(-1,3).tolist(),'normalized_power':power.ravel().tolist(),'rows':37,'cols':73,'time_s':t,'peak_power_before_normalization':maximum,'scope':'Tx normalized power at frame start; display length is not detection range'}
    return {'time_s':t,'platform_position_ned_m':pos.tolist(),'platform_velocity_ned_mps':vel.tolist(),'basis_ned':basis.tolist(),
        'coverage':coverage,'pattern':pattern,'array':layout.metadata(),'checks':job.estimate.get('applicability'),
        'objects':[{'id':o.id,'position_ned_m':(np.asarray(o.position_ned_m)+t*np.asarray(o.velocity_ned_mps)).tolist()} for o in req.scene.objects]}


@router.get('/experiments/{job_id}/export')
def export_experiment(job_id:str):
    job=job_or_404(job_id)
    # Freeze the completed-frame manifest: never bundle a concurrently written frame.
    frames=deepcopy(job.frames);paths=[job.root/n for n in ['request.json','scene.json','terrain.json','scatterers.npz','applicability.json','run.json','source.zip','source_manifest.json'] if (job.root/n).is_file()]
    for i in range(len(frames)):
        directory=job.root/f'frame_{i:04d}'
        paths.extend(directory/n for n in ['metadata.json','coverage.json'] if (directory/n).is_file())
        paths.extend(directory/v['file'] for v in frames[i]['products'].values())
    root=runtime()/'exports';root.mkdir(parents=True,exist_ok=True)
    fd=tempfile.NamedTemporaryFile(suffix='.zip',dir=root,delete=False);path=Path(fd.name);fd.close()
    try:
        with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as bundle:
            for p in paths:
                if p.is_symlink() or not p.resolve().is_relative_to(job.root.resolve()):raise ValueError('invalid product path')
                bundle.write(p,p.relative_to(job.root).as_posix())
            bundle.writestr('experiment.json',json.dumps({'job':{**job.public(),'frames':frames},'config_hash':signature(job.request.model_dump(mode='json')),'partial':len(frames)!=job.frame_count},ensure_ascii=False))
            # Include the current source for reproducibility only for new runs; historical source.zip is retained as-is.
            if not (job.root/'source.zip').exists():bundle.writestr('source_note.txt','Use request.json with scripts/run_scene.py. Solver source provenance was not recorded for this run; current code is not proof of historical source.\n')
    except Exception:
        path.unlink(missing_ok=True);raise
    return FileResponse(path,filename=f'experiment-{job_id}.zip',background=BackgroundTask(path.unlink,missing_ok=True))


@router.get('/jobs/{job_id}/frames/{index}/display-data')
def display_data(job_id:str,index:int,channel:str='combined'):
    from .workbench import parse_channel
    job=job_or_404(job_id)
    try:
        rd=read_product(job,index,'rd_channels');selected=parse_channel(channel,rd.shape[2],rd.shape[3])
        power=np.asarray(read_product(job,index,'rd_power'),float) if selected is None else abs(rd[:,:,selected[0],selected[1]].astype(np.complex128))**2
    except (ValueError,KeyError) as exc:raise HTTPException(422,str(exc)) from exc
    return {'power':power.T.tolist(),'axes':job.frames[index]['axes'],'frame_start_s':job.frames[index]['frame_start_s'],
        'channel':channel,'units':'relative linear power','processing':job.frames[index].get('processing'),
        'config_hash':signature(job.request.model_dump(mode='json'))}

class DraftBody(BaseModel):
    request:dict
    features:dict
    preset:str='custom'
    auto:bool=False
    result_job_id:str|None=None


@router.get('/draft')
def read_draft():
    path=runtime()/'workbench_draft.json'
    return json.loads(path.read_text()) if path.exists() else None


@router.put('/draft')
def save_draft(body:DraftBody):
    data=body.model_dump()
    if len(json.dumps(data))>30_000_000:raise HTTPException(422,'草稿超过 30 MB')
    root=runtime();root.mkdir(parents=True,exist_ok=True);write_json(root/'workbench_draft.json',data)
    return {'saved':True}


@router.post('/normalize')
def normalize(request:dict):
    from .unified.schema import Simulation
    try:return Simulation.model_validate(request).model_dump(mode='json')
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc


@router.get('/jobs/{job_id}/series')
def time_series(job_id:str):
    job=job_or_404(job_id);r=job.request
    trajectory=TrajectoryEvaluator(r.platform.trajectory) if r.platform.trajectory else None
    def pose(t):
        if trajectory:
            p,v,_,_=trajectory.at(t);return p,v
        return np.asarray(r.platform.position_ned_m)+t*np.asarray(r.platform.velocity_ned_mps),np.asarray(r.platform.velocity_ned_mps)
    platform=[];objects={o.id:[] for o in r.scene.objects}
    for f in job.frames:
        t=f['frame_start_s'];p,v=pose(t);platform.append({'time_s':t,'height_m':float(-p[2]),'speed_mps':float(np.linalg.norm(v))})
        for o in r.scene.objects:
            delta=np.asarray(o.position_ned_m)+t*np.asarray(o.velocity_ned_mps)-p;distance=float(np.linalg.norm(delta))
            objects[o.id].append({'time_s':t,'range_m':distance,'closing_mps':float(np.dot(v-np.asarray(o.velocity_ned_mps),delta)/distance) if distance>0 else None})
    return {'platform':platform,'objects':objects,'scope':'configured object reference position and phase-center radial speed at frame starts; not detected peaks'}
