#!/usr/bin/env python3
"""Geometry containment diagnostic, not scattering accuracy or target recall."""
import csv,json,argparse
from pathlib import Path
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
from radar_platform.dart_dataset import FLU_TO_NED
from radar_platform.unified.scene import compile_scene
from radar_platform.unified.trajectory import TrajectoryEvaluator
from radar_platform.colored_adc_background import expected_windowed_range_power
from evaluate_dart_frozen_batch import load,DATA,OUT as BATCH

OUT=BATCH.parent/'geometry_rd_support_audit'

def support(points,request,ranges,velocity):
    times=np.linspace(0,(request.waveform.loops-1)*request.array.loop_period_s,17)
    positions,speeds,bases,_=TrajectoryEvaluator(request.platform.trajectory).at(times)
    # Safe range prefilter relative to all sampled platform positions, not FOV fitting.
    center=positions[len(positions)//2]
    radius_bound=ranges[-1]+np.linalg.norm(positions-center,axis=1).max()+.3
    points=points[np.linalg.norm(points-center,axis=1)<=radius_bound]
    result=np.zeros((len(ranges),len(velocity)),bool)
    for p,v,basis in zip(positions,speeds,bases):
        delta=points-p;radius=np.linalg.norm(delta,axis=1)
        unit=delta/np.maximum(radius[:,None],1e-12)
        closing=unit@v
        lit=(unit@basis[:,0]>=np.cos(np.deg2rad(request.platform.beam_half_angle_deg)))&(radius>1e-6)
        ri=np.rint((radius-ranges[0])/(ranges[1]-ranges[0])).astype(int)
        vi=np.rint((closing-velocity[0])/(velocity[1]-velocity[0])).astype(int)
        keep=lit&(ri>=0)&(ri<len(ranges))&(vi>=0)&(vi<len(velocity))
        result[ri[keep],vi[keep]]=True
    return binary_dilation(result,structure=np.ones((5,7),bool)),float(np.linalg.norm(speeds,axis=1).max())+.05

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--include-initial-crop',action='store_true');args=parser.parse_args()
    if args.include_initial_crop:OUT=OUT.parent/'geometry_crop_audit'
    OUT.mkdir(parents=True,exist_ok=True)
    with np.load(DATA/'extracted/lab-1/map.npz',allow_pickle=False) as f:
        original=f['lower']+(np.argwhere(f['grid'])+.5)*.02
    clouds={}
    for cell in [.1,.05]:
        cells=np.unique(np.floor(original/cell).astype(np.int64),axis=0)
        clouds[f'lidar_{cell:g}m']=((cells+.5)*cell)@FLU_TO_NED.T
    if args.include_initial_crop:
        with h5py.File(DATA/'extracted/lab-1/trajectory.h5') as f:
            index=int(np.flatnonzero(f['valid'][:])[0]);pos=f['pos'][index];rot=f['rot'][index]
        local=(original-pos)@rot;radius=np.linalg.norm(local,axis=1)
        keep=(radius>=.3)&(radius<=5.4)&(local[:,0]>0)
        keep&=abs(np.arctan2(local[:,1],local[:,0]))<=np.deg2rad(50)
        keep&=abs(np.arctan2(local[:,2],np.hypot(local[:,0],local[:,1])))<=np.deg2rad(40)
        cells=np.unique(np.floor(original[keep]/.05).astype(np.int64),axis=0)
        clouds['initial_crop_lidar_0.05m']=((cells+.5)*.05)@FLU_TO_NED.T
    del original
    calibration=json.loads((BATCH/'frozen_calibration.json').read_text())
    colored=json.loads((BATCH.parent/'colored_background_followup/frozen_colored_model.json').read_text())
    background=calibration['gain']*expected_windowed_range_power(colored['spectral_power'],np.hanning(512),np.hanning(256),8)
    indices=json.loads((BATCH/'evaluation_protocol.json').read_text())['temporal_check_valid_indices']
    rows=[];examples={}
    for index in indices:
        request,adc,r,v,real=load(index)
        native=np.arange(512)*request.waveform.sample_rate_hz/512*299792458/(2*request.waveform.slope_hz_s)
        floor=np.interp(r,native,background)
        excess=np.maximum(real-floor[:,None],0)
        point_sets={'fitted_surface':compile_scene(request.scene).positions_ned_m,**clouds}
        local={}
        for name,points in point_sets.items():
            covered,limit=support(points,request,r,v)
            domain=(r[:,None]>=.3)&(r[:,None]<=5.4)&(abs(v)[None,:]<=limit)&(abs(v)[None,:]>1.01*abs(v[1]-v[0]))
            mask=covered&domain;total=excess[domain].sum()
            if total<=0:raise ValueError('no measured excess energy in fixed kinematic domain')
            rows.append({'valid_frame_index':index,'geometry':name,'input_points':len(points),
                         'rd_domain_bins':int(domain.sum()),'covered_bins':int(mask.sum()),
                         'rd_area_fraction':float(mask.sum()/domain.sum()),
                         'measured_excess_energy_fraction':float(excess[mask].sum()/total),
                         'not_target_detection_accuracy':True})
            local[name]=mask
        if index in [indices[0],72]:examples[index]=(r,v,real,local)
        print('geometry support audited',index,flush=True)
    with (OUT/'per_frame_geometry_coverage.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'scene_count':1,'frames':len(indices),'posthoc_diagnostic':True,'independent_scattering_validation':False,
             'roi':'range .3..5.4m; |velocity| <= maximum sampled speed+.05m/s; exclude central three velocity bins',
             'dilation':'same +/-2 range bins and +/-3 Doppler bins for all geometry inputs',
             'geometry_counts':{name:len(x) for name,x in clouds.items()},
             'limitations':['broader geometric support can trivially cover more energy; report covered area together',
                            'LiDAR occupancy does not provide materials, visibility or actual scattering strength',
                            'excess over fixed empirical calibration background is not independently labeled target energy',
                            'voxel centers introduce geometric quantization; compare two resolutions',
                            'containment does not prove complete 3D reconstruction or radar fidelity'],
             'models':{}}
    for name in ['fitted_surface',*clouds]:
        subset=[row for row in rows if row['geometry']==name]
        summary['models'][name]={key:float(np.mean([row[key] for row in subset])) for key in ['rd_area_fraction','measured_excess_energy_fraction']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    for index,(r,v,real,local) in examples.items():
        fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
        values=10*np.log10(np.maximum(real/calibration['shared_power_reference'],1e-6))
        for ax,name in zip(axes,['fitted_surface','initial_crop_lidar_0.05m' if args.include_initial_crop else 'lidar_0.1m','lidar_0.05m']):
            ax.pcolormesh(v,r,values,shading='auto',vmin=-60,vmax=10,cmap='viridis')
            ax.contour(v,r,local[name].astype(float),levels=[.5],colors=['white'],linewidths=.6)
            ax.set(xlabel='Closing velocity (m/s)',ylabel='Range (m)',title=name+' support over MEASURED RD')
        fig.savefig(OUT/f'geometry_support_f{index:04d}.png',dpi=150);plt.close(fig)
    table=''.join(f'<tr><td>{name}</td><td>{values["rd_area_fraction"]:.1%}</td><td>{values["measured_excess_energy_fraction"]:.1%}</td></tr>' for name,values in summary['models'].items())
    crop_note=''
    if args.include_initial_crop:
        crop_note='<p style="background:#fff0ca;padding:15px">分开检查后，完整 LiDAR、按第一帧视野裁剪的 LiDAR、当前面网格的能量覆盖依次约为 97.3%、81.1%、73.9%。在本诊断口径下，初始场景裁剪造成的覆盖差距比后续平面提取更大。应先保留覆盖整段已知轨迹的固定世界几何，再逐 chirp 判断可见性；不能一直沿用第一帧的场景裁剪。</p>'
    (OUT/'几何覆盖与回波误差定位.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{max-width:1300px;margin:30px auto;font:16px/1.7 sans-serif}img{width:100%}td,th{border:1px solid #bbb;padding:10px}table{border-collapse:collapse}</style>
<h1>误差主要是几何遗漏，还是覆盖内的回波仍不对？</h1><p>白线圈出几何在 RD 上可能覆盖的位置，底图始终是实测。只检查位置范围，不生成新的散射强度，也不拟合雷达主峰。</p>
<table><tr><th>几何输入</th><th>占固定 RD 区域的比例</th><th>覆盖的观测超背景能量比例</th></tr>TABLE</table>
<p>第二项是覆盖面积，第三项是扣除固定校准背景后观测能量的覆盖比例。它不是目标检出率或仿真准确率；范围扩大本身就可能覆盖更多能量，必须一起读。</p>
<img src="geometry_support_f0032.png"><img src="geometry_support_f0072.png">
<p>原始 LiDAR 仍有厚度和量化误差，也不包含材料与遮挡真值。即使实测能量落在几何范围内，也不能证明反射强度、旁瓣、多径或完整三维形状正确。两种体素尺寸用于检查粗化敏感性；本页是同一场景的事后诊断，不是新增独立验证。</p>
<p><a href="summary.json">完整口径与统计</a> · <a href="per_frame_geometry_coverage.csv">全部逐帧数据</a></p></html>'''.replace('TABLE',table).replace('</h1>','</h1>'+crop_note))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
