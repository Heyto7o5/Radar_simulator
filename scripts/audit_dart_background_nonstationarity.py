#!/usr/bin/env python3
"""Measured-only audit over all valid frames; no new fitting to test targets."""
import csv,json
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/nas/datasets/hyt/datasets/radar_open/dart_lab')
OUT=Path(__file__).resolve().parents[1]/'docs/validation_diagnostics/2026-10-15/dart/background_audit'
BATCH=OUT.parent/'batch_v1'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    folder=ROOT/'extracted/lab-1'
    sensor=json.loads((folder/'sensor.json').read_text());velocity=np.linspace(*sensor['d']);ranges=np.linspace(*sensor['r'])
    cal=json.loads((BATCH/'frozen_calibration.json').read_text());reference=cal['measured_floor_proxy']
    with h5py.File(folder/'trajectory.h5') as f:
        valid=np.flatnonzero(f['valid'][:]);raw=np.flatnonzero(f['mask'][:]);time=f['t'][:];speed=np.linalg.norm(f['vel'][:],axis=1)
    profiles=np.full((len(valid),len(ranges)),np.nan);rows=[]
    with h5py.File(folder/'radar.h5') as f:
        for j,index in enumerate(valid):
            inside=(time>=time[index]-.128)&(time<=time[index]+.128)
            limit=float(speed[inside].max())+.05
            mask=abs(velocity)>limit
            # At least four velocity bins: no forced estimate when support absent.
            row={'valid_frame_index':j,'trajectory_index':int(index),'raw_frame_index':int(raw[index]),
                 'elapsed_s':float(time[index]-time[valid[0]]),'speed_bound_mps':limit,
                 'off_kinematic_bins':int(mask.sum()),'floor_available':bool(mask.sum()>=4),
                 'median_power':None,'median_vs_fixed_calibration_db':None,'range_p90_p10_span_db':None}
            if mask.sum()>=4:
                # Reader checks magnitude representation; keep absolute published scale.
                magnitude=f['rad'][raw[index]].astype(np.float64)
                if not np.isfinite(magnitude).all() or np.any(magnitude<0):raise ValueError('invalid measured magnitude')
                power=np.mean(magnitude**2,axis=2)
                background=power[:,mask];profile=np.median(background,axis=1)
                profiles[j]=profile
                positive=profile[profile>0]
                if len(positive)<len(profile):row['zero_profile_bins']=int((profile<=0).sum())
                median=float(np.median(background));row['median_power']=median
                if median>0:row['median_vs_fixed_calibration_db']=float(10*np.log10(median/reference))
                if len(positive):row['range_p90_p10_span_db']=float(10*np.log10(np.quantile(positive,.9)/np.quantile(positive,.1)))
            rows.append(row)
            if (j+1)%500==0:print('audited',j+1,'of',len(valid),flush=True)
    keys=sorted(set().union(*(row.keys() for row in rows)))
    with (OUT/'all_frame_background.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    np.savez_compressed(ROOT/'processed/all_valid_background_profiles.npz',range_m=ranges,profiles=profiles,
                        valid_indices=np.arange(len(valid)),trajectory_indices=valid,raw_indices=raw[valid],timestamp_s=time[valid])
    finite=np.isfinite(profiles).all(axis=1)
    shifts=[row['median_vs_fixed_calibration_db'] for row in rows if row['median_vs_fixed_calibration_db'] is not None]
    spans=[row['range_p90_p10_span_db'] for row in rows if row['range_p90_p10_span_db'] is not None]
    summary={'scene_count':1,'valid_frames':len(valid),'frames_with_floor_support':int(finite.sum()),
             'frames_without_sufficient_off_kinematic_support':int((~finite).sum()),
             'fixed_calibration_floor':reference,'static_scene_assumption':True,
             'floor_is_noise_truth':False,'method':'median power across angle-averaged bins with |published velocity| > maximum sampled platform speed within CPI + .05m/s; at least 4 Doppler bins',
             'limitations':['out-of-kinematic energy may be sidelobes, phase discontinuity, interference or moving objects, not purely thermal noise',
                            'published magnitude data cannot recover missing ADC phase','overlapping frames are not independent samples',
                            'all-frame diagnostic is not used as a calibration fit or held-out performance score'],
             'frame_median_shift_db':{'p10':float(np.quantile(shifts,.1)),'median':float(np.median(shifts)),'p90':float(np.quantile(shifts,.9))},
             'within_frame_range_variation_db':{'median_p90_p10_span':float(np.median(spans)),'p90_span':float(np.quantile(spans,.9))}}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    elapsed=time[valid]-time[valid[0]]
    db=10*np.log10(np.maximum(profiles/reference,1e-12))
    image=axes[0].pcolormesh(elapsed,ranges,db.T,shading='auto',cmap='RdBu_r',vmin=-15,vmax=15)
    axes[0].set(xlabel='Time since first valid pose (s)',ylabel='Range (m)',title='MEASURED off-kinematic power / frozen calibration floor')
    fig.colorbar(image,ax=axes[0],label='dB (display clipped +/-15); blank = insufficient support')
    axes[1].plot(elapsed,[row['median_vs_fixed_calibration_db'] if row['median_vs_fixed_calibration_db'] is not None else np.nan for row in rows],lw=.7)
    axes[1].axhline(0,c='black',ls='--');axes[1].set(xlabel='Time (s)',ylabel='Frame median shift (dB)',title='A fixed white-noise floor predicts the horizontal zero line')
    fig.savefig(OUT/'measured_background_variation.png',dpi=150);plt.close(fig)
    (OUT/'背景随距离与时间变化.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{max-width:1200px;margin:30px auto;font:16px/1.7 sans-serif}img{width:100%}</style>
<h1>实测背景是否能用一个固定白噪声强度解释？</h1><p>本页检查整段有效轨迹，不调整仿真参数。上图按时间和距离展示运动范围外的实测功率；下图展示每帧的中位数，相对之前三个校准帧得到的固定底噪。</p>
<img src="measured_background_variation.png"><p>红色比固定底噪强，蓝色比固定底噪弱，空白表示没有足够的运动范围外速度格，不能估计。这里的背景代理可能包含旁瓣、干扰、相位不连续或未知运动物体，不能直接解释为热噪声或地物散射真值。</p>
<p>同一场景的连续帧并非独立场景；本页是事后诊断，不作为留出验证，也不用于反过来重标定上一批测试结果。</p><p><a href="summary.json">完整统计</a> · <a href="all_frame_background.csv">所有有效帧与不可估计记录</a></p></html>''')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
