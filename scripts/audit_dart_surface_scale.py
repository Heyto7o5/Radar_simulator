#!/usr/bin/env python3
"""Fixed geometry-only scale audit; no RD-based threshold optimization."""
import csv
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

ROOT=Path('/nas/datasets/hyt/datasets/radar_open/dart_lab/extracted/lab-1')
OUT=Path(__file__).resolve().parents[1]/'docs/validation_diagnostics/2026-10-15/dart'


def covariance_metrics(points):
    centered=points-points.mean(axis=0)
    values=np.linalg.eigvalsh(centered.T@centered/len(points))
    return float(values[0]/values.sum()),float(np.sqrt(max(values[0],0))),float(values[1]/values[2])


def main():
    with h5py.File(ROOT/'trajectory.h5') as f:
        i=int(np.flatnonzero(f['valid'][:])[0]);pos=f['pos'][i];rot=f['rot'][i]
    with np.load(ROOT/'map.npz',allow_pickle=False) as f:
        points=f['lower']+(np.argwhere(f['grid'])+.5)*.02
    local=(points-pos)@rot;r=np.linalg.norm(local,axis=1)
    mask=(r>=.3)&(r<=5.4)&(local[:,0]>0)
    mask&=abs(np.arctan2(local[:,1],local[:,0]))<=np.deg2rad(50)
    mask&=abs(np.arctan2(local[:,2],np.hypot(local[:,0],local[:,1])))<=np.deg2rad(40)
    points=points[mask]
    _,inverse=np.unique(np.floor(points/.15).astype(int),axis=0,return_inverse=True)
    counts=np.bincount(inverse)
    centers=np.column_stack([np.bincount(inverse,weights=points[:,j])/counts for j in range(3)])
    chosen=np.linspace(0,len(centers)-1,min(512,len(centers)),dtype=int)
    tree=cKDTree(points);rows=[]
    for radius in [.1,.22,.4]:
        for index in chosen:
            ids=tree.query_ball_point(centers[index],radius)
            if len(ids)<8:
                rows.append(dict(radius_m=radius,candidate_index=int(index),neighbors=len(ids),ratio=None,
                                 normal_rms_m=None,second_to_first=None,accepted=False));continue
            ratio,thickness,linearity=covariance_metrics(points[ids])
            rows.append(dict(radius_m=radius,candidate_index=int(index),neighbors=len(ids),ratio=ratio,
                             normal_rms_m=thickness,second_to_first=linearity,accepted=ratio<=.03))
    with (OUT/'surface_scale_samples.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for radius in [.1,.22,.4]:
        group=[x for x in rows if x['radius_m']==radius];valid=[x for x in group if x['ratio'] is not None]
        ratios=np.array([x['ratio'] for x in valid]);thickness=np.array([x['normal_rms_m'] for x in valid])
        entry={'radius_m':radius,'sampled_candidates':len(group),'valid_neighborhoods':len(valid),
               'accepted_fraction':sum(x['accepted'] for x in group)/len(group),
               'median_normal_variance_ratio':float(np.median(ratios)),
               'median_normal_rms_m':float(np.median(thickness))}
        summary.append(entry)
        axes[0].hist(ratios,bins=np.linspace(0,1/3,35),histtype='step',label=f'{radius} m')
        axes[1].hist(thickness*100,bins=np.linspace(0,20,41),histtype='step',label=f'{radius} m')
        axes[2].scatter(radius,entry['accepted_fraction']*100,label=f'{radius} m')
    axes[0].axvline(.03,c='red',ls='--',label='fixed planarity threshold')
    axes[0].set(xlabel='Normal variance / total variance',ylabel='Count',title='Local shape (0=plane, 1/3=isotropic)')
    axes[1].set(xlabel='Normal-direction RMS extent (cm)',ylabel='Count',title='Neighborhood thickness, NOT ranging error')
    axes[2].set(xlabel='Neighborhood radius (m)',ylabel='Accepted candidates (%)',title='Geometry-only scale sensitivity')
    for ax in axes:ax.legend(fontsize=8)
    fig.savefig(OUT/'dart_surface_scale.png',dpi=160);plt.close(fig)
    result={'selection':'512 evenly spaced sorted coarse-cell indices; fixed before seeing radar',
            'scene_count':1,'threshold':.03,'metrics':summary,'radar_used':False,
            'interpretation':'PCA thickness measures local occupied geometry, not sensor noise; corners, multiple surfaces and map thickness can all increase it',
            'no_threshold_selected_from_radar':True}
    (OUT/'surface_scale_audit.json').write_text(json.dumps(result,indent=2))
    (OUT/'DART表面尺度诊断.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{max-width:1200px;margin:30px auto;font:16px/1.7 sans-serif}img{width:100%}td,th{padding:9px;border:1px solid #bbb}table{border-collapse:collapse}</style><h1>为什么多数局部区域没有被判定为平面？</h1><p>同一份 LiDAR 几何、固定 512 个候选位置、固定判据，只改变邻域尺度。没有使用雷达图挑选尺度或阈值。</p><img src="dart_surface_scale.png"><table><tr><th>图</th><th>含义</th></tr><tr><td>左：法向方差占比</td><td>越接近 0，局部越像薄平面；越接近 1/3，三个方向都较厚，不能可靠地赋予唯一面法向。</td></tr><tr><td>中：法向厚度</td><td>邻域点沿最薄方向的均方根范围，不是雷达测距误差，也不能直接解释为 LiDAR 噪声。</td></tr><tr><td>右：通过比例</td><td>固定阈值下的尺度敏感性。通过更多不等于表面更真实，不能据此自动选一个最好看的尺度。</td></tr></table><p>下一步应区分真实交界面、叠加表面与地图厚度，再检查重建覆盖；不通过调高阈值将三维厚点簇硬当作平面。</p><p><a href="surface_scale_audit.json">数值汇总</a> · <a href="surface_scale_samples.csv">逐候选记录</a></p></html>''')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
