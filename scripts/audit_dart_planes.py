#!/usr/bin/env python3
"""Fixed multi-plane baseline with spatially separate geometry holdout cells."""
import json
import argparse
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radar_platform.plane_extraction import extract_planes,nearest_plane_distance,supported_plane_mesh

ROOT=Path('/nas/datasets/hyt/datasets/radar_open/dart_lab')
OUT=Path(__file__).resolve().parents[1]/'docs/validation_diagnostics/2026-10-15/dart'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--cell-m',type=float,default=.15);parser.add_argument('--max-planes',type=int,default=8)
    parser.add_argument('--crop-mode',choices=['initial','trajectory_bounds'],default='initial');args=parser.parse_args()
    folder=ROOT/'extracted/lab-1'
    with h5py.File(folder/'trajectory.h5') as f:
        index=int(np.flatnonzero(f['valid'][:])[0]);pos=f['pos'][index];rot=f['rot'][index]
    with np.load(folder/'map.npz',allow_pickle=False) as f:
        points=f['lower']+(np.argwhere(f['grid'])+.5)*.02
    local=(points-pos)@rot;r=np.linalg.norm(local,axis=1)
    keep=(r>=.3)&(r<=5.4)&(local[:,0]>0)
    keep&=abs(np.arctan2(local[:,1],local[:,0]))<=np.deg2rad(50)
    keep&=abs(np.arctan2(local[:,2],np.hypot(local[:,0],local[:,1])))<=np.deg2rad(40)
    if args.crop_mode=='trajectory_bounds':
        manifest=json.loads((OUT/'batch_v1/manifest.json').read_text())
        selected=[r['trajectory_index'] for r in manifest['records']]
        with h5py.File(folder/'trajectory.h5') as f:
            pose_start=max(0,min(selected)-3);pose_stop=min(len(f['pos']),max(selected)+4)
            trajectory=f['pos'][pose_start:pose_stop]
        lower=trajectory.min(axis=0)-5.6;upper=trajectory.max(axis=0)+5.6
        keep=np.all((points>=lower)&(points<=upper),axis=1)
    points=points[keep]
    # Whole 15 cm cells, not pointwise random train/test allocation.
    cells=np.floor(points/.15).astype(np.int64)
    hold=(cells@np.array([73856093,19349663,83492791]))%5==0
    def bounded(x):return x[np.linspace(0,len(x)-1,min(30000,len(x)),dtype=int)]
    train,test=bounded(points[~hold]),bounded(points[hold])
    planes=extract_planes(train,tolerance_m=.04,max_planes=args.max_planes,trials=256,min_inliers=300,seed=42)
    distances=nearest_plane_distance(test,planes)
    vertices=[];faces=[];offset=0
    for plane in planes:
        v,f=supported_plane_mesh(plane,train,cell_m=args.cell_m,tolerance_m=.04)
        vertices.append(v);faces.append(f+offset);offset+=len(v)
    # A candidate geometry asset, not a calibrated scattering scene.
    asset=ROOT/'processed'/('lab-1_supported_planes.npz' if args.cell_m==.15 else f'lab-1_supported_planes_cell_{args.cell_m:g}.npz')
    if args.max_planes!=8:asset=asset.with_name(asset.stem+f'_maxplanes_{args.max_planes}'+asset.suffix)
    if args.crop_mode!='initial':asset=asset.with_name(asset.stem+'_'+args.crop_mode+asset.suffix)
    np.savez_compressed(asset,
                        vertices_flu_m=np.concatenate(vertices),faces=np.concatenate(faces))
    report={'radar_data_used':False,'source_scene_count':1,'fitting_points':len(train),'heldout_points':len(test),
            'crop_mode':args.crop_mode,'geometry_points_after_crop':len(points),
            'world_crop_bounds_flu_m':[lower.tolist(),upper.tolist()] if args.crop_mode=='trajectory_bounds' else None,
            'trajectory_pose_indices_for_bounds':[pose_start,pose_stop] if args.crop_mode=='trajectory_bounds' else None,
            'trajectory_bounds_uses_observed_rd':False,
            'holdout':'whole 15cm cells selected by fixed integer hash modulo 5; spatial interpolation check, not new-scene validation',
            'tolerance_m':.04,'seed':42,'max_planes':args.max_planes,'planes_found':len(planes),
            'heldout_fraction_within_4cm':float(np.mean(distances<=.04)),
            'heldout_median_distance_m':float(np.median(distances)),
            'heldout_p90_distance_m':float(np.quantile(distances,.9)),
            'distance_definition':'nearest finite enclosing plane rectangle; optimistic at holes within each rectangle',
            'rectangle_surface_area_verified':False,'occlusion_verified':False,'materials_verified':False,
            'supported_mesh_triangles':sum(len(f) for f in faces),
            'supported_mesh_cell_m':args.cell_m,'supported_mesh_uses_only_fit_points':True,
            'supported_mesh_area_calibrated':False,
            'ready_for_full_scene_fidelity_claim':False,
            'planes':[{'center':p.center.tolist(),'normal':p.normal.tolist(),'tangent_u':p.tangent_u.tolist(),
                       'tangent_v':p.tangent_v.tolist(),'bounds_uv':p.bounds_uv.tolist(),'inlier_count':p.inlier_count} for p in planes]}
    if args.cell_m!=.15 or args.max_planes!=8 or args.crop_mode!='initial':
        asset.with_suffix('.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='planes'}),flush=True)
        return
    (OUT/'plane_holdout_audit.json').write_text(json.dumps(report,indent=2))
    fig=plt.figure(figsize=(13,5),layout='constrained');ax=fig.add_subplot(121,projection='3d')
    s=test[::10];ax.scatter(s[:,0],s[:,1],s[:,2],s=.3,c='gray',alpha=.3)
    for p in planes:
        (u0,v0),(u1,v1)=p.bounds_uv
        uv=np.array([[u0,v0],[u1,v0],[u1,v1],[u0,v1],[u0,v0]])
        q=p.center+uv[:,0,None]*p.tangent_u+uv[:,1,None]*p.tangent_v
        ax.plot(q[:,0],q[:,1],q[:,2],lw=1)
    ax.set(xlabel='x front (m)',ylabel='y left (m)',zlabel='z up (m)',title='Fitted finite plane bounds + held-out geometry')
    ax=fig.add_subplot(122);ordered=np.sort(distances)
    ax.plot(ordered,np.arange(1,len(ordered)+1)/len(ordered));ax.axvline(.04,c='red',ls='--')
    ax.set(xlabel='Held-out distance to nearest finite plane (m)',ylabel='Fraction of held-out points',
           title='All held-out points retained, including uncovered regions',xlim=(0,max(.1,float(np.quantile(distances,.99)))))
    fig.savefig(OUT/'dart_plane_holdout.png',dpi=160);plt.close(fig)
    (OUT/'DART平面拟合复核.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{max-width:1100px;margin:30px auto;font:16px/1.7 sans-serif}img{width:100%}</style>
<h1>多平面提取：使用未参与拟合的几何点复核</h1><p>只使用 LiDAR 几何。以完整 15 cm 网格分开拟合区与检查区，固定种子、4 cm 容差和最多八个平面，不使用雷达图选择参数。</p>
<img src="dart_plane_holdout.png"><p>左图：彩色框是拟合平面的有限边界，灰点为未参与拟合的几何点。右图：所有检查点到最近平面的距离，越早接近 1 表示几何覆盖越好；红线为固定 4 cm 容差。</p>
<p><strong>限制：</strong>矩形框可能跨过孔洞，距离指标可能偏乐观；尚未验证实际表面积、材料或遮挡。这只是同一场景的几何插值检查，不是独立场景验证，更不是雷达保真通过。未被平面覆盖的点仍保留在误差统计中。</p><p><a href="plane_holdout_audit.json">完整参数、平面与指标</a></p></html>''')
    print(json.dumps({k:v for k,v in report.items() if k!='planes'}),flush=True)

if __name__=='__main__':main()
