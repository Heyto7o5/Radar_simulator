#!/usr/bin/env python3
"""Extract a geometry-only candidate surface; never fit radar observations."""
import json
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radar_platform.lidar_surface import planar_patches

ROOT=Path('/nas/datasets/hyt/datasets/radar_open/dart_lab')
OUT=Path(__file__).resolve().parents[1]/'docs/validation_diagnostics/2026-10-15/dart'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    path=ROOT/'extracted/lab-1'
    with h5py.File(path/'trajectory.h5') as f:
        index=int(np.flatnonzero(f['valid'][:])[0]);pos=f['pos'][index];rot=f['rot'][index]
    with np.load(path/'map.npz',allow_pickle=False) as z:
        grid=z['grid'];lower=z['lower'];upper=z['upper']
        cell=.02 # explicit assumption from official map.py default, checked against bounds/shape
        if not np.array_equal(np.ceil((upper-lower)/cell).astype(int),grid.shape):
            raise ValueError('official default resolution inconsistent with map dimensions')
        points=lower+(np.argwhere(grid)+.5)*cell
    local=(points-pos)@rot
    radius=np.linalg.norm(local,axis=1)
    keep=(radius>=.3)&(radius<=5.4)&(local[:,0]>0)
    keep &= np.abs(np.arctan2(local[:,1],local[:,0]))<=np.deg2rad(50)
    keep &= np.abs(np.arctan2(local[:,2],np.hypot(local[:,0],local[:,1])))<=np.deg2rad(40)
    selected=points[keep]
    surface=planar_patches(selected,spacing_m=.15,neighborhood_m=.22,observer_m=pos)
    vertices,faces=surface.triangle_mesh()
    np.savez_compressed(ROOT/'processed/lab-1_surface_candidate.npz',vertices_flu_m=vertices,faces=faces,
                        centers_flu_m=surface.centers_m,normals_flu=surface.normals,
                        source_trajectory_index=index,observer_flu_m=pos)
    summary={'source':'https://zenodo.org/records/10938617','source_map':'lab-1/map.npz',
             'radar_observations_used_for_geometry':False,'trajectory_index':index,
             'occupied_input_cells':int(len(points)),'geometry_crop_cells':len(selected),
             'candidate_patches':surface.candidate_count,'accepted_patches':len(surface.centers_m),
             'accepted_candidate_fraction':len(surface.centers_m)/surface.candidate_count,
             'geometry_sufficient_for_full_scene_validation':False,
             'rejected_nonplanar_or_sparse':surface.rejected_count,'patch_width_m':.15,
             'normal_neighborhood_m':.22,'max_normal_variance_ratio':.03,
             'assumed_source_cell_m':cell,'cell_convention':'cell center under official map.py default; not survey-verified',
             'geometry_crop':'0.3 to 5.4 m; +/-50 deg azimuth, +/-40 deg elevation relative to first valid pose',
             'crop_is_measured_antenna_pattern':False,
             'normal_orientation':'toward selected observer, not known closed-object exterior',
             'surface_area_is_calibrated':False,'occlusion_implemented':False,
             'material_parameters_available':False,'radar_fidelity_verified':False}
    (OUT/'surface_input_audit.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,5),constrained_layout=True)
    stride=max(1,len(selected)//50000)
    axes[0].scatter(selected[::stride,0],selected[::stride,1],s=.1,c=selected[::stride,2],cmap='viridis')
    axes[0].set_title('Independent LiDAR occupied-cell centers')
    q=surface.centers_m
    axes[1].scatter(q[:,0],q[:,1],c=q[:,2],s=4,cmap='viridis')
    axes[1].set_title(f'Geometry-only planar patches: {len(q)} accepted')
    for ax in axes:
        ax.scatter(*pos[:2],marker='x',color='red',s=50)
        ax.set(xlabel='World front x (m)',ylabel='World left y (m)',aspect='equal')
    fig.savefig(OUT/'dart_geometry_input.png',dpi=160);plt.close(fig)
    (OUT/'DART几何输入.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>DART 几何输入</title>
<style>body{max-width:1100px;margin:30px auto;font:16px/1.7 sans-serif}img{width:100%}td,th{padding:10px;border:1px solid #bbb}table{border-collapse:collapse}</style>
<h1>从公开 LiDAR 几何到面片输入</h1><p>只使用独立 LiDAR 地图与雷达姿态，未使用实测 RD 调整几何。红叉为雷达位置；颜色表示高度。</p>
<p style="background:#fff0cb;padding:15px">当前几何检查未通过：8,117 个候选区域仅保留 116 个（约 1.4%）。现有尺度的局部平面法覆盖不足；不能用这批面片代表完整场景，也不能将其缺失回波归咎于散射模型。</p>
<img src="dart_geometry_input.png"><table><tr><th>已做到</th><th>不能当作真值的部分</th></tr>
<tr><td>按固定距离和视角范围选择实际地图；用邻域平面拟合估计面片法向；退化或不平区域计数保留。</td><td>15 cm 面片尺寸、面积和单面朝向均为建模选择，不是原始材料或封闭表面真值。</td></tr>
<tr><td>面片可转换为三角网格，作为现有面回波模型的几何输入。</td><td>还缺遮挡、材料系数及采集处理对齐；当前不是仿真实测对比图。</td></tr></table>
<p>本图采用官方地图生成代码默认 2 cm 单元尺寸，已核对其与文件边界及网格形状一致；采用单元中心代表激光点位置，存在单元内位置不确定性。视角裁剪不是测得的天线方向图。</p>
<p><a href="https://github.com/thetianshuhuang/rover/blob/main/processing/scripts/map.py">地图生成方法</a> · <a href="surface_input_audit.json">完整输入与筛选统计</a></p></html>''')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
