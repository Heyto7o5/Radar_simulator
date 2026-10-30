#!/usr/bin/env python3
"""Capacity ablation uses fixed geometry fit tolerance, no radar-based fitting."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radar_platform.dart_dataset import FLU_TO_NED
from radar_platform.colored_adc_background import expected_windowed_range_power
from audit_dart_geometry_rd_support import support
from evaluate_dart_frozen_batch import load,DATA,OUT as BATCH

OUT=BATCH.parent/'geometry_capacity_audit'

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    models={}
    for capacity in [8,16,32]:
        stem='lab-1_supported_planes_cell_0.25'+('' if capacity==8 else f'_maxplanes_{capacity}')
        meta=json.loads((DATA/'processed'/f'{stem}.json').read_text())
        with np.load(DATA/'processed'/f'{stem}.npz',allow_pickle=False) as f:
            centers=f['vertices_flu_m'][f['faces']].mean(axis=1)@FLU_TO_NED.T
        models[capacity]=(meta,centers)
    calibration=json.loads((BATCH/'frozen_calibration.json').read_text())
    model=json.loads((BATCH.parent/'colored_background_followup/frozen_colored_model.json').read_text())
    floor=calibration['gain']*expected_windowed_range_power(model['spectral_power'],np.hanning(512),np.hanning(256),8)
    indices=json.loads((BATCH/'evaluation_protocol.json').read_text())['temporal_check_valid_indices'];rows=[]
    for index in indices:
        request,adc,r,v,real=load(index)
        native=np.arange(512)*request.waveform.sample_rate_hz/512*299792458/(2*request.waveform.slope_hz_s)
        excess=np.maximum(real-np.interp(r,native,floor)[:,None],0)
        for capacity,(meta,points) in models.items():
            covered,limit=support(points,request,r,v)
            domain=(r[:,None]>=.3)&(r[:,None]<=5.4)&(abs(v)[None,:]<=limit)&(abs(v)[None,:]>1.01*abs(v[1]-v[0]))
            rows.append({'valid_frame_index':index,'max_planes':capacity,
                         'rd_area_fraction':float((covered&domain).sum()/domain.sum()),
                         'measured_excess_energy_fraction':float(excess[covered&domain].sum()/excess[domain].sum())})
    with (OUT/'per_frame_support.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'scattering_simulation_rerun':False,'real_rd_mae_improvement_proven':False,'geometry_fit_used_radar':False,
             'fixed_fit_tolerance_m':.04,'fixed_mesh_cell_m':.25,'same_geometry_fit_holdout_cells':True,
             'scene_count':1,'rd_frames':8,'models':{}}
    for capacity,(meta,points) in models.items():
        selected=[row for row in rows if row['max_planes']==capacity]
        summary['models'][str(capacity)]={'planes_found':meta['planes_found'],'triangles':len(points),
             'heldout_fraction_within_4cm_of_plane_rectangles':meta['heldout_fraction_within_4cm'],
             'heldout_p90_distance_to_plane_rectangles_m':meta['heldout_p90_distance_m'],
             'rd_area_fraction':float(np.mean([row['rd_area_fraction'] for row in selected])),
             'measured_excess_energy_fraction':float(np.mean([row['measured_excess_energy_fraction'] for row in selected]))}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout='constrained')
    for ax,key,title in zip(axes,['heldout_fraction_within_4cm_of_plane_rectangles','rd_area_fraction','measured_excess_energy_fraction'],
                            ['Held-out geometry near fitted planes','RD domain area covered','Observed excess energy covered (NOT accuracy)']):
        values=[summary['models'][str(c)][key]*100 for c in [8,16,32]]
        ax.bar(range(3),values);ax.set(xticks=range(3),xticklabels=['8 / 8','16 / 16','20 / 32'],ylabel='Percent',xlabel='Planes found / capacity',title=title,ylim=(0,100))
    fig.savefig(OUT/'geometry_capacity_comparison.png',dpi=160);plt.close(fig)
    table=''.join(f'<tr><td>{c}</td><td>{x["planes_found"]}</td><td>{x["triangles"]}</td><td>{x["heldout_fraction_within_4cm_of_plane_rectangles"]:.1%}</td><td>{x["measured_excess_energy_fraction"]:.1%}</td></tr>' for c,x in summary['models'].items())
    (OUT/'表面提取容量对照.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{max-width:1200px;margin:30px auto;font:16px/1.7 sans-serif}img{width:100%}td,th{border:1px solid #bbb;padding:9px}table{border-collapse:collapse}</style>
<h1>只增加表面提取容量，不放宽拟合容差</h1><p>同一份 LiDAR 输入、同一几何训练/检查划分、4 cm 容差、25 cm 面片网格和固定种子。仅改变允许提取的平面数量。雷达观测没有参与平面拟合。</p>
<img src="geometry_capacity_comparison.png"><table><tr><th>容量上限</th><th>实际平面数</th><th>三角面片数</th><th>检查点距有限平面矩形≤4 cm</th><th>RD 超背景观测能量覆盖</th></tr>TABLE</table>
<p>容量 32 时只找到 20 个满足固定支持点条件的平面，没有为了凑数量放宽门槛。几何距离使用平面矩形边界，孔洞处可能偏乐观；RD 覆盖则使用实际生成的面片中心，并报告其覆盖面积。</p>
<p><strong>本页还没有重跑回波仿真，不能说 RD MAE 已改善。</strong>覆盖增加不等于反射强度正确，新增平面的面积、法向、重叠、遮挡和材料仍需检查。下一步应把这组固定几何送入同一处理链，保留原模型与校准/检查划分，比较真实误差。</p><p><a href="summary.json">完整统计</a> · <a href="per_frame_support.csv">逐帧覆盖</a></p></html>'''.replace('TABLE',table))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
