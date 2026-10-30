#!/usr/bin/env python3
"""World-geometry comparison: show unchanged and calibration-only refitted gain."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dart_batch_metrics import db_power

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'docs/validation_diagnostics/2026-10-15/dart'
OUT=ROOT/'batch_v2_world_geometry'
DATA=Path('/nas/datasets/hyt/datasets/radar_open/dart_lab/processed')

def main():
    summary=json.loads((OUT/'summary.json').read_text())
    protocol=json.loads((OUT/'evaluation_protocol.json').read_text())
    previous=json.loads((ROOT/'colored_background_followup/summary.json').read_text())['models']['surface_plus_colored_noise']
    rows=list(csv.DictReader((OUT/'per_frame_metrics.csv').open()))
    indices=protocol['evaluation_indices'];reference=summary['calibration']['reference']
    errors={i:np.mean([float(r['primary_mae_db']) for r in rows if int(r['valid_frame_index'])==i and r['model']=='expanded_colored_refitted']) for i in indices}
    worst=max(indices,key=lambda i:errors[i]);examples=list(dict.fromkeys(indices[:3]+[worst]))
    (OUT/'images').mkdir(exist_ok=True)
    names=['Old cropped scene','Expanded, old gain','Expanded, refitted gain']
    model_metrics=[previous,summary['models']['expanded_colored_fixed'],summary['models']['expanded_colored_refitted']]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,region in zip(axes,['primary','geometry']):
        values=[m[region]['mean_mae_db'] for m in model_metrics]
        ax.bar(range(3),values,color=['#777777','#397ba2','#408f71'])
        for i,value in enumerate(values):ax.text(i,value,f'{value:.2f}',ha='center',va='bottom')
        ax.set(xticks=range(3),xticklabels=names,ylabel='MAE (dB)',title='Primary score' if region=='primary' else 'Same v1 geometry region')
        ax.tick_params(axis='x',labelrotation=15)
    fig.savefig(OUT/'images/geometry_model_comparison.png',dpi=160);plt.close(fig)
    illustrations=[]
    for index in examples:
        with np.load(DATA/'colored_background_followup'/f'f{index:04d}.npz',allow_pickle=False) as f:
            old=db_power(f['simulated_power'],reference)
        with np.load(DATA/'batch_v2_world_geometry/evaluation'/f'f{index:04d}.npz',allow_pickle=False) as f:
            real=db_power(f['real_power'],reference);fixed=db_power(f['expanded_colored_fixed'],reference)
            refit=db_power(f['expanded_colored_refitted'],reference);ranges=f['range_m'];velocity=f['velocity_mps']
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        arrays=[real,old,old-real,fixed,refit,refit-real]
        titles=['Measured','Old cropped geometry','Old simulation minus measured','Expanded geometry, OLD gain','Expanded geometry, refitted gain','Expanded refitted minus measured']
        for i,(ax,values,title) in enumerate(zip(axes.flat,arrays,titles)):
            diff=i in [2,5]
            image=ax.pcolormesh(velocity,ranges,values,shading='auto',cmap='RdBu_r' if diff else 'viridis',
                                 vmin=-30 if diff else -60,vmax=30 if diff else 10)
            ax.set(xlabel='Closing velocity (m/s)',ylabel='Range (m)',title=title)
            fig.colorbar(image,ax=ax,label='dB difference' if diff else 'dB, SAME calibration reference')
        fig.suptitle(f'Frame {index}, fixed seed 11'+(' | worst expanded/refitted primary MAE' if index==worst else ''))
        filename=f'images/frame_{index:04d}.png';fig.savefig(OUT/filename,dpi=140);plt.close(fig)
        illustrations.append(f'<h3>有效帧 {index}'+('（新几何重校准后的最差帧）' if index==worst else '')+f'</h3><img src="{filename}">')
    captions=['旧裁剪几何＋有色背景','扩展世界几何，保留旧幅度比例','扩展世界几何，仅用校准帧重估幅度']
    table=''.join('<tr><td>'+label+'</td>'+''.join(f'<td>{m[region][key]:.2f}</td>' for region,key in [('primary','mean_mae_db'),('primary','p90_mae_db'),('geometry','mean_mae_db')])+'</tr>' for label,m in zip(captions,model_metrics))
    changes={'examples':examples,'worst_refitted_frame':worst,'seed_for_figures':11,
             'fixed_gain_primary_improvement_db':previous['primary']['mean_mae_db']-model_metrics[1]['primary']['mean_mae_db'],
             'refitted_gain_primary_improvement_db':previous['primary']['mean_mae_db']-model_metrics[2]['primary']['mean_mae_db'],
             'refitted_gain_geometry_improvement_db':previous['geometry']['mean_mae_db']-model_metrics[2]['geometry']['mean_mae_db'],
             'positive_improvement_means_lower_error':True,'independent_holdout_scenes':0}
    (OUT/'report_selection.json').write_text(json.dumps(changes,indent=2))
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{max-width:1400px;margin:30px auto;font:16px/1.7 sans-serif}img{width:100%}th,td{border:1px solid #bbb;padding:9px}table{border-collapse:collapse;width:100%}.note{background:#fff0ce;padding:15px}</style>
<h1>扩大固定世界几何，是否真正改善 RD？</h1><p class="note">仍为同一场景：3 帧校准＋8 帧开发检查。此前已看过这些帧，不能当作新的独立留出验证。本页同时保留不重新缩放和只用校准帧重新缩放的结果。</p>
<p>本次结果不能概括为“几何越多越准确”：保留旧比例时，主指标略有退化，旧几何区域的误差也增大；只用原校准帧重新估计一个幅度比例后，两个指标才改善。因此，改善依赖几何模型与幅度校准的组合，尚未证明材料反射强度正确。</p>
<table><tr><th>模型</th><th>主指标平均 MAE</th><th>主指标 P90</th><th>相同旧几何区域 MAE</th></tr>TABLE_ROWS</table>
<p>单位均为 dB，越低越好。主指标排除实测上游处理过的中央三个速度格；几何区域严格沿用旧掩码，不通过扩大统计区域改变分母。P90 表示 90% 的帧误差不超过该值。</p><img src="images/geometry_model_comparison.png">
<h2>真实图、旧仿真、新仿真和差值</h2><p>相同显示参考和 −60 dB 下限，没有各图独立按峰值归一化。红色为仿真偏强，蓝色为仿真偏弱；差值显示截在 ±30 dB，计算不按显示范围截断。</p>EXAMPLES
<h2>哪些变了，哪些保持不变？</h2><p>几何从第一帧视野内的 3,788 个三角面片，扩展为覆盖整段轨迹固定世界范围的 19,090 个面片。不是每帧重建目标。波形、采样轨迹、波束、面元尺寸、相对散射系数及噪声功率参考不变。</p>
<p>两种幅度条件分别为旧比例和仅用原三个校准帧重新估计的单一比例；调整比例时反向缩放 ADC 噪声方差，使报告单位中的噪声功率和随机实现不变。</p>
<p>需要注意：场景裁剪范围、拟合平面和网格划分发生变化，虽然种子相同，漫反射相位不是逐面片一一配对，因此不是纯粹单参数的电磁散射实验。完整材料、遮挡、多径和实测方向图仍缺失。图像更接近也不代表这些物理因素已验证。</p>
<p>数据来源：<a href="https://zenodo.org/records/10938617">DART 官方记录</a>（CC BY 4.0，正式作者与引用见原记录）。图为本项目处理与仿真的派生结果。</p>
<p><a href="manifest.json">冻结输入与代码记录</a> · <a href="evaluation_protocol.json">预先固定的评价规则</a> · <a href="frozen_calibration.json">两种幅度比例</a> · <a href="per_frame_metrics.csv">完整逐帧指标</a> · <a href="summary.json">汇总</a></p></html>'''
    (OUT/'扩展世界几何RD对照.html').write_text(html.replace('TABLE_ROWS',table).replace('EXAMPLES',''.join(illustrations)))
    print(json.dumps(changes),flush=True)

if __name__=='__main__':main()
