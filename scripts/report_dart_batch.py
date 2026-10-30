#!/usr/bin/env python3
"""Render fixed-order/worst-case comparisons from completed frozen-batch scores."""
import csv,html,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dart_batch_metrics import db_power

PROJECT=Path(__file__).resolve().parents[1]
OUT=PROJECT/'docs/validation_diagnostics/2026-10-15/dart/batch_v1'
CACHE=Path('/nas/datasets/hyt/datasets/radar_open/dart_lab/processed/batch_v1/evaluation')

def main():
    summary=json.loads((OUT/'summary.json').read_text())
    protocol=json.loads((OUT/'evaluation_protocol.json').read_text())
    rows=list(csv.DictReader((OUT/'per_frame_metrics.csv').open()))
    indices=protocol['temporal_check_valid_indices']
    by_frame={i:np.mean([float(row['primary_mae_db']) for row in rows if int(row['valid_frame_index'])==i and row['model']=='surface_plus_noise']) for i in indices}
    worst=max(indices,key=lambda i:by_frame[i]);examples=list(dict.fromkeys(indices[:3]+[worst]))
    ref=summary['calibration']['shared_power_reference'];seed=protocol['noise_seed_list'][0]
    (OUT/'images').mkdir(exist_ok=True)
    labels={'surface':'Surface only','noise_only':'Noise only','surface_plus_noise':'Surface + noise'}
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,region in zip(axes,['primary','geometry','off_kinematic']):
        for j,model in enumerate(labels):
            values=[]
            for i in indices:
                numbers=[float(row[f'{region}_mae_db']) for row in rows if int(row['valid_frame_index'])==i and row['model']==model and row[f'{region}_mae_db']]
                if numbers:values.append(np.mean(numbers))
            if values:
                ax.bar(j,np.mean(values),color=['#4078a8','#999999','#429b72'][j],alpha=.65)
                ax.scatter(j+np.linspace(-.13,.13,len(values)),values,c='black',s=12)
            else:
                ax.text(j,0,'No supported region',rotation=90,ha='center')
        ax.set(xticks=range(3),xticklabels=list(labels.values()),ylabel='MAE (dB)',title=region)
        ax.tick_params(axis='x',labelrotation=20)
    fig.suptitle('8 temporal-check frames, 1 scene; seeds averaged within each frame')
    fig.savefig(OUT/'images/model_comparison.png',dpi=150);plt.close(fig)
    pictures=[]
    for i in examples:
        with np.load(CACHE/f'f{i:04d}_power_products.npz',allow_pickle=False) as data:
            real=db_power(data['real_power'],ref);sim=db_power(data[f'surface_plus_noise_seed_{seed}'],ref)
            ranges=data['range_m'];velocity=data['closing_velocity_mps']
        difference=sim-real
        row=next(row for row in rows if int(row['valid_frame_index'])==i and row['model']=='surface_plus_noise' and int(row['seed'])==seed)
        fig,axes=plt.subplots(1,3,figsize=(15,4),layout='constrained')
        for ax,values,title,is_diff in zip(axes,[real,sim,difference],['Measured','Simulated surface + noise','Simulation minus measured'],[False,False,True]):
            im=ax.pcolormesh(velocity,ranges,values,shading='auto',cmap='RdBu_r' if is_diff else 'viridis',
                             vmin=-30 if is_diff else -60,vmax=30 if is_diff else 10)
            ax.set(xlabel='Closing velocity (m/s)',ylabel='Range (m)',title=title)
            fig.colorbar(im,ax=ax,label='dB difference' if is_diff else 'dB, shared calibration reference')
        geometry_label=f'{float(row["geometry_mae_db"]):.2f} dB' if row['geometry_mae_db'] else 'unavailable (empty support)'
        fig.suptitle(f'Valid frame {i}, seed {seed} | primary MAE={float(row["primary_mae_db"]):.2f} dB | geometry MAE={geometry_label}'+(' | worst frame by seed-mean MAE' if i==worst else ''))
        filename=f'images/frame_{i:04d}.png';fig.savefig(OUT/filename,dpi=150);plt.close(fig)
        pictures.append(f'<h3>有效帧 {i}'+('（按种子平均误差最大的帧）' if i==worst else '')+f'</h3><img src="{filename}">')
    table=[]
    names={'surface':'仅面回波','noise_only':'仅高斯噪声','surface_plus_noise':'面回波＋高斯噪声'}
    for model in labels:
        value=summary['models'][model]
        fmt=lambda region,key: '不可用' if value[region][key] is None else f'{value[region][key]:.2f}'
        table.append('<tr><td>'+names[model]+'</td>'+''.join('<td>'+fmt(region,key)+'</td>' for region,key in [('primary','mean_mae_db'),('primary','median_mae_db'),('primary','p90_mae_db'),('geometry','mean_mae_db'),('off_kinematic','mean_mae_db')])+'</tr>')
    surface=summary['models']['surface'];combined=summary['models']['surface_plus_noise'];noise=summary['models']['noise_only']
    changes={'primary_mae_improvement_db':surface['primary']['mean_mae_db']-combined['primary']['mean_mae_db'],
             'geometry_mae_improvement_db':None if surface['geometry']['mean_mae_db'] is None or combined['geometry']['mean_mae_db'] is None else surface['geometry']['mean_mae_db']-combined['geometry']['mean_mae_db'],
             'noise_only_primary_mae_db':noise['primary']['mean_mae_db'],'worst_frame':worst,'examples':examples,
             'display_seed':seed,'seed_count':3,'independent_scenes':1,'independent_holdout_scenes':0,
             'amplitude_reference_is_calibration_only':True,'no_per_frame_peak_normalization':True}
    (OUT/'report_selection.json').write_text(json.dumps(changes,indent=2))
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>DART 面目标与背景批量检查</title><style>body{max-width:1400px;margin:30px auto;font:16px/1.7 sans-serif;color:#172634}img{width:100%}table{border-collapse:collapse;width:100%}th,td{border:1px solid #bbb;padding:9px;text-align:left}.note{padding:16px;background:#fff0ca}h2{margin-top:30px}</style>
<h1>面目标与高斯背景：固定配置批量检查</h1><p class="note">11 个仿真帧：3 个校准帧＋8 个时间段检查帧，来自同一次采集、同一个场景，仅覆盖最早有效轨迹中的一小段。不是 11 个独立场景，也不代表整次采集。下面有真实差值与 MAE，但尚不能当作独立场景保真验收。</p>
<p>CONCLUSION</p>
<h2>结果汇总（单位 dB，越低越好）</h2><p>噪声重复 3 个固定种子，先在每帧内平均，再汇总 8 帧；P90 表示 90% 的帧误差不超过该值。黑点是一帧的种子平均结果，不是一个新场景。</p>
<table><tr><th>模型</th><th>主指标均值</th><th>中位数</th><th>P90</th><th>几何对应区域均值</th><th>运动范围外区域均值</th></tr>TABLE_ROWS</table>
<img src="images/model_comparison.png"><p>主指标：相同物理网格上的 dB MAE，排除公开处理中已扣背景的中央三个速度格；CSV 另保留包含它们的全图 MAE。几何对应区域来自固定仿真几何，不是独立目标标注，因此必须与全图一起读，不能借此忽略漏建目标。</p>
<h2>固定顺序三帧＋最差帧</h2><p>全部图片使用同一校准功率参考和 −60 dB 下限，没有各自按峰值归一化。示例固定种子 11；差值为仿真减实测，红色表示仿真偏强、蓝色表示仿真偏弱。差值显示裁剪到 ±30 dB，但计算不按这个显示范围裁剪。未覆盖的坐标格保留为空白并计数，不补成零误差。</p>EXAMPLES
<h2>输入、校准与限制</h2><table><tr><th>部分</th><th>实际做法</th></tr><tr><td>面目标输入</td><td>独立 LiDAR 地图提取的部分表面、固定 25 cm 网格、统一相对漫反射系数与固定随机相位；不逐帧按照实测图改目标。</td></tr><tr><td>平台与雷达</td><td>公开轨迹按 chirp 插值位置与姿态；公开配置候选、理想 2Tx4Rx 阵列、固定 65° 半角和 5° 平滑边缘。不是实测天线方向图。</td></tr><tr><td>幅度</td><td>只用三个校准帧估计一个总功率比例，冻结后应用到检查帧。这是相对尺度校准，不是材料 RCS 标定。</td></tr><tr><td>高斯背景</td><td>只用校准帧在静态场景运动范围外的功率中位数估计底噪代理；转换为复数 ADC 噪声方差，加入 ADC 后重新 FFT。“仅噪声”是防止背景主导 MAE 的负对照。</td></tr><tr><td>仍未复现</td><td>完整场景、材料差异、遮挡、环境多径、实测方向图、完整接收机以及真实上游背景扣除。时间段检查不能替代独立采集或跨场景验证。</td></tr></table>
<p>数据来源：<a href="https://zenodo.org/records/10938617">DART 作者发布的官方数据记录</a>（CC BY 4.0；作者与正式引用见原记录），<a href="https://github.com/WiseLabCMU/dart">官方项目</a>。本报告的实测图经过角度功率平均、速度坐标转换和统一参考 dB 显示；仿真图为本平台基于其 LiDAR 几何及轨迹生成，不是作者原始仿真结果。</p>
<p><a href="manifest.json">固定批次清单与输入校验</a> · <a href="evaluation_protocol.json">预先固定的评价规则</a> · <a href="frozen_calibration.json">校准参数</a> · <a href="per_frame_metrics.csv">全部逐帧指标 CSV</a> · <a href="summary.json">汇总数据</a></p></html>'''
    text=text.replace('TABLE_ROWS',''.join(table)).replace('EXAMPLES',''.join(pictures))
    conclusion=(f'本批结果：加高斯噪声后，主指标 MAE 从 {surface["primary"]["mean_mae_db"]:.2f} 降到 {combined["primary"]["mean_mae_db"]:.2f} dB；'
                f'几何对应区域从 {surface["geometry"]["mean_mae_db"]:.2f} 降到 {combined["geometry"]["mean_mae_db"]:.2f} dB。'
                '改善主要来自背景，不能据此说面目标已经仿真准确。仅噪声在整图上也能取得看似接近的分数，但在几何对应区域明显更差，这正是保留负对照的原因。')
    text=text.replace('CONCLUSION',conclusion)
    (OUT/'DART面目标与背景批量报告.html').write_text(text)
    print(json.dumps(changes),flush=True)

if __name__=='__main__':main()
