#!/usr/bin/env python3
"""Build the final comparison only from complete, model-bound experiment results."""
import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from radar_platform.navigation.io import local_path,read_json,write_json
from radar_platform.navigation.episode import model_fingerprint


def campaign(path,expected,*,model_dir=None):
    path=local_path(path);status=read_json(path/'status.json')
    if status.get('status')!='complete':raise ValueError(f'campaign is incomplete: {path}')
    rows=read_json(path/'results.json')
    if len(rows)!=expected or len({r['episode'] for r in rows})!=expected:raise ValueError('campaign case count/IDs do not match declared protocol')
    if model_dir is not None:
        fp=model_fingerprint(model_dir);protocol=read_json(path/'protocol.json')
        specs={r['id']:r for r in protocol['records']}
        if set(specs)!={r['episode'] for r in rows}:raise ValueError('campaign omitted or added a declared case')
        for row in rows:
            run=path/row['episode'];p=read_json(run/'protocol.json');actual=read_json(run/'complete.json')
            if actual!=row or p['spec']!=specs[row['episode']]:raise ValueError('campaign aggregate differs from underlying episode')
            if p.get('model_sha256')!=fp:raise ValueError('campaign uses different or unverified weights')
            if not row.get('complete_experiment'):raise ValueError('development frame limit is not a completed experiment')
    return rows


def summary(rows):
    return dict(cases=len(rows),successes=sum(r['success'] for r in rows),success_rate=float(np.mean([r['success'] for r in rows])),
        horizontal_mean_m=float(np.mean([r['horizontal_error_m'] for r in rows])),
        horizontal_p90_m=float(np.percentile([r['horizontal_error_m'] for r in rows],90)),
        height_mean_m=float(np.mean([r['height_error_m'] for r in rows])),
        hover_3d_mean_m=float(np.mean([r['hover_target_error_3d_m'] for r in rows])),
        simulation_time_mean_s=float(np.mean([r['elapsed_simulation_s'] for r in rows])),
        terminations={t:sum(r['termination']==t for r in rows) for t in sorted({r['termination'] for r in rows})})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True);p.add_argument('--terrain-campaign',required=True)
    p.add_argument('--models',required=True);p.add_argument('--terrain-models',required=True)
    p.add_argument('--demo',required=True);p.add_argument('--no-control',required=True)
    p.add_argument('--oracle-control',required=True);p.add_argument('--angle-comparison',required=True)
    p.add_argument('--inference-audit',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=local_path(args.output);root.mkdir(parents=True,exist_ok=True)
    main_rows=campaign(args.campaign,48,model_dir=args.models)
    terrain_rows=campaign(args.terrain_campaign,24,model_dir=args.terrain_models)
    bare=campaign(args.no_control,48);oracle=campaign(args.oracle_control,48)
    normal=[r for r in main_rows if r['episode'].startswith('normal_')]
    stress=[r for r in main_rows if r['episode'].startswith('stress_')]
    if len(normal)!=24 or len(stress)!=24:raise ValueError('normal/stress allocation changed')
    normal_ids={r['episode'] for r in normal}
    if {r['episode'] for r in terrain_rows}!=normal_ids:raise ValueError('terrain control is not paired to normal cases')
    grouped={'RDA + building':normal,'RDA terrain only':terrain_rows,
             'No correction':[r for r in bare if r['episode'] in normal_ids],
             'Truth control':[r for r in oracle if r['episode'] in normal_ids]}
    stats={name:summary(rows) for name,rows in grouped.items()}
    pressure={}
    for group in sorted({'_'.join(r['episode'].split('_')[:-1]) for r in stress}):
        pressure[group]=summary([r for r in stress if r['episode'].startswith(group+'_')])
    models=local_path(args.models);terrain_models=local_path(args.terrain_models)
    learned_metrics=read_json(models/'test_metrics.json');terrain_metrics=read_json(terrain_models/'test_metrics.json')
    if learned_metrics.get('evaluation_counts',{}).get('total_states')!=1024 or terrain_metrics.get('evaluation_counts',{}).get('total_states')!=1024:
        raise ValueError('both model conditions require all1024 predeclared test states')
    audit=read_json(local_path(args.inference_audit)/'audit.json')
    if not audit.get('predictions_match_frozen_batch') or audit.get('attempted_forbidden_reads'):
        raise ValueError('independent inference boundary audit did not pass')
    if audit.get('model_sha256')!=model_fingerprint(models) or audit.get('split')!='test':
        raise ValueError('independent inference audit does not cover the final frozen model and test split')
    demo_root=local_path(args.demo);demo=read_json(demo_root/'complete.json');render=read_json(demo_root/'render_manifest.json')
    if render.get('development_only'):raise ValueError('development-pool model cannot stand in for final trained-model evidence')
    if demo['mode']!='learned' or read_json(demo_root/'protocol.json').get('model_sha256')!=model_fingerprint(models):
        raise ValueError('representative movie is not a flight with the evaluated learned model')
    for name in ['flight.mp4','flight.gif','report.html']:
        if not (demo_root/name).is_file():raise ValueError(f'missing representative artifact {name}')
    stress_demo_root=local_path(args.campaign)/'stress_speed_12_0'
    stress_demo=read_json(stress_demo_root/'complete.json')
    stress_render=read_json(stress_demo_root/'render_manifest.json')
    stress_protocol=read_json(stress_demo_root/'protocol.json')
    if stress_demo['mode']!='learned' or stress_protocol.get('model_sha256')!=model_fingerprint(models):
        raise ValueError('speed-stress movie must use the evaluated learned model')
    if stress_render.get('development_only') or stress_render['radar_frames_interpolated']:
        raise ValueError('speed-stress movie must show actual acquisitions from the formal model')
    for name in ['flight.mp4','flight.gif','report.html']:
        if not (stress_demo_root/name).is_file():raise ValueError(f'missing speed-stress artifact {name}')
    height_demo_root=local_path(args.campaign)/'stress_height_60_0'
    height_demo=read_json(height_demo_root/'complete.json')
    height_render=read_json(height_demo_root/'render_manifest.json')
    height_protocol=read_json(height_demo_root/'protocol.json')
    if height_demo['mode']!='learned' or height_protocol.get('model_sha256')!=model_fingerprint(models):
        raise ValueError('height-stress movie must use the evaluated learned model')
    if height_render.get('development_only') or height_render['radar_frames_interpolated']:
        raise ValueError('height-stress movie must show actual acquisitions from the formal model')
    for name in ['flight.mp4','flight.gif','report.html']:
        if not (height_demo_root/name).is_file():raise ValueError(f'missing height-stress artifact {name}')
    angle=read_json(local_path(args.angle_comparison)/'comparison.json')
    acceptance=dict(normal_success_rate_at_least_90_percent=stats['RDA + building']['success_rate']>=.9,
        representative_rda_flight_success=demo['success'],all_48_main_cases_retained=True,all_24_terrain_cases_retained=True,
        independent_inference_matches=True,truth_reads_denied=audit['truth_and_scene_reads_denied'],
        video_from_actual_acquisitions=not render['radar_frames_interpolated'])
    result=dict(normal=stats,stress=pressure,offline_test=learned_metrics,terrain_offline_test=terrain_metrics,
        representative=demo,stress_representative=stress_demo,stress_representative_spec=stress_protocol['spec'],
        height_representative=height_demo,height_representative_spec=height_protocol['spec'],
        acceptance=acceptance,scope='Fixed real DSM, assumed diffuse building scattering, bounded kinematic flight; no unknown-map or measured-flight accuracy claim')
    write_json(root/'summary.json',result)
    fields=['condition','episode','success','termination','horizontal_error_m','height_error_m','hover_target_error_3d_m',
            'ground_target_distance_3d_m','final_speed_mps','elapsed_simulation_s','truth_hold_duration_s','trajectory_length_m']
    with (root/'cases.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for condition,rows in [('RDA + building',main_rows),('RDA terrain only',terrain_rows),('No correction',bare),('Truth control',oracle)]:
            for row in sorted(rows,key=lambda x:x['episode']):writer.writerow({'condition':condition,**{k:row[k] for k in fields[1:]}})
    plt.rcParams.update({'font.size':11,'figure.facecolor':'white'})
    fig,axes=plt.subplots(1,3,figsize=(15,4.5),constrained_layout=True)
    names=list(stats);colors=['#1660c9','#bf4b95','#a4acb7','#159b61'];x=np.arange(len(names))
    for ax,key,label in [(axes[0],'success_rate','Successful arrivals / 24'),(axes[1],'horizontal_mean_m','Mean final horizontal error [m]'),(axes[2],'height_mean_m','Mean final height error [m]')]:
        values=[stats[n][key] for n in names];ax.bar(x,values,color=colors);ax.set_xticks(x,names,rotation=20,ha='right');ax.set_ylabel(label);ax.grid(axis='y',alpha=.2)
        for i,value in enumerate(values):ax.text(i,value,f'{value:.3g}',ha='center',va='bottom')
    axes[0].axhline(.9,color='#222',ls='--',alpha=.5);axes[0].set_ylim(0,1.1)
    fig.savefig(root/'normal_comparison.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True);groups=list(pressure);x=np.arange(len(groups))
    axes[0].bar(x,[pressure[g]['success_rate'] for g in groups],color='#1660c9');axes[0].set_ylim(0,1.1);axes[0].set_ylabel('Successful arrivals / 3')
    axes[1].bar(x,[pressure[g]['horizontal_mean_m'] for g in groups],color='#de8f1d');axes[1].set_ylabel('Mean final horizontal error [m]')
    for ax in axes:ax.set_xticks(x,[g.replace('stress_','').replace('_',' ') for g in groups],rotation=35,ha='right');ax.grid(axis='y',alpha=.2)
    fig.savefig(root/'stress_comparison.png',dpi=150);plt.close(fig)
    def table(rows,columns):
        head=''.join(f'<th>{html.escape(c)}</th>' for c in columns)
        body=''.join('<tr>'+''.join(f'<td>{html.escape(str(row.get(c,"")))}</td>' for c in columns)+'</tr>' for row in rows)
        return f'<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
    normal_table=table([{'condition':name,**s} for name,s in stats.items()],['condition','cases','successes','success_rate','horizontal_mean_m','height_mean_m'])
    pressure_table=table([{'condition':name,**s} for name,s in pressure.items()],['condition','cases','successes','horizontal_mean_m','height_mean_m','terminations'])
    detail_table=table(sorted(main_rows,key=lambda r:r['episode']),['episode','success','termination','horizontal_error_m','height_error_m','elapsed_simulation_s'])
    def metric_text(values,key):
        value=values.get(key)
        return '—' if value is None else f'{float(value):.3f}'
    offline_rows=[];baseline_rows=[];axis_rows=[]
    for condition,values in [('含中心建筑',learned_metrics),('纯地形',terrain_metrics)]:
        counts=values['evaluation_counts'];all_states=values['cnn_ensemble_all_states'];accepted=values['cnn_accepted_estimates']
        offline_rows.append({'条件':condition,'全部状态数':counts['total_states'],
            '条件有效数':counts['eligible_regression_states'],'在线接受数':counts['accepted_estimates'],
            '接受但误差>3m':counts['accepted_estimates_over_3m'],
            '全状态位置均误(m)':metric_text(all_states,'position_mean_m'),
            '全状态位置P90(m)':metric_text(all_states,'position_p90_m'),
            '全状态速度均误(m/s)':metric_text(all_states,'velocity_mean_mps'),
            '接受位置均误(m)':metric_text(accepted,'position_mean_m')})
        axis_rows.append({'条件':condition,'前向x平均绝对误差(m)':f"{all_states['axis_mae'][0]:.3f}",
            '右向y平均绝对误差(m)':f"{all_states['axis_mae'][1]:.3f}",
            '向上z平均绝对误差(m)':f"{all_states['axis_mae'][2]:.3f}"})
        for key,label in [('cnn_ensemble','CNN集成'),('nearest_neighbor','最近邻'),
                          ('shuffled_input_control','输入打乱'),('training_mean','训练均值')]:
            metrics_row=values[key]
            baseline_rows.append({'条件':condition,'方法':label,'样本数':metrics_row['cases'],
                '位置均误(m)':metric_text(metrics_row,'position_mean_m'),
                '位置P90(m)':metric_text(metrics_row,'position_p90_m'),
                '速度均误(m/s)':metric_text(metrics_row,'velocity_mean_mps')})
    offline_table=table(offline_rows,list(offline_rows[0]))
    baseline_table=table(baseline_rows,list(baseline_rows[0]))
    axis_table=table(axis_rows,list(axis_rows[0]))
    relative_demo=Path(os.path.relpath(demo_root,root))
    relative_stress_demo=Path(os.path.relpath(stress_demo_root,root))
    relative_height_demo=Path(os.path.relpath(height_demo_root,root))
    stress_limits=stress_protocol['spec']['limits']
    stress_initial_speed=float(np.linalg.norm(stress_protocol['spec']['velocity_neu_mps']))
    stress_success_text='通过' if stress_demo['success'] else '未通过'
    height_success_text='通过' if height_demo['success'] else '未通过'
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>雷达闭环纠偏实验报告</title>
<style>body{{font:16px/1.6 system-ui;max-width:1400px;margin:30px auto;padding:0 24px;color:#203049;background:#f4f7fb}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{border:1px solid #d6dfe9;padding:8px;text-align:left}}img,video{{width:100%;border-radius:10px}}pre{{white-space:pre-wrap;background:white;padding:20px}}a{{color:#165bb0}}</style>
<h1>真实 DEM 与中心建筑上的仅雷达图闭环纠偏</h1>
<p>正常工况成功 {stats['RDA + building']['successes']}/24；预设验收门限为至少90%。成功要求水平误差≤2米、高度误差≤1.5米、速度≤0.3米/秒连续保持3秒，并在60秒内由在线控制器宣告到达。压力工况逐项保留成功与失败。</p>
<h2>代表性完整飞行</h2><video controls src="{relative_demo}/flight.mp4"></video><p><a href="{relative_demo}/report.html">逐帧联动报告</a> · <a href="{relative_demo}/flight.gif">GIF</a> · <a href="cases.csv">全部案例CSV</a></p>
<p>图中的雷达数据逐帧来自回波合成。机体系位置与速度仅由雷达图推断；世界坐标图使用模拟航向作显示转换。目标为DEM中心地面点正上方30米，因此成功悬停时距地面点的三维距离仍约为30米。</p>
<h2>正常工况与配对对照</h2><img src="normal_comparison.png">{normal_table}
<p>无纠偏与真值控制是单独标记的运动对照。它们的控制指令不依赖雷达图，因此省略不影响运动结果的图像计算；完整真值控制参考已校验同采集时钟下的轨迹完全一致。不同终止原因可能产生不同的终止时刻，表中误差均为各自实际终止位置。真值参考同样采用末端死区和停止规则，其终点误差不表示最小可达误差下界。</p>
<h2>压力工况</h2><img src="stress_comparison.png">{pressure_table}
<p>噪声压力档将接收端复ADC噪声方差乘10；本实验对应由10⁻⁵增至10⁻⁴。radar_localization_lost表示连续2秒无法形成满足有效性与误差尺度门限的定位而终止该次仿真，这类失败计入完整结果。</p>
<h3>预设的首个12米/秒案例</h3><video controls src="{relative_stress_demo}/flight.mp4"></video>
<p><a href="{relative_stress_demo}/report.html">高速案例逐帧报告</a> · <a href="{relative_stress_demo}/flight.gif">GIF</a> · <a href="{relative_stress_demo}/protocol.json">案例配置</a></p>
<p>该案例初始速度为{stress_initial_speed:g}米/秒，水平加速度上限为{stress_limits['horizontal_acceleration_mps2']:g}米/秒²，偏航角速度上限为{stress_limits['yaw_rate_deg_s']:g}度/秒。速度压力档使用协议中声明的运动上限，其结果反映速度与这些运动约束的组合。该案例实际验收{stress_success_text}，最终水平误差{stress_demo['horizontal_error_m']:.3f}米、高度误差{stress_demo['height_error_m']:.3f}米。</p>
<h3>预设的首个60米起始高度案例</h3><video controls src="{relative_height_demo}/flight.mp4"></video>
<p><a href="{relative_height_demo}/report.html">高起始高度案例逐帧报告</a> · <a href="{relative_height_demo}/flight.gif">GIF</a> · <a href="{relative_height_demo}/protocol.json">案例配置</a></p>
<p>该案例从相对目标地面{height_protocol['spec']['position_neu_m'][2]:g}米高度接近30米悬停目标，垂直速度上限为{height_protocol['spec']['limits']['vertical_speed_mps']:g}米/秒。实际验收{height_success_text}，最终水平误差{height_demo['horizontal_error_m']:.3f}米、高度误差{height_demo['height_error_m']:.3f}米。</p>
<h2>冻结模型与独立测试</h2>{offline_table}
<p>全状态指标保留全部1024个预定状态，用于两组比较。含建筑条件的“条件有效”表示至少一个物理建筑散射单元有贡献，纯地形条件则包含全部有限观测。在线接受数由冻结的有效性评分与位置误差尺度门限共同决定；接受数为0时，该项误差显示为“—”。误差尺度是验证残差的标定量，表中同时列出被接受但真实位置误差大于3米的状态数。</p>
<h3>相对位置三轴误差（全部状态）</h3>{axis_table}
<p>坐标采用机体系：x前向、y右向、z向上。三轴误差的标签为地面目标相对向量；飞行验收采用地面目标正上方30米的悬停点。</p>
<h3>条件内基线比较</h3>{baseline_table}
<p>每种条件内的四种方法使用相同的条件有效状态。位置与速度分别统计；最近邻超参数仅通过验证集选择。</p>
<details><summary>含建筑条件的完整测试指标</summary><pre>{html.escape(json.dumps(learned_metrics,ensure_ascii=False,indent=2))}</pre></details>
<details><summary>纯地形条件的完整测试指标</summary><pre>{html.escape(json.dumps(terrain_metrics,ensure_ascii=False,indent=2))}</pre></details>
<p>测试前冻结权重，模型选择不使用测试集。数据按初始状态空间单元划分，训练、验证、测试记录的相位与噪声种子均无交集。独立推断在禁止读取实验真值与场景文件的条件下与批量预测核对。训练和测试仍属于同一真实地域，独立相位种子不等于独立地图。</p>
<h2>单角度与双角度诊断</h2><pre>{html.escape(json.dumps(angle,ensure_ascii=False,indent=2))}</pre><p>两种阵列的通道数与孔径也有差异，该对照不能单独归因于增加一个角度维。</p>
<h2>全部主实验案例</h2>{detail_table}
<h2>验收状态与适用范围</h2><pre>{html.escape(json.dumps(acceptance,ensure_ascii=False,indent=2))}</pre>
<p>中心建筑为48个三角质心漫散射单元，同一飞行相位固定；每个Tx/Rx时隙分别计算地形及建筑遮挡。真实DSM不等于实测散射参数，运动模型包含有界平移与航向变化，不包含完整旋翼动力学或肥料落点。回波合成与推断离线执行，飞行动力学按虚拟采集时钟推进，未计入实机计算和通信延迟。</p></html>'''
    (root/'report.html').write_text(page)
    write_json(root/'artifact_manifest.json',dict(report='report.html',csv='cases.csv',figures=['normal_comparison.png','stress_comparison.png'],
        representative_movie=str(demo_root/'flight.mp4'),stress_representative_movie=str(stress_demo_root/'flight.mp4'),
        height_representative_movie=str(height_demo_root/'flight.mp4'),acceptance=acceptance,
        source_paths=dict(oracle_control=str(local_path(args.oracle_control)),no_control=str(local_path(args.no_control)),
            angle_comparison=str(local_path(args.angle_comparison)),inference_audit=str(local_path(args.inference_audit)),
            stress_demo=str(stress_demo_root),height_demo=str(height_demo_root)),
        main_model_sha256=model_fingerprint(models),terrain_model_sha256=model_fingerprint(terrain_models)))
    print(json.dumps(dict(output=str(root),acceptance=acceptance)))


if __name__=='__main__':
    main()
