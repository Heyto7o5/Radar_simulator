#!/usr/bin/env python3
"""Current evidence index without rewriting archived reports or claiming fidelity."""
import json
from datetime import datetime,timezone
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
DOC=PROJECT/'docs'
OUT=DOC/'deliveries/current_extension_evidence'

def read(path):return json.loads((DOC/path).read_text())

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sea=read('background_extension/2026-10-15/metrics/summary.json')
    grass=read('background_extension/2026-10-15/metrics/ground_summary.json')
    dart=read('validation_diagnostics/2026-10-15/dart/colored_background_followup/summary.json')
    delay=read('validation_diagnostics/2026-10-15/metrics/maqu_delay_summary.json')
    state=read('validation_diagnostics/2026-10-15/dart/batch_v2_world_geometry/state.json')
    integrity=read('validation_diagnostics/2026-10-15/dart/batch_v2_world_geometry/output_integrity_audit.json')
    world_path=DOC/'validation_diagnostics/2026-10-15/dart/batch_v2_world_geometry/summary.json'
    world=json.loads(world_path.read_text()) if world_path.exists() else None
    sea_metrics={row['model']:row['mean_rd_mae_db']['mean'] for row in sea['models']}
    completed=sum(job['status']=='complete' for job in state['jobs'].values())
    evidence={'as_of_utc':datetime.now(timezone.utc).isoformat(),
              'surface':{'implemented':['diffuse area cells','explicit scalar single-bounce PO quadrature','pulsed LFM and FMCW complex ADC/RD',
                                        'sampled platform translation and attitude','optional smooth beam edge'],
                         'measured_static_reference':'Maqu metal plate; cross-session discrepancy preserved',
                         'measured_dynamic_reference':'DART processed RAD with independent LiDAR and published trajectory',
                         'dart_temporal_check_frames':8,'dart_independent_scenes':1,'dart_independent_holdout_scenes':0,
                         'world_geometry_batch_status':state['status'],'world_geometry_completed_frames':completed,'planned_frames':11},
              'scattering_and_background_are_distinct':True,
              'land_coefficient_validation':grass['groups'],
              'sea_statistical_validation':{'records':14,'mean_RD_MAE_dB':sea_metrics,'type':'within-record prefix calibration to later segment; not zero-shot'},
              'dart_colored_background':dart['models']['surface_plus_colored_noise'],
              'maqu_measured_only_increment_m':delay['measured_increment_m'],'maqu_nominal_increment_m':delay['nominal_increment_m'],
              'output_integrity_checked_frames':integrity['checked_completed_frames'],
              'world_geometry_comparison':world,
              'unproven':['absolute RCS and receiver calibration','complete scene geometry/material reconstruction',
                          'environment occlusion and general multipath','high-altitude near-vertical centimeter-wave measured fidelity'],
              'precipitation':'MRR data and adapter acquired; forward rain/snow scattering is not claimed',
              'frontend_started':False,'old_projects_modified':False}
    (OUT/'evidence.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2))
    g=grass['groups'][0]
    world_result=f'当前状态：{state["status"]}，已完成 {completed}/11 帧。'
    world_boundary='完整评价尚未生成时，不发布部分帧的改善结论。'
    world_link=''
    if world is not None and state['status']=='complete':
        fixed=world['models']['expanded_colored_fixed'];refit=world['models']['expanded_colored_refitted']
        world_result+=f' 主指标：旧比例 {fixed["primary"]["mean_mae_db"]:.2f} dB；仅校准帧重估比例 {refit["primary"]["mean_mae_db"]:.2f} dB。'
        world_boundary='保留旧比例时略有退化；重新校准后改善。不能把全部改善归因于几何补全，也不是独立场景验证。'
        world_link='<p><a href="../../validation_diagnostics/2026-10-15/dart/batch_v2_world_geometry/扩展世界几何RD对照.html">扩展世界几何最终对照</a></p>'
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{{max-width:1250px;margin:30px auto;font:16px/1.7 sans-serif}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #bbb;padding:10px;text-align:left}}.note{{background:#fff1cc;padding:15px}}</style>
<h1>本轮扩展：功能、证据与边界</h1><p>生成时间（UTC）：{evidence['as_of_utc']}。这是现有证据总览，不是整体高保真认证。</p>
<table><tr><th>用户目标</th><th>已实现</th><th>当前验证证据</th><th>尚未证明</th></tr>
<tr><td>点目标扩展到面目标 RD</td><td>漫反射面元、标量物理光学面积积分；脉冲与 FMCW 复数回波；轨迹与姿态更新。</td><td>玛曲金属板静态响应；理想动态解析检查；DART 实测 RD 与仿真多帧比较。</td><td>玛曲不是连续动态 RD；DART 材料、遮挡、多径和完整场景仍不充分。</td></tr>
<tr><td>增加散射模型与数据类别</td><td>面回波模型、经验地表后向散射系数；海面统计背景；实验室面目标输入；降水谱数据适配。</td><td>地表角度/频段条件检查、海面多记录统计对照、独立 LiDAR 几何投影。</td><td>统计背景不等于物理散射求解；降水数据接入不等于雨雪电磁模型已实现。</td></tr>
<tr><td>背景更逼真并可验证</td><td>白高斯、有色高斯、相关高斯与 Gamma 纹理背景；校准与检查分开，噪声在正确处理阶段加入。</td><td>IPIX 14 条记录；DART 3 帧校准、8 帧开发检查、3 个噪声种子；原始线性功率保留。</td><td>目标结构不会因加噪声自动正确；相邻帧、种子不算独立场景。</td></tr></table>
<h2>数值结果必须按各自口径阅读</h2><table><tr><th>数据</th><th>结果</th><th>不能据此声称</th></tr>
<tr><td>玛曲草地</td><td>经验角度插值系数误差 {g['mean_sigma0_mae_db']:.2f} dB，常量基线 {g['constant_training_mean_mae_db']:.2f} dB；{g['comparisons']}/{g['total_comparisons']} 项有支持，{g['unsupported']} 项不支持。</td><td>这是后向散射系数比较，不是 RD 图 MAE，也不能外推到未覆盖的频段和角度。</td></tr>
<tr><td>IPIX 海面</td><td>白高斯平均 RD MAE {sea_metrics['white_gaussian']:.2f} dB；相关高斯 {sea_metrics['correlated_gaussian']:.2f} dB；Gamma 纹理 {sea_metrics['compound_gamma']:.2f} dB。</td><td>这是用每条记录前段估参、预测后段的条件统计检查；不是海浪三维几何孪生。</td></tr>
<tr><td>DART 旧几何＋有色背景</td><td>主指标 MAE {dart['models']['surface_plus_colored_noise']['primary']['mean_mae_db']:.2f} dB；几何区域 {dart['models']['surface_plus_colored_noise']['geometry']['mean_mae_db']:.2f} dB。</td><td>仅一个场景、已查看的八帧；几何区域误差仍明显，不能称为高保真通过。</td></tr>
<tr><td>DART 扩展世界几何</td><td>{world_result}</td><td>{world_boundary}</td></tr></table>
<p class="note">不同数据的 dB 指标对应不同物理量、归一化和任务，不能横向排名。厘米波高空近垂直场景仍以物理/数值自洽检查为主，毫米波室内结果不能直接证明其保真。</p>
<h2>报告入口</h2><p><a href="../../surface_extension/2026-10-15_refined/面目标扩展与验证.html">面模型与玛曲静态验证</a> · <a href="../../validation_diagnostics/2026-10-15/玛曲偏差诊断.html">玛曲距离偏差诊断</a> · <a href="../../background_extension/2026-10-15/背景与地表散射验证.html">海面与地表统计验证</a></p>
<p><a href="../../validation_diagnostics/2026-10-15/dart/batch_v1/DART面目标与背景批量报告.html">DART 冻结批次</a> · <a href="../../validation_diagnostics/2026-10-15/dart/colored_background_followup/距离相关高斯背景对照.html">有色高斯开发对照</a> · <a href="../../validation_diagnostics/2026-10-15/dart/geometry_crop_audit/几何覆盖与回波误差定位.html">几何裁剪诊断</a></p>
{world_link}<p><a href="evidence.json">结构化证据记录</a>。前端保持关闭，旧工程及原始数据不修改。</p></html>'''
    (OUT/'扩展证据总览.html').write_text(body)
    print(json.dumps({'report':str(OUT/'扩展证据总览.html'),'world_batch_completed':completed,'world_batch_status':state['status']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
