#!/usr/bin/env python3
"""Isolate sampled motion effects, without pretending image change proves fidelity."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radar_platform.dart_dataset import load_dart_frame
from radar_platform.dart_processing import published_to_closing_axis
from radar_platform.unified.schema import Simulation
from radar_platform.unified.scene import compile_scene
from radar_platform.unified.trajectory import TrajectoryEvaluator

ROOT=Path('/nas/datasets/hyt/datasets/radar_open/dart_lab')
OUT=Path(__file__).resolve().parents[1]/'docs/validation_diagnostics/2026-10-15/dart'

def main():
    request=Simulation.model_validate_json((OUT/'surface_sampled_motion_request.json').read_text())
    scene=compile_scene(request.scene)
    times=np.arange(request.waveform.loops)*request.array.loop_period_s
    positions,_,bases,_=TrajectoryEvaluator(request.platform.trajectory).at(times)
    delta=scene.positions_ned_m[:,None,:]-positions[None,:,:]
    distances=np.linalg.norm(delta,axis=-1)
    gates=np.sum(delta*bases[None,:,:,0],axis=-1)/distances>=np.cos(np.deg2rad(request.platform.beam_half_angle_deg))
    switched=np.flatnonzero(np.any(np.diff(gates.astype(int),axis=1)!=0,axis=1))
    arrays=[];adcs=[]
    for tag in ['surface_baseline','surface_sampled_motion']:
        with np.load(ROOT/'processed'/f'lab-1_{tag}.npz',allow_pickle=False) as f:
            arrays.append(np.mean(abs(f['rd_channels'].astype(np.complex128))**2,axis=(2,3)))
            adcs.append(f['adc']);ranges=f['range_m'];velocity=f['closing_velocity_mps']
    reference=float(arrays[0].max())
    db=[10*np.log10(np.maximum(p/reference,1e-6)) for p in arrays]
    diff=db[1]-db[0]
    measured=load_dart_frame(ROOT/'extracted/lab-1',0)
    real_v,real=published_to_closing_axis(measured.published_velocity_mps,measured.rd_power)
    real_db=10*np.log10(np.maximum(real/real.max(),1e-6))
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for ax,power,title in zip(axes.flat[:2],db,['SIMULATED before: constant velocity / attitude','SIMULATED after: sampled translation / attitude']):
        im=ax.pcolormesh(velocity,ranges,power,shading='auto',cmap='viridis',vmin=-60,vmax=0)
        ax.set_title(title);fig.colorbar(im,ax=ax,label='dB, SAME baseline simulation reference')
    ax=axes.flat[2];im=ax.pcolormesh(velocity,ranges,diff,shading='auto',cmap='RdBu_r',vmin=-20,vmax=20)
    ax.set_title('After minus before (simulation ablation ONLY)');fig.colorbar(im,ax=ax,label='dB difference (display clipped +/-20)')
    ax=axes.flat[3];im=ax.pcolormesh(real_v,measured.range_m,real_db,shading='auto',cmap='viridis',vmin=-60,vmax=0)
    ax.set_title('MEASURED structure reference; SEPARATE peak scale');fig.colorbar(im,ax=ax,label='dB relative to measured peak; not amplitude-matched')
    for ax in axes.flat:ax.set(xlabel='Closing velocity (m/s)',ylabel='Range (m)',xlim=(-.95,.95),ylim=(0,5.4))
    fig.savefig(OUT/'dart_motion_ablation.png',dpi=160);plt.close(fig)
    result={'simulation_change_mae_db':float(np.mean(abs(diff))),
            'metric_meaning':'average absolute dB change between two simulations, NOT error to real data',
            'shared_simulation_power_reference':reference,'floor_db':-60,
            'complex_adc_relative_change':float(np.linalg.norm(adcs[1]-adcs[0])/np.linalg.norm(adcs[0])),
            'real_data_mae':None,'real_fidelity_improvement_proven':False,
            'first_tx_hard_beam_switching_cells':len(switched),
            'first_tx_switching_cell_median_ranges_m':np.median(distances[switched],axis=1).tolist(),
            'controlled_variables':['same geometry','same radar configuration','same diffuse phases and seed','same material coefficients','same no-noise setting'],
            'changed_variable':'platform motion: fixed velocity and attitude -> published sampled trajectory with Hermite and SLERP',
            'scene_count':1,'development_frames':1,'heldout_scene_count':0}
    (OUT/'motion_ablation_metrics.json').write_text(json.dumps(result,indent=2))
    (OUT/'DART轨迹驱动对比.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{max-width:1200px;margin:30px auto;font:16px/1.7 sans-serif}img{width:100%}td,th{border:1px solid #bbb;padding:10px}table{border-collapse:collapse}</style>
<h1>新增通用能力：采样轨迹驱动的平台运动</h1><p>位置使用带速度约束的平滑插值，姿态使用旋转插值；每个 Tx 时隙更新阵元位置和速度，包括阵元随平台转动产生的速度。时间超出输入轨迹时直接报错，不自行延长轨迹。</p>
<img src="dart_motion_ablation.png"><p>上排两张仿真图使用同一个功率参考；左下是“新仿真减旧仿真”，不是实测误差。右下实测仅作结构参考，尚未进行共同幅度标定。</p>
<p style="background:#fff0c9;padding:15px">新图约 1.64 m 处出现横向细线。检查发现第一 Tx 有一个面片在 CPI 内跨越硬波束边界，其距离约 1.64 m，与细线位置一致。这提示硬开关照明带来的频谱扩散；尚需连续方向图对照才能确认贡献，不能把它解释为实测背景被复现。</p>
<table><tr><th>本轮验证</th><th>边界</th></tr><tr><td>匀速轨迹复现旧链路、加速度插值、转动阵元相位、跨帧连续性、时间覆盖检查。</td><td>使用的是作者平滑轨迹及其插值，不是独立运动捕捉真值；chirp 内仍采用一阶距离近似。</td></tr><tr><td>同一面几何、散射系数和随机种子，只改变平台运动模型。</td><td>只是一帧开发样例，不能从图像变化宣称保真提高；尚需多帧和独立保留数据。</td></tr><tr><td>保留每个收发通道的复数 ADC 与 RD。</td><td>角度处理采用 CPI 中点姿态，尚无阵列运动补偿；暂不允许与后距离压缩的统计背景接口混用，以免默认方向错误。</td></tr></table>
<p><a href="motion_ablation_metrics.json">指标及含义</a> · <a href="surface_sampled_motion_request.json">含轨迹的完整输入</a> · <a href="surface_sampled_motion_status.json">运行记录</a></p></html>''')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
