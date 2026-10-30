#!/usr/bin/env python3
"""Controlled hard-edge/taper experiment; not a measured antenna calibration."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radar_platform.unified.antenna import two_way_voltage_gain

ROOT=Path('/nas/datasets/hyt/datasets/radar_open/dart_lab')
OUT=Path(__file__).resolve().parents[1]/'docs/validation_diagnostics/2026-10-15/dart'

def main():
    powers=[]
    for suffix in ['', '_smooth_beam']:
        with np.load(ROOT/'processed'/f'lab-1_surface_sampled_motion{suffix}.npz',allow_pickle=False) as f:
            powers.append(np.mean(abs(f['rd_channels'].astype(np.complex128))**2,axis=(2,3)))
            ranges=f['range_m'];velocity=f['closing_velocity_mps']
    ref=float(powers[0].max());db=[10*np.log10(np.maximum(p/ref,1e-6)) for p in powers]
    region=(ranges>=1.55)&(ranges<=1.75);far=abs(velocity)>.5
    energy=[float(p[np.ix_(region,far)].sum()) for p in powers]
    total=[float(p[region].sum()) for p in powers]
    ratio=[a/b for a,b in zip(energy,total)]
    result={'experiment':'same sampled trajectory and geometry, hard cone versus 5 degree inner cosine edge',
            'scene_count':1,'frames':1,'target_region_m':[1.55,1.75],'off_band_abs_velocity_gt_mps':.5,
            'region_selection':'fixed from previous hard-edge switching diagnosis at 1.641 m, before smooth run; diagnostic not held-out validation',
            'off_band_power':[energy[0],energy[1]],'region_total_power':total,'off_band_fraction':ratio,
            'off_band_power_change_db':float(10*np.log10(energy[1]/energy[0])),
            'off_band_fraction_change_db':float(10*np.log10(ratio[1]/ratio[0])),
            'meaning':'negative change means less simulated energy far from main velocities, not improved measured-data MAE',
            'measured_antenna_pattern':False,'real_data_mae':None,'real_fidelity_improvement_proven':False,
            'half_angle_deg':65,'taper_deg':5,'shared_power_reference':ref,'floor_db':-60}
    (OUT/'beam_ablation_metrics.json').write_text(json.dumps(result,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    angles=np.linspace(55,70,1000)
    for model in ['hard_cone','cosine_edge']:
        gain=two_way_voltage_gain(np.cos(np.deg2rad(angles)),65,model,5)
        axes[0,0].plot(angles,gain**2,label=model)
    axes[0,0].set(xlabel='Off-boresight angle (deg)',ylabel='Relative received power',title='Same cutoff: 65 deg; only inner edge changes')
    axes[0,0].legend()
    for ax,power,title in [(axes[0,1],db[0],'Before: hard beam edge'),(axes[1,0],db[1],'After: smooth inner edge')]:
        im=ax.pcolormesh(velocity,ranges,power,shading='auto',vmin=-60,vmax=0,cmap='viridis')
        ax.set(xlabel='Closing velocity (m/s)',ylabel='Range (m)',title=title)
        fig.colorbar(im,ax=ax,label='dB: SAME hard-edge simulation reference')
    ax=axes[1,1];im=ax.pcolormesh(velocity,ranges,db[1]-db[0],shading='auto',vmin=-20,vmax=20,cmap='RdBu_r')
    ax.set(xlabel='Closing velocity (m/s)',ylabel='Range (m)',title='Smooth minus hard: simulation-only difference')
    fig.colorbar(im,ax=ax,label='dB change (display clipped +/-20)')
    fig.savefig(OUT/'dart_beam_ablation.png',dpi=160);plt.close(fig)
    text='''<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{max-width:1200px;margin:30px auto;font:16px/1.7 sans-serif}img{width:100%}td,th{border:1px solid #bbb;padding:10px}table{border-collapse:collapse}</style>
<h1>硬波束边界会不会制造额外频谱？</h1><p>几何、采样轨迹、散射系数和随机种子全部不变；半角仍为 65°，仅将最后 5° 的开关边界替换为连续衰减。此宽度为预先指定的数值实验参数，未拟合实测。</p>
<img src="dart_beam_ablation.png"><p>四图依次为：波束响应、原仿真、平滑边缘仿真、仿真间差值。两张 RD 共用同一功率参考，没有各自重新归一化。</p>
<table><tr><th>检查区域</th><th>指标</th><th>变化</th></tr><tr><td>1.55～1.75 m，|速度| &gt; 0.5 m/s</td><td>远离主要速度的总能量</td><td>ENERGY_CHANGE dB</td></tr><tr><td>同一区域</td><td>该能量占此距离带总能量的比例</td><td>FRACTION_CHANGE dB</td></tr></table>
<p>负数表示数值实验中的远端频谱减弱。第二项避免仅因整体信号减弱就误认为泄漏比例改善。这不是实测 MAE，也不证明天线方向图正确；未补上材料、遮挡、多径或真实噪声。</p>
<p><a href="beam_ablation_metrics.json">完整指标与预先固定的区域</a> · <a href="surface_sampled_motion_smooth_beam_request.json">完整输入</a></p></html>'''
    text=text.replace('ENERGY_CHANGE',f"{result['off_band_power_change_db']:.2f}").replace('FRACTION_CHANGE',f"{result['off_band_fraction_change_db']:.2f}")
    (OUT/'DART波束边缘对照.html').write_text(text)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
