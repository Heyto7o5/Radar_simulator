#!/usr/bin/env python3
"""Display completed baseline without inventing an aligned real-data score."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/nas/datasets/hyt/datasets/radar_open/dart_lab')
OUT=Path(__file__).resolve().parents[1]/'docs/validation_diagnostics/2026-10-15/dart'

def main():
    status=json.loads((OUT/'surface_baseline_status.json').read_text())
    with np.load(ROOT/'processed/lab-1_surface_baseline.npz',allow_pickle=False) as f:
        rd=f['rd_channels'];ranges=f['range_m'];velocity=f['closing_velocity_mps']
    power=np.mean(abs(rd.astype(np.complex128))**2,axis=(2,3))
    # This figure is only a single simulated image; its normalization is not
    # reused as an amplitude calibration against real data.
    db=10*np.log10(np.maximum(power/power.max(),1e-6))
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained')
    im=ax.pcolormesh(velocity,ranges,db,shading='auto',vmin=-60,vmax=0,cmap='viridis')
    ax.set(xlabel='Closing velocity (m/s, approaching positive)',ylabel='Range (m)',
           title='SIMULATED: LiDAR-derived diffuse surfaces, ideal 2Tx4Rx')
    fig.colorbar(im,ax=ax,label='Relative simulated power (dB, own peak for display only)')
    fig.savefig(OUT/'dart_simulated_surface_rd.png',dpi=160);plt.close(fig)
    (OUT/'DART面目标仿真基线.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><style>body{max-width:1050px;margin:30px auto;font:16px/1.7 sans-serif}img{max-width:100%}.note{background:#fff1cf;padding:15px}</style>
<h1>面目标回波链路：首帧仿真基线</h1><p class="note">本图是我们的仿真，不是实测，也不是保真通过结果。仅用于确认公开几何可以进入统一回波链路；尚未输出实测差值或 MAE。</p>
<img src="dart_simulated_surface_rd.png"><p>输入：独立 LiDAR 地图拟合得到的部分表面、公开 FMCW 配置候选、第一有效帧的平台位置和速度。使用 2Tx4Rx 理想阵列，按 chirp 更新相对距离，输出复数 ADC 和各通道 RD。</p>
<p>当前假设：25 cm 面片离散、统一相对漫反射系数、固定随机相位、双面响应、CPI 内固定姿态及匀速。尚无材料标定、遮挡、环境多径和背景噪声。阵元坐标、速度符号与实测零速度背景处理尚未完全对齐。</p>
<p>本图按自身峰值归一化仅为查看结构，不能用于证明与实测幅度一致。后续比较须统一速度方向、距离零点、处理链和独立校准参考。</p>
<p><a href="surface_baseline_request.json">完整仿真输入</a> · <a href="surface_baseline_status.json">结果与限制</a> · <a href="DART面目标输入核查.html">实测与几何输入核查（生成后可用）</a></p></html>''')
    print('saved simulated RD figure; no fidelity metric claimed',flush=True)

if __name__=='__main__':main()
