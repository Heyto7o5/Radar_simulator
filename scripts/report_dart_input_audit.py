#!/usr/bin/env python3
"""Preview independent geometry and measured radar; not a simulation score."""
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/nas/datasets/hyt/datasets/radar_open/dart_lab')
OUT = Path(__file__).resolve().parents[1] / 'docs/validation_diagnostics/2026-10-15/dart'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    root = ROOT / 'extracted/lab-1'
    with np.load(root / 'radar_first_valid_subset.npz', allow_pickle=False) as data:
        amplitude = data['rad'].astype(np.float64)
        indices = data['aligned_indices']
        raw_indices = data['raw_indices']
    with np.load(root / 'map.npz', allow_pickle=False) as data:
        projection = np.any(data['grid'], axis=2)
        lower, upper = data['lower'], data['upper']
    with h5py.File(root / 'trajectory.h5') as trajectory:
        positions = trajectory['pos'][:]
        velocity = trajectory['vel'][indices]
        timestamps = trajectory['t'][indices]
    sensor = json.loads((root / 'sensor.json').read_text())
    ranges, speeds = [np.linspace(*sensor[key]) for key in ['r', 'd']]
    power = np.mean(amplitude**2, axis=-1)
    reference = float(power.max())
    db = 10 * np.log10(np.maximum(power / reference, 1e-6))
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    ax = axes.flat[0]
    ax.imshow(projection.T, origin='lower', extent=[lower[0], upper[0], lower[1], upper[1]],
              cmap='Greys', interpolation='nearest')
    ax.plot(positions[:, 0], positions[:, 1], color='royalblue', lw=.7, label='Radar trajectory')
    ax.scatter(positions[indices, 0], positions[indices, 1], c='red', s=30, label='Selected frames')
    ax.set(xlabel='World front x (m)', ylabel='World left y (m)',
           title='Independent LiDAR occupancy: height projection')
    ax.legend(fontsize=8)
    for i, ax in enumerate(axes.flat[1:]):
        image = ax.pcolormesh(speeds, ranges, db[i], shading='auto', cmap='viridis', vmin=-60, vmax=0)
        ax.axvline(np.linalg.norm(velocity[i]), color='white', ls=':', lw=.8)
        ax.axvline(-np.linalg.norm(velocity[i]), color='white', ls=':', lw=.8)
        ax.set(xlabel='Published Doppler velocity axis (m/s)', ylabel='Range (m)',
               title=f'Measured raw frame {raw_indices[i]} | speed {np.linalg.norm(velocity[i]):.3f} m/s')
        fig.colorbar(image, ax=ax, label='Relative power (dB; one shared reference)')
    fig.savefig(OUT / 'dart_inputs.png', dpi=160)
    plt.close(fig)
    summary = {'is_real': True, 'simulation_performed': False, 'mae': None,
               'raw_indices': raw_indices.tolist(), 'trajectory_indices': indices.tolist(),
               'timestamp_s': timestamps.tolist(), 'speed_mps': np.linalg.norm(velocity, axis=1).tolist(),
               'display': 'mean squared magnitude over 8 angle bins, then dB with one shared reference',
               'reference_power': reference, 'floor_db': -60,
               'independent_scenes': 1, 'adjacent_frames_are_not_independent_scenes': True,
               'map_is_radar_scattering_truth': False,
               'processing_caution': 'Hann windows, angle FFT, zero-Doppler magnitude subtraction, scale and float16 clipping in upstream code',
               'source': 'https://zenodo.org/records/10938617',
               'processing_source': 'https://github.com/thetianshuhuang/rover/blob/main/processing/process/radar.py'}
    (OUT / 'input_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (OUT / 'DART面目标输入核查.html').write_text('''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>DART 面目标输入核查</title><style>body{max-width:1100px;margin:32px auto;font:16px/1.7 sans-serif;color:#182633}img{width:100%}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:10px;text-align:left}.note{background:#fff1cf;padding:15px}</style>
<h1>面目标验证：先核对实测与几何输入</h1>
<p class="note">这不是仿真通过报告：下图三张 RD 均为实测。尚未生成对应仿真，也没有 MAE。三帧来自同一次采集，不代表三个独立场景。</p>
<img src="dart_inputs.png" alt="LiDAR 几何、雷达轨迹和三帧实测 RD">
<table><tr><th>项目</th><th>已获得的证据</th><th>尚不能声称</th></tr>
<tr><td>场景输入</td><td>lab-1：实验室内五个不同材料的箱体；独立 LiDAR 占据地图和雷达轨迹</td><td>占据网格不等于材料参数、法向或完整封闭表面</td></tr>
<tr><td>雷达观测</td><td>AWR1843BOOST；4,939 帧，128 距离 × 256 速度 × 8 角度幅度；已提取前三个有效轨迹对应帧</td><td>不是原始 ADC，也没有保留复相位；8 个角度格不等于 8 条原始接收通道</td></tr>
<tr><td>对齐</td><td>按官方 mask 再 valid 对齐；轨迹使用前–左–上坐标</td><td>不能将轨迹序号直接当原始雷达帧号；速度正负仍需核验</td></tr>
<tr><td>显示</td><td>八个角度幅度平方后平均，再转 dB；三帧共用参考与 −60 dB 下限。白虚线为 ±平台速率</td><td>不是绝对功率；虚线不是逐目标速度真值</td></tr>
<tr><td>下一步</td><td>从 LiDAR 提取表面，核对姿态、方向图、距离零点、处理窗及背景扣除，再做对应仿真</td><td>不能先拟合实测强度，再称为独立散射验证</td></tr></table>
<p>下载与完整性校验状态以数据目录 manifest.json 为准；核查图只使用按有效轨迹顺序选出的三帧，不按图像效果挑选。</p>
<p><a href="https://zenodo.org/records/10938617">官方数据与 CC BY 4.0 许可</a> · <a href="https://github.com/WiseLabCMU/dart/blob/main/tools/dataset.py">官方帧对应关系</a> · <a href="https://github.com/thetianshuhuang/rover/blob/main/processing/process/radar.py">官方雷达处理代码</a></p></html>''')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
