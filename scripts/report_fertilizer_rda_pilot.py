#!/usr/bin/env python3
"""Render inspectable figures and a standalone HTML report from saved results."""
import argparse
import base64
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    root = args.run
    cfg = read(root/'buildings_config.json')
    t = cfg['scene']['terrain']
    fig, ax = plt.subplots(figsize=(7, 6))
    contour = ax.contourf(t['east_m'], t['north_m'], np.array(t['elevation_up_m']).T, levels=15, cmap='terrain')
    fig.colorbar(contour, ax=ax, label='Elevation relative to target (m)')
    for obj in cfg['scene']['objects']:
        n, e = obj['position_ned_m'][:2]
        sn, se = obj['size_m'][:2]
        ax.add_patch(Rectangle((e-se/2, n-sn/2), se, sn, fc='#d85735', ec='black'))
        ax.text(e, n+sn/2+2, obj['id'], fontsize=8, ha='center')
    ax.scatter([0], [0], marker='+', c='black', s=150, linewidths=2, label='Target = DEM center')
    ax.set(xlabel='East (m)', ylabel='North (m)', title='Real DSM + synthetic building landmarks', aspect='equal')
    ax.legend(loc='lower right')
    fig.tight_layout(); fig.savefig(root/'scene.png', dpi=160); plt.close(fig)
    if not (root/'results.json').exists():
        return
    rows = read(root/'results.json')
    summary = read(root/'summary.json')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, name in zip(axes, ['terrain', 'buildings']):
        rr = [r for r in rows if r['condition']==name]
        for r in rr:
            b, a = r['before_ne_m'], r['after_ne_m']
            ax.plot([b[1], a[1]], [b[0], a[0]], color='#9aa5b1', alpha=.6)
        ax.scatter([r['before_ne_m'][1] for r in rr], [r['before_ne_m'][0] for r in rr], c='#d85735', label='Before')
        ax.scatter([r['after_ne_m'][1] for r in rr], [r['after_ne_m'][0] for r in rr], c='#177e89', label='After ideal adjustment')
        ax.scatter([0], [0], marker='+', s=180, c='black', label='Target')
        ax.set(title=name, xlabel='East landing error (m)', ylabel='North landing error (m)', aspect='equal')
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Held-out states; same calibrated scene; no wind or drag')
    fig.tight_layout(); fig.savefig(root/'landing_errors.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, name in enumerate(['terrain', 'buildings']):
        with np.load(root/name/'test_000'/'rda.npz') as f:
            cube = f['magnitude'][0]**2
            rd = cube.sum(axis=2); ra = cube.sum(axis=1)
            for col, (data, x, label) in enumerate([(rd, f['velocity_mps'], 'Closing speed (m/s)'),
                                                    (ra, f['azimuth_deg'], 'Azimuth (deg)')]):
                db = 10*np.log10(np.maximum(data/data.max(), 1e-6))
                im = axes[row, col].pcolormesh(x, f['range_m'], db, shading='auto', vmin=-40, vmax=0, cmap='viridis')
                axes[row, col].set(title=f'{name}: '+('RD power projection' if col==0 else 'RA power projection'),
                                   xlabel=label, ylabel='Range (m)')
                fig.colorbar(im, ax=axes[row, col], label='Relative dB')
    fig.suptitle('Display projections only; predictor uses the full two-frame RDA magnitude')
    fig.tight_layout(); fig.savefig(root/'rda_projections.png', dpi=160); plt.close(fig)

    def embed(filename):
        data = base64.b64encode((root/filename).read_bytes()).decode()
        return f'<img src="data:image/png;base64,{data}" alt="{filename}">'

    table = ''.join(f'<tr><td>{name}</td><td>{r["before_mean_m"]:.3f}</td><td>{r["after_mean_m"]:.3f}</td>'
                    f'<td>{r["after_p90_m"]:.3f}</td><td>{r["improved"]}/{r["cases"]}</td></tr>'
                    for name, r in summary.items())
    stress = read(root/'stress'/'summary.json') if (root/'stress'/'summary.json').exists() else None
    stress_table = '<p>压力检查尚未完成。</p>' if stress is None else '<table><tr><th>条件</th><th>纠偏后均值 m</th><th>改善例数</th></tr>'+''.join(
        f'<tr><td>{k}</td><td>{v["mean_after_m"]:.3f}</td><td>{v["improved"]}/{v["cases"]}</td></tr>' for k, v in stress.items())+'</table>'
    html = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>RDA 施肥落点纠偏受控实验</title>
<style>body{{font:16px/1.7 system-ui;max-width:1080px;margin:36px auto;padding:0 20px;color:#18313d}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd6dd;padding:9px;text-align:left}}code{{background:#edf2f5}}</style>
<h1>真实 DSM + 建筑物：仅用多帧 RDA 的落点纠偏实验</h1>
<p>目标为 DEM 中心。固定场景离线标定；预测函数仅接收两帧数值 RDA 幅度，输出位置与水平速度估计。
建筑布局与目标的关系隐含于标定样本，不是未知地图下自动发现目标。</p>
<p>真实地形来自已有 Copernicus DSM 配置，选取首个 4×4 主导类别均为农田的分类窗口；分类并非现场核验。
30m 原始几何细分至 10m 三角面供数值散射，不增加真实地形分辨率。建筑物及地表反射系数是假设，未标定真实材料。</p>
{embed('scene.png')}
<p>30m 固定高度、水平近北向飞行；FMCW 约 10GHz、200MHz 采样带宽、2Tx8Rx 线阵、32 loops；
两帧间隔 1s。RDA 的 A 为方位角，未额外向预测器提供俯仰维。81 个离线标定状态，12 个独立连续测试状态，
均来自同一场景。邻居数固定为4，不根据测试结果挑选。没有使用接收机噪声。</p>
{embed('rda_projections.png')}
<h2>落点评价</h2><p>固定 0.2s 出料延迟、无风、无阻力、释放初始垂直速度为零，水平速度继承无人机。
真值使用抛体轨迹与原场景地形的首次相交。纠偏动作是释放前理想的水平位置调整，不包含飞控执行、延迟或转弯动态。</p>
<table><tr><th>场景</th><th>纠偏前均值 m</th><th>纠偏后均值 m</th><th>纠偏后 P90 m</th><th>改善例数</th></tr>{table}</table>
{embed('landing_errors.png')}<h2>冻结标定后的压力检查</h2>{stress_table}
<h2>证据边界</h2><p>结果仅评价当前固定场景、固定姿态高度与参数下的仿真算法。不是实际施肥精度、跨地域泛化、颗粒雷达可见性或完整飞控闭环验证。
建筑物是否有益应以配对结果判断。主测试与压力测试中的失败保留，不用于删样本或重新选参数。</p>
<p>复现入口：scripts/run_fertilizer_rda_pilot.py；压力检查：scripts/stress_fertilizer_rda_pilot.py。
同目录保留 protocol.json、results.json、summary.json、source_snapshot.zip、全部 RDA 和代表案例 ADC。</p></html>'''
    (root/'report.html').write_text(html)


if __name__ == '__main__':
    main()
