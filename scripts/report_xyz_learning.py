#!/usr/bin/env python3
"""Make a portable scientific report for the completed direct XYZ experiment."""
import argparse,base64,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--model',type=Path,required=True)
    args=p.parse_args();root=args.model
    if json.loads((root/'status.json').read_text())['status']!='complete':raise ValueError('training incomplete')
    m=json.loads((root/'metrics.json').read_text());training=json.loads((root/'training_summary.json').read_text())
    with np.load(root/'test_predictions.npz') as f:truth=f['truth_xyz_m'];pred=f['cnn_ensemble']
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for k,(ax,label) in enumerate(zip(axes,['x / North','y / East','z / Up'])):
        ax.scatter(truth[:,k],pred[:,k],s=15,alpha=.7,c='#176b89')
        lo=min(truth[:,k].min(),pred[:,k].min());hi=max(truth[:,k].max(),pred[:,k].max())
        ax.plot([lo,hi],[lo,hi],'--',c='black',lw=1)
        ax.set(xlabel=f'True {label} (m)',ylabel=f'Predicted {label} (m)',title=f'MAE {np.mean(abs(pred[:,k]-truth[:,k])):.2f} m')
        ax.grid(alpha=.2)
    fig.suptitle('Direct XYZ regression: 128 states in held-out 3D cells')
    fig.tight_layout();fig.savefig(root/'xyz_predictions.png',dpi=170);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    for row in training:
        history=json.loads((root/f'history_{row["seed"]}.json').read_text())
        ax.plot([h['epoch'] for h in history],[h['val_mean_3d_error_m'] for h in history],label=f'Seed {row["seed"]}')
    ax.set(xlabel='Epoch',ylabel='Validation mean 3D error (m)',yscale='log',title='Checkpoint selection uses validation only')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(root/'learning_curves.png',dpi=160);plt.close(fig)
    def image(name):return '<img src="data:image/png;base64,'+base64.b64encode((root/name).read_bytes()).decode()+'">'
    names={'train_mean':'训练集均值','knn':'近邻匹配','ridge':'线性回归','cnn_ensemble':'三模型平均','shuffled_input_control':'打乱输入对应关系'}
    table=''
    for name,s in m['test'].items():
        table+=f'<tr><td>{names.get(name,name)}</td>'+''.join(f'<td>{v:.3f}</td>' for v in s['axis_mae_m'])+f'<td>{s["mean_3d_error_m"]:.3f}</td><td>{s["p90_3d_error_m"]:.3f}</td></tr>'
    stress=''
    for split,label in [('grid5','散射网格10m→5m'),('building_height','建筑增高2m'),('noise10','噪声功率×10')]:
        s=m[split]['cnn_ensemble'];stress+=f'<tr><td>{label}</td><td>{s["cases"]}</td>'+''.join(f'<td>{v:.3f}</td>' for v in s['axis_mae_m'])+f'<td>{s["mean_3d_error_m"]:.3f}</td></tr>'
    result=m['test']['cnn_ensemble'];constant=m['test']['train_mean']
    html=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>多帧 RDA 直接反演 XYZ</title>
<style>body{{max-width:1080px;margin:32px auto;font:16px/1.7 system-ui;padding:0 20px;color:#19313b}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccd5db;padding:8px}}code{{background:#edf1f4}}</style>
<h1>深度学习：两帧 RDA 直接回归 x、y、z</h1>
<p>测试集平均三维位置误差 {result['mean_3d_error_m']:.3f}m，x/y/z 的 MAE 分别为
{result['axis_mae_m'][0]:.3f}/{result['axis_mae_m'][1]:.3f}/{result['axis_mae_m'][2]:.3f}m。
常数基线平均三维误差 {constant['mean_3d_error_m']:.3f}m。完整基线与失败结果见下表，不预设CNN一定最优。</p>
<h2>实验定义</h2><p>x=北向、y=东向、z=向上，相对真实 DSM 中心，标签对应首帧CPI起点。
中心建筑占地6×8m、声明高度5m；x覆盖±18m，y覆盖±12m，z独立覆盖22–42m。
平台姿态固定、以3m/s向北运动，两帧间隔1s，每帧由原平台重新生成ADC与RDA，没有时间插值。</p>
<p>384训练状态/192空间小区、64验证状态/32小区、128测试状态/64小区，整个三维小区不跨集合。
每个样本使用独立随机散射相位和接收机噪声种子，同一序列内相位保持一致。
噪声为声明的固定复ADC样本方差1e-5，不按每例信号强度重新缩放。</p>
<p>仍使用2Tx8Rx线阵、一个角度维，未偷偷增加俯仰观测。为覆盖变化高度，接收采样窗口扩展，保留200MHz采样带宽。
输入为两帧RDA幅度，经无标签的对数功率变换和固定网格重采样到2×64×16×24；三维卷积网络保留空间排列，直接输出三坐标。
固定场景关系通过训练学习，推断不读取DEM、位姿或建筑坐标。不是未知地图或未知姿态定位。</p>
<h2>独立测试结果（米）</h2><table><tr><th>方法</th><th>x MAE</th><th>y MAE</th><th>z MAE</th><th>平均三维误差</th><th>三维误差P90</th></tr>{table}</table>
{image('xyz_predictions.png')}{image('learning_curves.png')}
<h2>冻结模型的配对压力测试</h2><table><tr><th>变化</th><th>样本数</th><th>x MAE</th><th>y MAE</th><th>z MAE</th><th>平均三维误差</th></tr>{stress}</table>
<p>压力样本复用前8个测试位置，不是新增独立地理场景。散射网格变化伴随散射点与相位变化，不是纯积分收敛实验。</p>
<h2>证据边界与复现</h2><p>仅一个真实 DSM 窗口和固定建筑布局；原始地形约30m级，细分网格不增加真实地形细节。
材料散射与噪声参数是假设；未知速度、姿态、跨地域和真实雷达泛化尚未验证。输出为离线位置估计，不包含飞控、释放或落点控制。</p>
<p>生成：scripts/generate_xyz_learning_data.py；训练：scripts/train_xyz_learning.py；
独立推断：scripts/predict_xyz_learning.py。权重seed_*.pt、训练历史、逐例预测、协议和源码快照均在结果目录。
数据审计保存在数据目录audit.json。</p></html>'''
    (root/'report.html').write_text(html)
    print(json.dumps(dict(report=str(root/'report.html'),test=result)))


if __name__=='__main__':main()
