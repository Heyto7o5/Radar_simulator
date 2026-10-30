#!/usr/bin/env python3
"""Audit paired center-landmark experiments and write a compact figure/report."""
import argparse,base64,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from validate_center_landmark import feature


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--building',type=Path,required=True)
    p.add_argument('--terrain',type=Path,required=True)
    args=p.parse_args(); roots={'terrain':args.terrain,'center_building':args.building}
    protocols={k:json.loads((r/'protocol.json').read_text()) for k,r in roots.items()}
    assert protocols['terrain']['train']==protocols['center_building']['train']
    assert protocols['terrain']['test']==protocols['center_building']['test']
    summaries={k:json.loads((r/'summary.json').read_text()) for k,r in roots.items()}
    results={k:json.loads((r/'results.json').read_text()) for k,r in roots.items()}
    audit={}
    for name,root in roots.items():
        status={};count=0
        for path in root.glob('*/*/rda.npz'):
            with np.load(path) as f:
                a=f['magnitude'];assert a.ndim==4 and a.shape[0]==2 and np.isfinite(a).all() and a.max()>0
                assert set(f.files)=={'magnitude','range_m','velocity_mps','azimuth_deg','frame_times_s'}
            m=json.loads((path.parent/'summary.json').read_text())
            st=m['applicability']['status'];status[st]=status.get(st,0)+1;count+=1
        assert count==55
        assert len(results[name])==84
        audit[name]=dict(cases=count,frames=2*count,applicability=status)
    rng=np.random.default_rng(11);a=rng.uniform(.1,1,(2,4,4,5)).astype('float32')
    np.testing.assert_allclose(feature(a,'two_frame_rd'),feature(a[:,:,:,::-1],'two_frame_rd'),atol=1e-6)
    assert not np.allclose(feature(a,'two_frame_rda'),feature(a[:,:,:,::-1],'two_frame_rda'))
    audit['input_contract_checks']='RDA input contains no pose truth; angle reflection disappears from RD projection but not RDA'
    audit['paired_states_identical']=True
    (args.building/'audit.json').write_text(json.dumps(audit,indent=2))
    fig,axes=plt.subplots(1,2,figsize=(11,5),sharex=True,sharey=True)
    for ax,name in zip(axes,roots):
        rr=[r for r in results[name] if r['group']=='test' and r['mode']=='two_frame_rda']
        truth=np.array([r['true_ne_m'] for r in rr]);pred=np.array([r['predicted_ne_m'] for r in rr])
        for t,q in zip(truth,pred):ax.plot([t[1],q[1]],[t[0],q[0]],c='gray',alpha=.5)
        ax.scatter(truth[:,1],truth[:,0],c='#175c91',label='True platform position')
        ax.scatter(pred[:,1],pred[:,0],marker='x',c='#ce582a',label='Estimated from RDA')
        ax.scatter([0],[0],marker='+',s=150,c='black',label='Center reference')
        ax.set(title=name,xlabel='East (m)',ylabel='North (m)',aspect='equal');ax.grid(alpha=.2)
    fig.suptitle('Two-frame RDA, fresh off-grid states, fixed altitude and velocity')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,fontsize=9)
    fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(args.building/'position_estimates.png',dpi=160);plt.close(fig)
    mirror=json.loads((args.building/'mirror_control.json').read_text())
    mode_names={'one_frame_rda':'单帧 RDA','two_frame_rd':'双帧 RD 投影','two_frame_rda':'双帧 RDA'}
    rows=''
    for scene in roots:
        for mode in mode_names:
            m=summaries[scene]['test'][mode]
            rows+=f'<tr><td>{scene}</td><td>{mode_names[mode]}</td><td>{m["position_mae_m"]:.2f}</td><td>{m["radius_mae_m"]:.2f}</td><td>{m["signs_correct"]}/{m["n"]}</td><td>{m["bearing_mae_deg"]:.1f}°</td></tr>'
    stress=''
    for group in ['phase','grid5']:
        for scene in roots:
            m=summaries[scene][group]['two_frame_rda']
            stress+=f'<tr><td>{group}</td><td>{scene}</td><td>{m["n"]}</td><td>{m["position_mae_m"]:.2f}</td><td>{m["signs_correct"]}/{m["n"]}</td></tr>'
    image=base64.b64encode((args.building/'position_estimates.png').read_bytes()).decode()
    html=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>中心建筑定位可行性验证</title>
<style>body{{max-width:1050px;margin:32px auto;font:16px/1.7 system-ui;padding:0 18px}}table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ccc;padding:8px}}img{{max-width:100%}}</style>
<h1>中心建筑：距离与方向分开验证</h1>
<p>仅研究低空农业场景的离线定位：30m 固定高度、3m/s 固定北向速度、固定姿态，两帧间隔1s。
真实 DSM 与上轮相同，新增中心6×8m建筑、声明高度5m。中心建筑是定位参考，未将屋顶当作施肥落点。</p>
<p>25个四象限标定状态，16个新连续测试状态；冻结4邻居幅度匹配。标签分别包含有符号位置与平面距离。
输入仅为雷达图；场景和中心关系已通过离线标定提供。本轮不复用旧高速场景权重，不验证飞行控制。
RD投影指RDA功率在角度维求和，不能当作旧单通道RD模型的严格复现。</p>
<h2>镜像对照</h2><p>孤立中心反射点的左右等距离观测：RD投影相对差 {mirror['relative_rd_power_difference']:.3g}；
完整RDA相对差 {mirror['relative_rda_difference']:.3g}；反转角度轴后差 {mirror['reflected_angle_rda_difference']:.3g}。
该对照说明角度维在此观测几何下能区分左右，不证明任意场景方向均可辨。</p>
<h2>独立测试</h2><table><tr><th>场景</th><th>输入</th><th>二维位置平均误差 m</th><th>距离MAE m</th><th>两轴符号均正确</th><th>平均方位误差</th></tr>{rows}</table>
<img src="data:image/png;base64,{image}">
<h2>冻结标定后的变化检查</h2><table><tr><th>变化</th><th>场景</th><th>样本数</th><th>位置平均误差 m</th><th>两轴符号均正确</th></tr>{stress}</table>
<p>phase：另一个固定散射相位种子；grid5：保持原始地形曲面，将10m散射采样改为5m，伴随散射点与相位变化，不能解释成新增实测分辨率或纯积分收敛。
仅一个地理区域；没有接收机噪声、未知高度速度姿态或未见建筑布局检验。独立状态不等于独立地域。
位置误差下降不等于完整施肥纠偏验证，失败样本保留在results.json。</p>
<p>复现脚本：scripts/validate_center_landmark.py；配对控制加 --terrain-only。
共110组双帧序列、220帧；审计见audit.json，参数见protocol.json，源码见各运行目录source.zip。</p></html>'''
    (args.building/'report.html').write_text(html)
    print(json.dumps(dict(audit=audit,summaries=summaries,mirror=mirror)))


if __name__=='__main__':main()
