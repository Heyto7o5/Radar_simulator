#!/usr/bin/env python3
"""Audit the complete same-time scan experiment independently of the long main batch."""
import argparse,base64,importlib.util,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--campaign',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    spec=importlib.util.spec_from_file_location('stage2_evaluator',Path(__file__).with_name('evaluate_stage2_campaign.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data,_=module.load_campaign(a.campaign,6)
    a.output.mkdir(parents=True,exist_ok=False)
    reference=max(v[0].max() for v in data.values());rows=[]
    fig,axes=plt.subplots(3,3,figsize=(15,10),layout='constrained')
    for i,site in enumerate(['alpine','valley','foothill']):
        before,mb,rb=data[(site,'hold0_t20ms',11)];after,ma,ra=data[(site,'scan8_t20ms',11)]
        for field in ['waveform','platform','array','sequence','processing','visibility','scene']:
            if rb[field]!=ra[field]:raise ValueError('scan comparison changes '+field)
        if mb['frame_start_s']!=.02 or ma['frame_start_s']!=.02:raise ValueError('wrong simulation time')
        if mb['platform_position_ned_m']!=ma['platform_position_ned_m']:raise ValueError('different actual platform positions')
        schedule=ra['radiation']['transmit_aperture']['scan_schedule']
        if schedule!=[{'time_s':.02,'azimuth_deg':8.,'elevation_deg':0.}]:raise ValueError('unexpected scan schedule')
        if np.array_equal(before,after):raise ValueError('declared scan produced no change')
        b=module.db(before,reference);s=module.db(after,reference)
        row=dict(site=site,simulation_time_s=.02,platform_position_ned_m=ma['platform_position_ned_m'],
                 mean_absolute_db_change=float(abs(s-b).mean()),integrated_power_change_db=float(10*np.log10(after.sum()/before.sum())),
                 same_nonradiation_inputs=True,same_physical_axes=ma['axes']==mb['axes'])
        if not row['same_physical_axes']:raise ValueError('scan axes changed')
        rows.append(row)
        r=np.array(ma['axes']['range_m']);v=np.array(ma['axes']['velocity_mps'])
        extent=[r[0]-(r[1]-r[0])/2,r[-1]+(r[1]-r[0])/2,v[0]-(v[1]-v[0])/2,v[-1]+(v[1]-v[0])/2]
        for j,(value,title) in enumerate([(b,'held 0 deg, t=20 ms'),(s,'scan 8 deg, t=20 ms'),(s-b,'scan minus held')]):
            im=axes[i,j].imshow(value.T,origin='lower',aspect='auto',extent=extent,cmap='RdBu_r' if j==2 else 'viridis',
                                vmin=-30 if j==2 else -60,vmax=30 if j==2 else 0)
            axes[i,j].set(title=f'{site}: {title}',xlabel='Slant range (m)',ylabel='Apparent closing velocity (m/s)');fig.colorbar(im,ax=axes[i,j],label='dB')
    fig.suptitle('Real DSM + classified cells; synthetic same-time scanning experiment, not measured radar')
    fig.savefig(a.output/'scan_comparison.png',dpi=140);plt.close(fig)
    report={'cases':6,'real_sites':3,'seed':11,'time_s':.02,'reference_power':float(reference),'rows':rows,
        'meaning':'changes inside the stored RD window between two controlled simulations, not accuracy errors or total radiated power',
        'nrsc':'assumed common -20 dB, not measured materials','drive_power_normalization':'identical weights; only phase steering changes'}
    (a.output/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    image=base64.b64encode((a.output/'scan_comparison.png').read_bytes()).decode()
    page='<html lang="zh"><meta charset="utf-8"><title>真实地形上的同刻扫描对照</title><body style="max-width:1500px;margin:30px auto;font:16px sans-serif"><h1>同一时刻：保持指向与电子扫描</h1><p>六个运行全部完成。每个真实 DEM 窗口都在同一个 20 ms 时刻比较：相同平台位置、轨迹、波形、地物分类和相位种子，只改变发射阵元的指向相位。没有用不同时刻的平台位置制造扫描差异。</p><p>固定 seed 11 单帧，未做种子平均；六幅功率图共用一个参考，下限 −60 dB；差值色标裁剪为 ±30 dB。这里的差值不是实测误差。</p><img style="width:100%" src="data:image/png;base64,'+image+'"><p>真实输入只有 DSM 与公开分类。−20 dB NRCS 为假设控制值，阵元模型为理想模型，纵轴为单 PRF 表观速度，实际平台速度仍为 1200 m/s。不能据此声称实测地表或阵列标定完成。</p></body></html>'
    (a.output/'report.html').write_text(page);print(json.dumps(rows),flush=True)


if __name__=='__main__':main()
