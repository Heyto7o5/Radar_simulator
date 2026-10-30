"""Supplementary input diagrams and descriptive error partition, no model changes."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from radar_platform.comparison import compare_power
from render_scene import configure_font

P=Path(__file__).resolve().parents[1]
OUT=P/'docs/weekly/2026-10-10/补充说明'
ROOT=Path('/nas/datasets/hyt/datasets/radar_open/alignment_campaign')

def load(p):return json.loads(p.read_text())

def main():
    OUT.mkdir(exist_ok=True);configure_font()
    cfg=load(P/'docs/weekly/2026-10-10/configs/motion_unambiguous_1ghz.json')
    p=np.array(cfg['platform']['position_ned_m']);v=np.array(cfg['platform']['velocity_ned_mps']);b=np.array(cfg['platform']['boresight_ned'])
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    ax=axs[0];ax.plot([-100,1500],[0,0],c='#785235');ax.scatter(p[0],-p[2],marker='v',s=90,c='#24698b',label='平台起点')
    scale=650/np.linalg.norm(v)
    ax.arrow(p[0],-p[2],v[0]*scale,-v[2]*scale,width=10,head_width=60,color='#24698b',length_includes_head=True)
    center=p+b*(-p[2]/b[2]);ax.plot([p[0],center[0]],[-p[2],0],ls='--',c='gray',label='波束中心（80°下视）')
    for a in [76,84]:ax.plot([0,5000/np.tan(np.radians(a))],[5000,0],ls=':',c='gray')
    for obj in cfg['scene']['objects']:
        n,e,d=obj['position_ned_m'];ax.scatter(n,-d,c='#c86632',s=35);ax.annotate(obj['id'],(n,-d),xytext=(0,8),textcoords='offset points',ha='center')
    ax.set(xlim=(-150,1500),ylim=(-250,5350),xlabel='North / m',ylabel='高度 / m',title='输入几何：平台 + 3 个静止点目标')
    ax.text(300,4450,'平台速率 1200 m/s\n1 GHz，本例不发生速度模糊',fontsize=9);ax.legend(fontsize=8,loc='center right');ax.grid(alpha=.15)
    ax=axs[1];ax.axis('off')
    rows=[];t=cfg['waveform']['slot_period_s']*(cfg['waveform']['loops']-1)/2+cfg['waveform']['samples']/cfg['waveform']['sample_rate_hz']/2
    lam=299792458/cfg['waveform']['carrier_hz'];prf=1/cfg['waveform']['slot_period_s']
    for o in cfg['scene']['objects']:
        d=np.array(o['position_ned_m'])-p-v*t;closing=float(v@d/np.linalg.norm(d));fd=2*closing/lam
        fold=((fd+prf/2)%prf-prf/2)*lam/2
        rows.append([o['id'],str(tuple(o['position_ned_m'])),f"{o['amplitude']:g}",f'{closing:.2f}',f'{fold:.2f}'])
    table=ax.table(cellText=rows,colLabels=['目标','初始 NED 坐标 / m','幅度','真实径向\nm/s','理论 RD\nm/s'],loc='center',cellLoc='center',colWidths=[.1,.4,.1,.2,.2]);table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,2.5)
    ax.set_title('输入目标与 CPI 中点速度（接近为正）')
    ax.text(.02,.12,'三个目标都不是车或建筑模型；本例没有 DEM、地面杂波。\n地面线只用于解释位置，不参与回波计算。\n侧视图横纵尺度不同；左右两条虚线为 ±4° 波束边界。',fontsize=9,transform=ax.transAxes)
    fig.suptitle('1 GHz / 1200 m/s 输入场景：径向速度位于 ±1498.96 m/s 的无模糊范围内\nPRF 20 kHz、脉宽 5 μs；表中模糊阶数为零，实际 FFT 峰值见结果表',fontsize=12)
    fig.savefig(P/'docs/weekly/2026-10-10/images/motion_input.png',dpi=160,bbox_inches='tight');plt.close(fig)

    cfg=load(P/'docs/weekly/2026-10-10/configs/ramp_MIMO_acquisition.json');a=cfg['array']
    fig,axes=plt.subplots(1,2,figsize=(12,4.1),layout='constrained')
    for key,y,marker,color in [('tx_local_m',1,'^','#cc7334'),('rx_local_m',0,'o','#286b92')]:
        xs=np.array(a[key])[:,1]*1000;axes[0].scatter(xs,np.full(len(xs),y),marker=marker,c=color,s=90)
        for j,x in enumerate(xs):axes[0].annotate(('Tx' if y else 'Rx')+str(j+1)+f'\n{x:.2f} mm',(x,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8)
    axes[0].set(ylim=(-.5,1.7),yticks=[0,1],yticklabels=['Rx 接收位置','Tx 发射位置'],xlabel='局部横向坐标 / mm',title='实际使用的标称位置（两行仅为示意）');axes[0].grid(axis='x',alpha=.2)
    for tx,offset in enumerate([0,60]):
        for start in [offset,offset+120,offset+240]:
            axes[1].broken_barh([(start,32)],(1-tx-.14,.28),facecolors=['#cc7334','#286b92'][tx])
    axes[1].set(xlim=(-5,355),ylim=(-.5,1.5),yticks=[1,0],yticklabels=['Tx1 发射','Tx2 发射'],xlabel='时间 / μs',title='TDM 时序：发射间隔 60 μs；同 Tx 间隔 120 μs')
    axes[1].set_xticks([0,60,120,180,240,300]);axes[1].grid(axis='x',alpha=.2)
    fig.suptitle('SISO = Tx1/Rx1；SIMO = Tx1/全部 Rx；MIMO = 两个 Tx/全部 Rx\n所有 Rx 在同一发射时隙同步接收。子集保留原来的 120 μs 慢时间采样；色块为当前 32 μs 采样/调频近似。',fontsize=11)
    fig.savefig(OUT/'收发位置与时序.png',dpi=160,bbox_inches='tight');plt.close(fig)

    items=load(ROOT/'20261005/ramp_v3/simulation_inputs.json');power=load(ROOT/'20261005/ramp_v3/power_alignment.json');rows=[]
    for item in items:
        if item['role']!='heldout_record':continue
        fid=item['frame_id'];stem=fid.split('/')[1]+'_'+Path(fid).stem
        with np.load(ROOT/'weekly_20261010'/(stem+'.npz')) as z:
            m,s,r=z['MIMO_real_rd'],z['MIMO_sim_rd'],z['range_m'];roi=np.zeros(m.shape,bool)
            for l in item['raw_labels']:
                ext=.5*(abs(l['x_m']/l['range_m'])*l['width_m']+abs(l['y_m']/l['range_m'])*l['length_m'])
                roi|=(abs(r-l['range_m'])<=1.5+ext)[:,None]
            diff=abs(compare_power(m,s,reference_power=power['reference_RD_power']).difference_db)
            weak=m<m.max()*.01
            rows.append({'frame_id':fid,'roi_pixel_fraction':float(roi.mean()),'roi_mae_db':float(diff[roi].mean()),
                'outside_roi_mae_db':float(diff[~roi].mean()),'outside_roi_error_fraction':float(diff[~roi].sum()/diff.sum()),
                'weak_pixel_fraction':float(weak.mean()),'weak_error_fraction':float(diff[weak].sum()/diff.sum()),
                'weak_mae_db':float(diff[weak].mean()),'strong_mae_db':float(diff[~weak].mean())})
    with (OUT/'背景误差拆分.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    result={'frames':len(rows),'independent_records':3,'mode':'MIMO','aggregation':'equal-weight frame mean, 60 frames per record',
        'weak_definition':'measured power < each measured frame maximum minus 20 dB; mask only, not power renormalization',
        'caution':'weak cells and outside-label cells are NOT independently identified thermal noise',
        'mean':{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='frame_id'}}
    (OUT/'背景误差汇总.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
