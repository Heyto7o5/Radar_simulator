#!/usr/bin/env python3
"""2026-10-10 evidence-first weekly report. Never modifies the simulator.

RAMP: all 180 existing held-out frames, three explicitly nested channel subsets.
Motion: Galilean common-velocity invariance, NOT a relativistic/real-data test.
Terrain: existing local DEM, geometric cross-check, NOT measured radar fidelity.
"""
import argparse
import base64
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import shutil
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
from scipy.io import loadmat

from radar_platform.comparison import compare_power, power_db
from radar_platform.models import SimulationRequest
from radar_platform.ramp_alignment import read_ramp_adc
from radar_platform.ramp_comparison import _class_rcs_scale
from radar_platform.unified.adapters import from_pulsed_request
from radar_platform.unified.engine import simulate_frame
from radar_platform.unified.geometry import build_layout, sensor_basis, scan_directions
from radar_platform.unified.processing import process_adc
from radar_platform.unified.schema import Simulation, Acquisition
from radar_platform.unified.scene import compile_scene
from render_scene import configure_font
from verify_unified_ramp import acquisition
from validate_unified import case_request

PROJECT=Path(__file__).resolve().parents[1]
ROOT=Path('/nas/datasets/hyt/datasets/radar_open')
DATA=ROOT/'ramp_adc_77ghz/alignment_subset_20261005'
CAMPAIGN=ROOT/'alignment_campaign/20261005/ramp_v3'
OUT=PROJECT/'docs/weekly/2026-10-10'
CACHE=ROOT/'alignment_campaign/weekly_20261010'
DEM=Path('/nas/datasets/hyt/datasets/DEM/0715_Terrains_500m_0211')
MODES={'SISO':(1,1),'SIMO':(1,4),'MIMO':(2,4)}

def load(p): return json.loads(p.read_text())
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def csvout(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def savefig(name,fig):
    fig.savefig(OUT/'images'/name,dpi=155,bbox_inches='tight');plt.close(fig)
def heat(ax,a,x,y,title,*,diff=False,lo=-60,hi=10,ylabel='接近速度 / (m/s)'):
    im=ax.pcolormesh(x,y,a.T,shading='auto',cmap='RdBu_r' if diff else 'turbo',
                     vmin=-40 if diff else lo,vmax=40 if diff else hi,rasterized=True)
    ax.set(xlabel='距离 / m',ylabel=ylabel,title=title)
    return im
def triple(name,m,s,x,y,ref,title,*,ylabel='接近速度 / (m/s)',ra=False,labels=None):
    c=compare_power(m,s,reference_power=ref)
    fig,axes=plt.subplots(1,3,figsize=(14,4.1),layout='constrained')
    for k,(ax,a,t) in enumerate(zip(axes,[c.measured_db,c.simulated_db,c.difference_db],
            ['实测 RA' if ra else '实测 RD','仿真 RA' if ra else '仿真 RD','差值：仿真 − 实测'])):
        im=heat(ax,a,x,y,t,diff=k==2,ylabel=ylabel)
        if labels and k<2:
            ax.scatter([l['range_m'] for l in labels],[np.degrees(np.arctan2(l['x_m'],l['y_m'])) for l in labels],
                       marker='x',c='white',s=55,lw=1.5,label='标注中心（不是散射真值）')
            ax.legend(fontsize=7,loc='upper right')
        fig.colorbar(im,ax=ax,label='共同参考 / dB' if k<2 else '差值 / dB',shrink=.8,extend='both')
    fig.suptitle(title+f"\n全图 MAE = {c.metrics['mae_db']:.2f} dB；同一参考，统一 −60 dB 下限",fontsize=11)
    savefig(name,fig)

def subconfig(full,mode):
    d=full.model_dump();nt,nr=MODES[mode]
    d['array'].update(num_tx=nt,num_rx=nr,tx_local_m=d['array']['tx_local_m'][:nt],
        rx_local_m=d['array']['rx_local_m'][:nr],tx_offsets_s=d['array']['tx_offsets_s'][:nt])
    # Preserve the original 120 us same-Tx interval, even after selecting Tx1 only.
    if mode=='SISO':d['processing']['azimuth_deg']=[0.]
    return Acquisition.model_validate(d)

def camera_and_scene(item,name,request):
    member=item['frame_id'].replace('radar_raw_frame','images_0').replace('.mat','.jpg')
    archive=ROOT/'ramp_adc_77ghz/raw/Automotive.zip'
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),layout='constrained')
    with zipfile.ZipFile(archive) as z:
        candidates=[n for n in z.namelist() if str(Path(n).parent)==str(Path(member).parent)
                    and Path(n).suffix.lower() in ('.jpg','.jpeg','.png')
                    and Path(n).stem.isdigit() and int(Path(n).stem)==int(Path(member).stem)]
        if len(candidates)>1:raise ValueError('ambiguous camera frame number')
        if candidates:
            member=candidates[0]
            payload=z.read(member);axes[0].imshow(Image.open(io.BytesIO(payload)));axes[0].axis('off')
            camera_sha=hashlib.sha256(payload).hexdigest()
        else:
            axes[0].text(.5,.5,'配套相机帧缺失\n不以其他图片替代',ha='center');axes[0].axis('off');camera_sha=None
    axes[0].set_title('真实场景：官方同名相机帧')
    for l in item['raw_labels']:
        axes[1].add_patch(Rectangle((l['x_m']-l['width_m']/2,l['y_m']-l['length_m']/2),
              l['width_m'],l['length_m'],fill=False,ec='#276b91',lw=1.5))
        axes[1].annotate(str(l.get('source_uid',l['uid'])).split('@')[0],(l['x_m'],l['y_m']),fontsize=7)
    axes[1].set_title('真实输入：公开俯视标签框')
    for obj,l in zip(request.scene.objects,item['smoothed_labels']):
        n,e,_=obj.position_ned_m;vn,ve,_=obj.velocity_ned_mps
        axes[2].scatter(e,n,s=35,c='#c96428')
        axes[2].arrow(e,n,ve,vn,width=.03,head_width=.25,color='#c96428',length_includes_head=True)
        if l.get('radial_velocity_mps') is None:axes[2].annotate('速度未知→0占位',(e,n),fontsize=7)
    axes[2].set_title('仿真场景：中心点 + 标签估计速度')
    for ax in axes[1:]:
        ax.scatter(0,0,marker='^',c='black');ax.set(xlim=(-18,18),ylim=(-1,27),xlabel='横向 / m',ylabel='前向 / m')
        ax.set_aspect('equal');ax.grid(alpha=.15)
    fig.suptitle(item['frame_id']+'\n照片不是三维模型；目标中心也不等于真实主散射点',fontsize=10)
    savefig(name,fig)
    return {'member':member,'camera_sha256':camera_sha,'archive':str(archive)}

def ramp(run=True):
    inputs=load(CAMPAIGN/'simulation_inputs.json')
    items=sorted([x for x in inputs if x['role']=='heldout_record'],key=lambda x:x['frame_id'])
    power=load(CAMPAIGN/'power_alignment.json');ref=power['reference_RD_power'];gain=power['gains']['physical_smooth_roi_gain']
    selected=[]
    for seq in sorted({i['frame_id'].split('/')[1] for i in items}):
        group=[i for i in items if i['frame_id'].split('/')[1]==seq];selected.append(group[len(group)//2]['frame_id'])
    protocol={'frames':[x['frame_id'] for x in items],'independent_records':3,'illustration_frames_before_scoring':selected,
        'extra_example':'worst MIMO RD MAE after scoring, explicitly a failure example',
        'channel_subsets':MODES,'same_tx_period_s':120e-6,'independent_datasets':1,
        'calibration':str(CAMPAIGN/'power_alignment.json'),'calibration_sha256':sha(CAMPAIGN/'power_alignment.json'),
        'power_gain_frozen':gain,'reference_power_frozen':ref,'per_frame_fit':False,
        'source_input_sha256':sha(CAMPAIGN/'simulation_inputs.json'),
        'scope':'real ADC versus one-point-per-label simulation; not complete cars or independently measured motion',
        'shared_reference_for_RD_and_RA':True,'floor_db':-60,'metric':'dB MAE, no shift, no per-image peak normalization'}
    write(OUT/'metrics/ramp_protocol.json',protocol)
    cfg=acquisition();cfg.platform.beam_half_angle_deg=89.
    configs={m:subconfig(cfg,m) for m in MODES}
    for m,c in configs.items():write(OUT/'configs'/f'ramp_{m}_acquisition.json',c.model_dump(mode='json'))
    rows=[]
    for idx,item in enumerate(items):
        fid=item['frame_id']; stem=fid.split('/')[1]+'_'+Path(fid).stem
        cache=CACHE/(stem+'.npz')
        objs=[{'kind':'point','id':str(l['uid']),'position_ned_m':[l['y_m'],l['x_m'],0.],
            'velocity_ned_mps':l['velocity_ned_mps'],'amplitude':_class_rcs_scale(int(l['class_id']))*np.sqrt(gain)} for l in item['smoothed_labels']]
        req=Simulation.model_validate({**cfg.model_dump(),'scene':{'objects':objs,'amplitude_reference_range_m':10.},'sequence':{'frame_count':1}})
        arrays={}
        if run:
            adc=read_ramp_adc(DATA/fid);sim=simulate_frame(req)
            for mode,c in configs.items():
                nt,nr=MODES[mode];layout=build_layout(c.array,c.waveform,c.platform)
                real=process_adc(adc[:,:,:nr,:nt],c,layout)
                synth=sim.products if mode=='MIMO' else process_adc(sim.adc[:,:,:nr,:nt],c,layout)
                arrays.update({mode+'_real_rd':real.rd_power,mode+'_sim_rd':synth.rd_power,
                    mode+'_real_ra':real.ra_power[:,0,:],mode+'_sim_ra':synth.ra_power[:,0,:],mode+'_angles':real.azimuth_deg})
            arrays.update(range_m=real.range_m,velocity_mps=real.velocity_mps)
            np.savez_compressed(cache,**arrays)
        else:
            with np.load(cache) as z:arrays={k:z[k] for k in z.files}
        r=arrays['range_m'];roi=np.zeros((len(r),len(arrays['velocity_mps'])),bool)
        for l in item['raw_labels']:
            extent=.5*(abs(l['x_m']/l['range_m'])*l['width_m']+abs(l['y_m']/l['range_m'])*l['length_m'])
            roi|=(abs(r-l['range_m'])<=1.5+extent)[:,None]
        for mode in MODES:
            c=compare_power(arrays[mode+'_real_rd'],arrays[mode+'_sim_rd'],reference_power=ref,target_mask=roi)
            ra=compare_power(arrays[mode+'_real_ra'],arrays[mode+'_sim_ra'],reference_power=ref)
            rows.append({'frame_id':fid,'sequence':fid.split('/')[1],'mode':mode,'mae_db':c.metrics['mae_db'],
                'target_mae_db':c.metrics['target_mae_db'],'ra_mae_db':ra.metrics['mae_db'] if mode!='SISO' else None,
                'target_count':len(objs),'unknown_velocity_count':sum(l.get('radial_velocity_mps') is None for l in item['smoothed_labels']),
                'adc_sha256':sha(DATA/fid),'joint_floor_fraction':c.metrics['joint_floor_fraction']})
        if fid in selected:
            example=selected.index(fid)+1
            write(OUT/'configs'/f'ramp_example_{example}.json',req.model_dump(mode='json'))
            cam=camera_and_scene(item,f'ramp_scene_{example}.png',req)
            write(OUT/'metrics'/f'ramp_example_{example}_source.json',{'frame_id':fid,**cam,'raw_labels':item['raw_labels'],'simulation_labels':item['smoothed_labels']})
            for mode in MODES:
                triple(f'ramp_{mode}_{example}.png',arrays[mode+'_real_rd'],arrays[mode+'_sim_rd'],r,arrays['velocity_mps'],ref,
                    f'{mode} · 样例 {example} · '+fid.split('/')[1]+' / '+Path(fid).stem)
            for mode in ['SIMO','MIMO']:
                triple(f'ra_{mode}_{example}.png',arrays[mode+'_real_ra'],arrays[mode+'_sim_ra'],r,arrays[mode+'_angles'],ref,
                    f'{mode} · 样例 {example} · Doppler 线性功率平均；白叉为标签中心',ra=True,ylabel='方位角 / °',labels=item['raw_labels'])
        if idx%15==0:print('RAMP',idx+1,'/',len(items),flush=True)
    csvout(OUT/'metrics/ramp_frames.csv',rows)
    summary=[]
    for mode in MODES:
        rr=[r for r in rows if r['mode']==mode];v=np.array([r['mae_db'] for r in rr])
        summary.append({'mode':mode,'frames':len(v),'records':len({r['sequence'] for r in rr}),
            'mean':float(v.mean()),'median':float(np.median(v)),'p90':float(np.percentile(v,90)),
            'target_mean':float(np.mean([r['target_mae_db'] for r in rr if r['target_mae_db'] is not None]))})
    worst=max([r for r in rows if r['mode']=='MIMO'],key=lambda r:r['mae_db'])
    stem=worst['sequence']+'_'+Path(worst['frame_id']).stem
    with np.load(CACHE/(stem+'.npz')) as z:
        triple('ramp_worst.png',z['MIMO_real_rd'],z['MIMO_sim_rd'],z['range_m'],z['velocity_mps'],ref,
               '最差案例（按全部 180 帧 MIMO RD MAE 选择） · '+stem)
    write(OUT/'metrics/ramp_summary.json',{'groups':summary,'worst':worst})
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for j,mode in enumerate(MODES):
        vals=[r['mae_db'] for r in rows if r['mode']==mode]
        axes[0].scatter(np.full(len(vals),j)+np.linspace(-.16,.16,len(vals)),vals,s=8,alpha=.35)
        axes[1].plot(np.sort(vals),np.arange(1,len(vals)+1)/len(vals),label=mode)
    axes[0].set(xticks=range(3),xticklabels=list(MODES),ylabel='RD 全图 MAE / dB',title='每点是一帧；帧之间并非独立场景')
    axes[1].set(xlabel='RD 全图 MAE / dB',ylabel='累计帧比例',title='越靠左误差越小；不是通过率');axes[1].legend()
    savefig('ramp_batch.png',fig)

def motion():
    old=SimulationRequest.model_validate(load(PROJECT/'configs/high_altitude_missile_10ghz.json'))
    base=from_pulsed_request(old);base.processing.range_limits_m=(4800,5300);base.sequence.frame_count=3
    base.sequence.frame_interval_s=.05
    direction=np.array(base.platform.velocity_ned_mps)/np.linalg.norm(base.platform.velocity_ned_mps)
    base.platform.velocity_ned_mps=tuple(1200*direction)
    d=base.model_dump()
    d['waveform'].update(carrier_hz=1e9,slot_period_s=50e-6,pulse_width_s=5e-6,samples=7500)
    d['name']='1 GHz, 1200 m/s unambiguous demonstration; original 5 km point geometry'
    base=Simulation.model_validate(d)
    write(OUT/'configs/motion_unambiguous_1ghz.json',base.model_dump(mode='json'))
    rows=[];views={};ref=None
    ambiguity_rows=[]
    for speed in [0.,400.,800.,1200.]:
        b=base.model_copy(deep=True);b.platform.velocity_ned_mps=tuple(speed*direction)
        for fi in range(3):
            original=simulate_frame(b,fi)
            mid=original.metadata['truth_epoch_s'];limit=original.products.metadata['max_unambiguous_velocity_mps']
            for target in b.scene.objects:
                delta=np.array(target.position_ned_m)-np.array(b.platform.position_ned_m)-speed*direction*mid
                radius=float(np.linalg.norm(delta));closing=float(speed*direction@delta/radius)
                full_echo_end=2*radius/299792458+b.waveform.pulse_width_s
                ambiguity_rows.append({'speed_mps':speed,'frame':fi,'target':target.id,'true_range_m':radius,
                    'true_closing_mps':closing,'unambiguous_velocity_limit_mps':limit,
                    'velocity_unambiguous':abs(closing)<limit,'full_echo_inside_PRI':full_echo_end<b.waveform.slot_period_s})
                assert abs(closing)<limit and full_echo_end<b.waveform.slot_period_s
            for label,common in [('both_move',np.array([30.,-10.,5.])),('radar_static',-speed*direction)]:
                other=b.model_copy(deep=True);other.platform.velocity_ned_mps=tuple(speed*direction+common)
                for t in other.scene.objects:t.velocity_ned_mps=tuple(common)
                changed=simulate_frame(other,fi)
                relative_error=float(np.max(abs(changed.adc-original.adc))/max(np.max(abs(original.adc)),1e-30))
                reference=float(original.products.rd_power.max())
                err=compare_power(original.products.rd_power,changed.products.rd_power,reference_power=reference).metrics['mae_db']
                rows.append({'speed_mps':speed,'frame':fi,'time_s':fi*.05,'comparison':label,'complex_adc_relative_max_error':relative_error,'rd_mae_db':err,
                             'passed':relative_error<2e-5})
                if speed==1200 and fi==0:
                    views[label]=changed.products;ref=reference
                    write(OUT/'configs'/f'motion_{label}.json',other.model_dump(mode='json'))
            if speed==1200 and fi==0:views['original']=original.products
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,(key,title) in zip(axes,[('original','A：平台动，目标静止'),('both_move','B：双方增加相同速度'),('radar_static','C：平台静止，目标反向运动')]):
        p=views[key];im=heat(ax,power_db(p.rd_power,ref),p.range_m,p.velocity_mps,title,hi=0)
        fig.colorbar(im,ax=ax,label='共同参考 / dB',shrink=.8)
    fig.suptitle('1 GHz 无模糊演示：5 km / 1200 m/s / 80° 三点场景；相对运动相同，RD 应一致\nPRF 20 kHz、脉宽 5 μs；分米波理想模型自检，不是厘米波实测验证',fontsize=11)
    savefig('motion_rd.png',fig)
    csvout(OUT/'metrics/motion_cases.csv',rows)
    csvout(OUT/'metrics/motion_ambiguity_checks.csv',ambiguity_rows)
    p=views['original'];remaining=p.rd_power.copy();peaks=[]
    for _ in range(3):
        r,d=np.unravel_index(np.argmax(remaining),remaining.shape)
        peaks.append((float(p.range_m[r]),float(p.velocity_mps[d])))
        remaining[max(0,r-20):r+21,:]=0
    truth=sorted([r for r in ambiguity_rows if r['speed_mps']==1200 and r['frame']==0],key=lambda r:r['true_range_m'])
    peak_rows=[]
    for tr,(radius,velocity) in zip(truth,sorted(peaks)):
        peak_rows.append({**tr,'detected_range_m':radius,'detected_velocity_mps':velocity,
            'range_error_m':abs(radius-tr['true_range_m']),'velocity_error_mps':abs(velocity-tr['true_closing_mps'])})
    csvout(OUT/'metrics/motion_peak_checks.csv',peak_rows)
    write(OUT/'metrics/motion_peak_checks.json',peak_rows)
    write(OUT/'metrics/motion_summary.json',{'comparisons':len(rows),'passed':sum(r['passed'] for r in rows),
        'max_complex_error':max(r['complex_adc_relative_max_error'] for r in rows),
        'max_rd_mae_db':max(r['rd_mae_db'] for r in rows),'base_velocity_ned_mps':base.platform.velocity_ned_mps,
        'common_velocity_ned_mps':[30,-10,5],'unambiguous_checks':len(ambiguity_rows),'all_unambiguous':True,
        'native_velocity_cell_mps':p.metadata['native_velocity_cell_mps'],
        'scope':'common-velocity invariance in a nonrelativistic instantaneous-path model',
        'missing_physics':'retarded endpoint motion during propagation, acceleration, orientation changes, multipath'})
    print('motion finished',flush=True)

def angle_demo():
    case={'id':'ra_demo','waveform':'fmcw','pose':'nadir','range_m':5000.,'speed_mps':0.,
          'array':'MIMO_ULA','azimuth_deg':-12.,'elevation_deg':0.,'motion':'target_moves'}
    request=case_request(case)
    direction=scan_directions([12.],[0.])[0,0]@sensor_basis(request.platform).T
    request.scene.objects.append(request.scene.objects[0].model_copy(update={'id':'right','position_ned_m':tuple(np.array(request.platform.position_ned_m)+5000*direction)}))
    direction=scan_directions([0.],[0.])[0,0]@sensor_basis(request.platform).T
    request.scene.objects.append(request.scene.objects[0].model_copy(update={'id':'far','position_ned_m':tuple(np.array(request.platform.position_ned_m)+5120*direction),'amplitude':.6}))
    request.processing.range_limits_m=(4870,5260);request.processing.azimuth_deg=list(np.linspace(-40,40,321))
    request.processing.spatial_window='rectangular'
    products=[]
    for mode,(tx,rx) in MODES.items():
        r=request.model_copy(deep=True);r.array.num_tx=tx;r.array.num_rx=rx;r.array.layout='siso' if mode=='SISO' else 'ula'
        if mode=='SISO':r.processing.azimuth_deg=[0.]
        products.append(simulate_frame(r).products);write(OUT/'configs'/f'angle_{mode}.json',r.model_dump(mode='json'))
    ref=max(float(p.ra_power.max()) for p in products)
    fig,axes=plt.subplots(1,3,figsize=(14,4.6),layout='constrained')
    p=products[0];im=heat(axes[0],power_db(p.rd_power,float(p.rd_power.max())),p.range_m,p.velocity_mps,'SISO RD：左右目标重叠',lo=-40,hi=0)
    fig.colorbar(im,ax=axes[0],label='RD 自身峰值参考 / dB',shrink=.8)
    for ax,p,title in zip(axes[1:],products[1:],['SIMO RA：观察角度分辨能力','MIMO RA：同距目标分成左右峰']):
        # Display fixed zero-Doppler slice, not averaged RA, to show the controlled targets clearly.
        d=np.argmin(abs(p.velocity_mps));ra=abs(p.rad[:,d,0,:])**2
        localref=max(float(abs(q.rad[:,np.argmin(abs(q.velocity_mps)),0,:]).max()**2) for q in products)
        im=heat(ax,power_db(ra,localref),p.range_m,p.azimuth_deg,title,lo=-40,hi=0,ylabel='方位角 / °')
        ax.scatter([5000,5000,5120],[-12,12,0],marker='x',c='white',s=70,label='预设真值');ax.legend(fontsize=8)
        fig.colorbar(im,ax=ax,label='角度切片共同参考 / dB',shrink=.8)
    fig.suptitle('有信息量的 RA：同距离、同速度、不同方向的目标能否分开？\n纯理想仿真；RA 为零 Doppler 切片，白叉为预设真值；不代表真实高空角度精度',fontsize=11)
    savefig('angle_demo.png',fig)
    from scipy.signal import find_peaks
    fig,axes=plt.subplots(1,2,figsize=(12,4.1),layout='constrained')
    rows=[]
    for p,mode in zip(products[1:],['SIMO','MIMO']):
        r=np.argmin(abs(p.range_m-5000));d=np.argmin(abs(p.velocity_mps))
        curve=abs(p.rad[r,d,0,:]).astype(float)**2
        peaks,_=find_peaks(curve,prominence=.1*curve.max());peaks=peaks[abs(p.azimuth_deg[peaks])<25]
        dip=10*np.log10(min(curve[p.azimuth_deg<0].max(),curve[p.azimuth_deg>0].max())/curve[np.argmin(abs(p.azimuth_deg))])
        axes[1].plot(p.azimuth_deg,power_db(curve,float(curve.max())),label=f'{mode}：中央谷深 {dip:.1f} dB')
        rows.append({'mode':mode,'range_bin_m':float(p.range_m[r]),'peak_angles_deg':p.azimuth_deg[peaks].tolist(),
                     'central_dip_db':float(dip),'resolved_by_demo_rule':bool(len(peaks)>=2 and dip>=3)})
    for obj in request.scene.objects:
        delta=np.array(obj.position_ned_m)-np.array(request.platform.position_ned_m)
        local=delta@sensor_basis(request.platform)
        axes[0].scatter(local[1],local[0],s=70)
        axes[0].annotate(obj.id,(local[1],local[0]),xytext=(5,8),textcoords='offset points')
    axes[0].set(xlabel='雷达局部横向 / m',ylabel='雷达局部前向 / m',title='仿真目标几何：不是实际地面场景');axes[0].grid(alpha=.2)
    for a in [-12,12]:axes[1].axvline(a,c='gray',ls='--',lw=1)
    axes[1].set(xlim=(-32,32),ylim=(-30,1),xlabel='方位角 / °',ylabel='各曲线峰值归一 / dB',title='5000 m、零速度单元的角度切片');axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.suptitle('看曲线更直观：两个局部峰之间是否有明显凹谷？\n本演示判据：双峰且谷深 ≥3 dB；仅评估该固定相位、等幅理想场景',fontsize=11)
    savefig('angle_slice.png',fig);write(OUT/'metrics/angle_demo.json',rows)

def terrain():
    with (DEM/'0.csv').open() as f:catalog=list(csv.DictReader(f))
    ordered=sorted(catalog,key=lambda r:(float(r['range']),r['name']))
    choices=[ordered[0],ordered[len(ordered)//2],ordered[-1]]
    fig,axes=plt.subplots(3,3,figsize=(14,11),layout='constrained');rows=[]
    for row,(spec,axs) in enumerate(zip(choices,axes)):
        path=DEM/(spec['name']+'.mat');raw=loadmat(path);z=raw['z'].astype(float);spacing=float(raw['resolution'].item())
        # Exactly sampled source heights, no synthetic hills: central 1 km patch at 50 m grid spacing.
        stride=max(1,round(50/spacing));center=np.array(z.shape)//2
        iy=np.arange(center[0]-10*stride,center[0]+10*stride+1,stride)
        ix=np.arange(center[1]-10*stride,center[1]+10*stride+1,stride)
        sampled=z[np.ix_(iy,ix)];datum=float(sampled[10,10]);height=sampled-datum
        north=(ix-ix[10])*spacing;east=(iy-iy[10])*spacing
        req=Simulation.model_validate({'name':spec['name'],'waveform':{'kind':'pulsed_lfm','carrier_hz':10e9,
            'bandwidth_hz':20e6,'sample_rate_hz':40e6,'pulse_width_s':5e-6,'slot_period_s':100e-6,
            'adc_start_s':25e-6,'samples':1024,'loops':32},
            'platform':{'position_ned_m':[0,0,-5000],'velocity_ned_mps':[20,0,0],
                'boresight_ned':[0,0,1],'right_hint_ned':[0,1,0],'beam_half_angle_deg':15},
            'array':{'layout':'siso','num_tx':1,'num_rx':1},
            'scene':{'terrain':{'kind':'dem','north_m':north.tolist(),'east_m':east.tolist(),
                'elevation_up_m':height.tolist(),'relative_reflectivity':.002},'seed':42},
            'processing':{'range_limits_m':[4400,5700],'azimuth_deg':[0.],'elevation_deg':[0.]}})
        compiled=compile_scene(req.scene);frame=simulate_frame(req,compiled=compiled);p=frame.products
        # Independent geometry calculation from source samples, not scene_truth().
        n,e=np.meshgrid(north,east);verts=np.stack([n,e,-height],axis=-1)
        a,b,c,d=verts[:-1,:-1],verts[:-1,1:],verts[1:,:-1],verts[1:,1:]
        centroids=np.stack([(a+b+d)/3,(a+d+c)/3],axis=2).reshape(-1,3)
        source_r=np.linalg.norm(centroids-np.array([0,0,-5000]),axis=1)
        engine_r=np.linalg.norm(compiled.positions_ned_m-np.array([0,0,-5000]),axis=1)
        # Triangulation conventions can use the other diagonal: compare envelope, not assume index identity.
        envelope_difference=max(abs(source_r.min()-engine_r.min()),abs(source_r.max()-engine_r.max()))
        profile=np.mean(p.rd_power,axis=1);margin=2*299792458/(2*20e6)
        keep=(p.range_m>=source_r.min()-margin)&(p.range_m<=source_r.max()+margin)
        fraction=float(profile[keep].sum()/max(profile.sum(),1e-30))
        im=axs[0].pcolormesh(north,east,sampled,shading='auto',cmap='terrain');fig.colorbar(im,ax=axs[0],label='源文件高程 / m',shrink=.8)
        axs[0].set(title=f'本地 DEM {row+1}：中心区域起伏 {np.ptp(sampled):.0f} m',xlabel='局部 X → North / m',ylabel='局部 Y → East / m');axs[0].set_aspect('equal')
        im=heat(axs[1],power_db(p.rd_power,float(p.rd_power.max())),p.range_m,p.velocity_mps,'地形 RD（无配套实测）',hi=0)
        axs[1].set_xlim(source_r.min()-60,source_r.max()+60);fig.colorbar(im,ax=axs[1],label='各场景峰值归一 / dB，仅形状',shrink=.8)
        axs[2].plot(p.range_m,power_db(profile,float(profile.max())));axs[2].axvspan(source_r.min(),source_r.max(),color='#52a67d',alpha=.25,label='由源高程独立计算的斜距范围')
        axs[2].set(xlim=(source_r.min()-60,source_r.max()+60),ylim=(-60,2),xlabel='距离 / m',ylabel='距离响应 / dB',title=f'几何范围 ±2分辨单元内功率 {fraction:.1%}');axs[2].legend(fontsize=7)
        record={'source_file':str(path),'sha256':sha(path),'source_shape':list(z.shape),'source_resolution_m':spacing,
            'source_full_relief_m':float(np.ptp(z)),'patch_relief_m':float(np.ptp(sampled)),
            'center_elevation_subtracted_m':datum,'local_axes_assumption':'source x->north, y->east; geographic heading unknown',
            'sampling_indices_x':ix.tolist(),'sampling_indices_y':iy.tolist(),'simulation_grid_m':stride*spacing,
            'range_envelope_m':[float(source_r.min()),float(source_r.max())],
            'source_vs_engine_envelope_difference_m':float(envelope_difference),'energy_inside_envelope_plus_2_resolution':fraction,
            'original_acquisition_and_license':'not established from available local metadata','paired_measured_RD':False}
        rows.append(record);write(OUT/'configs'/f'dem_{row+1}.json',req.model_dump(mode='json'))
        print('DEM',row+1,spec['name'],fraction,flush=True)
    fig.suptitle('用旧工程已有 DEM 替代人工平地/山丘：输入可追到文件，回波尚无实测证明\n三块源地形按目录全幅起伏的最小 / 中位 / 最大预选；展示固定中心 1 km 区域，局部起伏不保证同序',fontsize=12)
    savefig('terrain_source_rd.png',fig);write(OUT/'metrics/terrain_sources.json',rows)

def table(headers,rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+str(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(v)+'</td>' for v in r)+'</tr>' for r in rows)+'</tbody></table></div>'
def figure(name,caption):return f'<figure><img src="images/{name}" alt="{html.escape(caption)}"><figcaption>{caption}</figcaption></figure>'

def report():
    mo=load(OUT/'metrics/motion_summary.json');rp=load(OUT/'metrics/ramp_summary.json');dem=load(OUT/'metrics/terrain_sources.json')
    body='<header><small>2026-10-10 · 雷达仿真平台周报 · 1 GHz / 1200 m/s 修订版</small><h1>功能扩展与验证</h1><p>主运动演示：1 GHz、1200 m/s、5 km，单 PRF 无模糊。<br>已扩展：双方运动、多通道、MIMO、RAD/RA、DEM 输入。<br>当前结论：多功能框架已拓宽，真实回波的保真仍不足。</p></header>'
    body+=table(['本报告的频段与场景','用途 / 边界'],[['1 GHz 分米波：1200 m/s 主运动场景','以合适参数展示距离与速度；不是把低速改标成高速，也没有使用多 PRF'],['77 GHz 毫米波：RAMP 实测对比','验证单通道、多接收、多发多收处理；不直接证明厘米波地表保真'],['10 GHz 厘米波：保留的 DEM 与理想角度示例','独立的既有功能演示，分别标注参数；不与 1 GHz 运动图混作同一场景']])
    body+=table(['项目','本次证据','能说明什么'],[
        ['相对运动',f"新跑 {mo['comparisons']} 组等价对照 / {mo['passed']} 组满足数值阈值",'检查相对速度、时间与相位处理；不是实测保真'],
        ['SISO / SIMO / MIMO','RAMP 3 段采集 × 60 帧 × 3 通道模式 = 540 项对比','同一数据的嵌套通道子集；不是 540 个独立场景'],
        ['地形','旧工程 3 个 DEM 文件 + 独立几何斜距范围','输入可复查；原始采集元数据和对应实测 RD 仍缺失'],
        ['RA','同距双目标理想演示 + RAMP 实测 / 仿真 / 差值','展示角度维的作用，同时保留真实失败案例']])
    body+='<section><h2>1. 平台与目标都能做匀速运动：1 GHz、1200 m/s</h2>'+figure('motion_input.png','三个理想点目标与原 5 km 场景位置一致，速度保留 1200 m/s。地面线仅说明坐标，不参与回波；使用 1 GHz 分米波雷达。')+figure('motion_rd.png','A：平台动、目标静；B：双方同时增加相同平移速度；C：平台静、目标反向运动。三者完整相对运动相同，RD 一致。')
    body+=table(['设置','数值'],[['雷达','1 GHz；脉冲 LFM；60 MHz；150 MHz ADC；5 μs；PRF 20 kHz；64 pulses'],['展示场景','5 km；1200 m/s；80°下视；3 个理想点目标；单发单收'],['无模糊边界','径向速度 ±1498.96 m/s；距离约 7.49 km。本例斜距约 5.03–5.14 km；回波结束约 39.3 μs，早于 50 μs PRI'],['B 的改动','平台和所有点目标同时加 (30, −10, 5) m/s；保持完整相对运动一致'],['批量范围','0、400、800、1200 m/s × 3 帧 × 2 种等价变换 = 24 组'],['新跑结果',f"复数 ADC 最大相对误差 {mo['max_complex_error']:.3g}；RD MAE 最大 {mo['max_rd_mae_db']:.3g} dB；36 个目标–帧状态无速度/距离模糊"]])
    body+=table(['目标','真实径向速度','RD 峰速度','速度误差'],[[r['target'],f"{r['true_closing_mps']:.2f} m/s",f"{r['detected_velocity_mps']:.2f} m/s",f"{r['velocity_error_mps']:.2f} m/s"] for r in load(OUT/'metrics/motion_peak_checks.json')])
    body+=f'<p class="note">当前原生速度单元为 {mo["native_velocity_cell_mps"]:.2f} m/s，约 1200 m/s 的真值落在约 1217.91 m/s 的 FFT 格点上；这是有限采样格点误差，不是速度折叠，也没有用真值校正坐标。等价对照只是模型内部自检，不代表实测保真。1 GHz 是分米波；保留 1200 m/s 的同时改变了载频、PRF 与脉宽，不能写成原 10 GHz 配置已经消除模糊。</p></section>'
    body+='<section id="ramp"><h2>2–4. SISO、SIMO、MIMO：同一实测场景逐级对照</h2>'
    body+='<p>数据：<a href="https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection">RAMP 官方原始 ADC 数据</a>。先处理实测，再用公开标签建立点目标场景；两边使用相同采集参数与处理窗。照片只说明真实场景，不能替代车辆三维模型。</p>'
    body+=table(['项目','真实数据','仿真输入'],[
        ['雷达 / 波形','TI AWR1843；77 GHz 线性 FMCW','使用同一公开配置；不把 77 GHz 结果当成 10 GHz 地表证明'],
        ['采样','128 samples；255 loops；4 Rx × 2 Tx','维度保持一致；SISO/SIMO 只选对应通道'],
        ['调频与时序','斜率 21 MHz/μs；ADC 4 MHz；单 Tx slot 60 μs','32 μs 采样窗口；同一 Tx 间隔仍为 120 μs，不因选子集而改成 60 μs'],
        ['不完全明确的时序','物理 ADC-start 延迟未完全核实','沿用固定 0 μs 起点近似；这不是已知真值'],
        ['处理','Hamming 距离 / Doppler / 空间窗；距离 1–24 m','相同处理；无逐图平移、无单独峰值归一化'],
        ['目标','同名照片 + 俯视框 / UID / 类别','每个标签一个理想点；速度由标签轨迹估计；未知速度用 0 占位'],
        ['幅度 / 环境','包含真实车身、道路、杂波与接收机效应','冻结历史标定增益 + 类别相对幅度；没有复现完整环境']])
    body+=figure('ramp_scene_1.png','汽车场景：左图是真实照片，中图是公开标签，右图才是当前实际参与仿真的目标。')
    for mode,text in [('SISO','取 Tx1–Rx1。它是实测单通道子集，不是另一台独立 SISO 雷达采集。'),('SIMO','取 Tx1 的 4 个 Rx。RD 展示通道功率平均；测角能力要看后面的 RA。'),('MIMO','使用全部 2Tx4Rx。先保留复数通道，再进行 TDM 补偿和角度处理；通道多不等于场景更逼真。')]:
        body+=f'<h3>{mode}</h3><p>{text}</p>'+figure(f'ramp_{mode}_1.png',f'{mode}：实测 → 仿真 → 差值。同一幅汽车场景，避免用不同场景制造“升级效果”。')
    body+='<h3>不是只跑一张：全部 180 帧、3 段保留采集的结果</h3>'+figure('ramp_batch.png','540 个模式–帧比较，共享 180 帧原始数据。没有把相邻帧当作独立场景，也没有预设实测一定通过。')
    body+='<p>沿用历史记录级分组与冻结增益。这三段数据此前已被查看和用于诊断，本次是跨记录复核，不是全新、从未看过的最终盲测。</p>'
    body+=table(['模式','帧 / 采集段','全图 MAE 均值','中位数','P90','标签距离区 MAE 均值'],[
        [s['mode'],f"{s['frames']} / {s['records']}",f"{s['mean']:.2f} dB",f"{s['median']:.2f} dB",f"{s['p90']:.2f} dB",f"{s['target_mean']:.2f} dB"] for s in rp['groups']])
    body+='<p class="note">MAE：每个像素“仿真功率 dB − 实测功率 dB”的绝对值取平均，越小越好，不是准确率。P90：90% 的帧误差不超过这个值。标签距离区保留标签附近距离、包含全部速度；不是精确目标分割。参考功率固定、下限 −60 dB，不能跨不同归一化报告直接比较。</p>'
    body+='<p>三种模式的 MAE 不构成“天线越多越差”的结论：选取的通道不同，真实背景与通道响应也不同；这张表检查各模式的实测差距，阵列增加的角度信息要在 RA 中判断。当前固定类别幅度与历史统一增益并非逐通道标定。</p>'
    background=load(OUT/'补充说明/背景误差汇总.json')['mean']
    body+='<h3>误差主要在哪里？</h3>'+table(['180 帧 MIMO 的探索性拆分','像素占比','区域内 MAE','总绝对误差贡献'],[
        ['低于该帧实测峰值 20 dB 的区域',f"{background['weak_pixel_fraction']:.2%}",f"{background['weak_mae_db']:.2f} dB",f"{background['weak_error_fraction']:.2%}"],
        ['其余较强回波区域',f"{1-background['weak_pixel_fraction']:.2%}",f"{background['strong_mae_db']:.2f} dB",f"{1-background['weak_error_fraction']:.2%}"]])
    body+='<p class="note">低功率像素数量多，主导全图 MAE；强回波区域的单像素差距仍很大。低功率不等于纯热噪声，还可能包含杂波、旁瓣和弱目标。不能靠随意加噪声把 MAE 降下来就称为保真提高。以上比例为逐帧计算后的平均值。</p>'
    for i in [2,3]:body+='<details><summary>展开固定样例 '+str(i)+'：场景与三个通道模式</summary>'+figure(f'ramp_scene_{i}.png','每段保留采集的中间帧，在计算指标之前固定选择。')+''.join(figure(f'ramp_{m}_{i}.png',m+' 实测与仿真对照') for m in MODES)+'</details>'
    body+=figure('ramp_worst.png','额外保留误差最大的 MIMO 案例，不只展示效果较好的图片。记录：'+rp['worst']['frame_id'])+'</section>'
    body+='<section><h2>5. 地形：使用旧工程已有 DEM，明确溯源边界</h2>'+figure('terrain_source_rd.png','地形不是这次手工画出的山丘。高程来自本地旧数据；统一零点与局部坐标只是仿真适配。每幅 RD 按自身峰值显示，只比较形状，不比较绝对强度。')
    body+=table(['DEM 源文件','原网格 / 仿真网格','当前区域起伏','几何包络交叉检查'],[[Path(d['source_file']).name,f"{d['source_resolution_m']:g} / {d['simulation_grid_m']:g} m",f"{d['patch_relief_m']:.0f} m",f"独立包络与引擎包络差 {d['source_vs_engine_envelope_difference_m']:.3g} m"] for d in dem])
    body+=table(['能追溯 / 检查','尚不能证明'],[['源路径、SHA-256、抽样索引、原始高程、输入配置均保存','尚未补齐原始产品、地理坐标、采集日期和许可；不能仅凭文件名认定来源'],['用源高程独立计算斜距范围，再查看距离响应所在区域','无配套真实 RD；几何一致不等于回波强度、谱形真实'],['统一 10 GHz、5 km、20 m/s、20 MHz、32 pulses；相同散射假设','不是历史复杂运动 RD 的严格重放；高程网格也没有材料和建筑分类']])+'</section>'
    body+='<section><h2>6. RA：看清“相同距离的目标来自不同方向”</h2>'+figure('angle_demo.png','5 km 理想场景：两个目标都在 5000 m、零速度，方位分别 −12° 和 +12°；另一个在 5120 m、0°。这张图展示功能，不包装成实测验证。')
    body+=figure('angle_slice.png','两条角度曲线各自按峰值归一，只解释角度分辨，不用于幅度保真比较。')
    angle=load(OUT/'metrics/angle_demo.json')
    body+=table(['模式','输入角度真值','本例角度峰','中央谷深','能否按本例判据分成双峰'],[
        [a['mode'],'−12° / +12°',' / '.join(f'{v:g}°' for v in a['peak_angles_deg']),f"{a['central_dip_db']:.2f} dB",'能' if a['resolved_by_demo_rule'] else '不能'] for a in angle])
    body+='<p class="note">分成两个峰不等于角度完全准确：本例 MIMO 峰仍偏离输入真值，有限孔径与同距同速相干目标会相互影响。这里只说明增加孔径后具备分离能力，不宣称达到某个实测角度精度。</p>'
    body+='<h3>实测 RA 的输入与对应目标</h3>'+figure('ramp_scene_1.png','本章汽车样例 1：左为同名真实相机图，中为公开标签，右为仿真采用的中心点与估计速度。实测 RA 由该帧多通道 ADC 计算，照片仅辅助理解，不是 RA 的计算输入。')
    body+=figure('ra_SIMO_1.png','同一汽车场景的 SIMO：Tx1 的四个 Rx；输入目标与上图一致。')
    body+=figure('ra_MIMO_1.png','同一汽车场景的 MIMO：全部 2Tx4Rx。统一色标，保留主峰偏差、背景与漏算；白叉是几何标签中心，不一定是雷达最强散射点。')
    body+='<details><summary>展开其他实测 RA：每组先展示输入场景</summary>'+figure('ramp_scene_2.png','固定样例 2：真实照片、标签及对应仿真输入。')+figure('ra_MIMO_2.png','固定样例 2 的 MIMO RA。')+figure('ramp_scene_3.png','固定样例 3：真实照片、标签及对应仿真输入。')+figure('ra_MIMO_3.png','固定样例 3 的 MIMO RA。')+'</details></section>'
    body+='<section><h2>当前缺陷与下一步</h2>'+table(['问题：直白地说','现在能看到的情况','下一步只补一个明确因素'],[
        ['目标太简单：用一个点代表一辆车','没有车壳、车轮与多个反射位置；真实 RD 的宽度、多个峰和微动难以复现','先给简单物体增加有来源的尺寸 / 多散射点；固定标定后做跨记录对比'],
        ['峰的位置不保证对得准','标签中心不是散射中心；速度来自标签差分，部分速度未知；时序与通道标定还不完全确定','分开检查距离零点、采样时序、速度标签和角度符号；不要把所有偏差归咎于一个因素'],
        ['不是“没有旁瓣”，而是旁瓣还对不齐','FFT、匹配滤波与阵列叠加会自然产生旁瓣；真实旁瓣还受窗、方向图、通道误差和背景影响','在受控单目标上固定窗与相位，先比较主峰和旁瓣，再加一个有依据的误差项'],
        ['背景没有复现，图会过于干净','统一点目标链路没有道路 / 植被背景、通用环境多径和真实接收机；实测有大量模型没有生成的能量','先加入可开关的简单背景并报告改善 / 退化；不逐帧调噪声来降低 MAE'],
        ['海杂波统计模型不能代表陆地 DEM','已有 IPIX 是用实测前段估计统计参数，再比较后段平均 RD；不是根据海浪三维表面生成回波','继续保留独立统计基准，后续再与地表模型衔接；不混入汽车验证通过率'],
        ['DEM 只有形状，没有完整散射信息','能计算坡面、斜距、相对运动；不能自动知道土壤、植被、湿度与对应频段的反射','先补产品来源和坐标，再做网格收敛；材料反射参数单独补充'],
        ['阵列仍是理想模型','可以输出 RA/RAD；不等于真实天线通道已标定，更不等于完整相控阵硬件','先单目标测角与通道顺序检查，再加入已测得的幅相 / 方向图参数'],
        ['运动等价只证明模型内部一致','当前为匀速、固定姿态和近似传播时延；不是高速全部物理效应','先独立解析基准与受控运动数据；再评估传播期间运动和姿态变化']])
    body+='<p class="conclusion">本周可写：完成多种运动、收发配置与地形输入的统一仿真和对照展示；公开实测对比仍存在明显差异，暂不宣称高保真。下一阶段先把输入、时序与目标几何补齐，再逐项增加物理效应。</p></section>'
    body+='<footer><h3>数据与复现</h3><p>完整参数：configs/；逐帧指标与来源：metrics/；图片：images/。原始 ADC 与缓存仍在 NAS，不随报告打包。保留旧报告与旧工程，未修改模型、未开启前端。</p><p><a href="https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection">RAMP 官方仓库</a> · <a href="https://ieee-dataport.org/documents/raw-adc-data-77ghz-mmwave-radar-automotive-object-detection">IEEE DataPort 发布页</a> · <a href="https://arxiv.org/abs/2011.08981">RAMP-CNN 论文</a></p><p>外部来源使用此前已核对的官方链接；本次网页连接失败。报告数值全部来自本地原始文件与新跑结果。</p></footer>'
    css='''body{margin:0;background:#f3f5f7;color:#223244;font:16px/1.6 system-ui,"Noto Sans CJK SC",sans-serif}main{max-width:1250px;margin:auto;padding:30px}header,section,footer{background:white;border:1px solid #dae2e9;border-radius:12px;padding:26px;margin-bottom:24px}h1{font-size:34px;margin:6px 0}h2{font-size:23px;border-left:5px solid #247b9d;padding-left:12px}h3{font-size:19px}p{margin:12px 0}.table{overflow:auto;margin:15px 0}table{width:100%;border-collapse:collapse;background:white;font-size:14px}th,td{padding:12px;text-align:left;border:1px solid #dae2e9;vertical-align:top}th{background:#e8f1f5}figure{margin:22px 0}img{display:block;width:100%;height:auto}figcaption{font-size:13px;color:#52677b;margin:8px 3px}.note{background:#fff5dc;padding:14px;font-size:14px}.conclusion{background:#eaf4f5;padding:18px}details{margin:18px 0;padding:14px;border:1px solid #ccd9df}summary{cursor:pointer}a{color:#14698e}small{color:#587589}@media(max-width:600px){main{padding:9px}header,section,footer{padding:14px}h1{font-size:27px}h2{font-size:20px}table{min-width:600px}}@media print{body{background:white}main{padding:0;max-width:none}section,header,footer{border:0}figure,tr{break-inside:avoid}details{display:block}}'''
    doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>雷达仿真平台周报 · 2026-10-10</title><style>'+css+'</style><body><main>'+body+'</main><script>Promise.all(Array.from(document.images).map(i=>i.decode().catch(()=>null))).then(()=>document.documentElement.dataset.imagesReady="yes")</script></body></html>'
    (OUT/'周报_分离图片版.html').write_text(doc)
    for p in (OUT/'images').glob('*.png'):
        doc=doc.replace('images/'+p.name,'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode())
    (OUT/'雷达仿真平台周报.html').write_text(doc)
    sources=[*PROJECT.glob('backend/radar_platform/unified/*.py'),PROJECT/'backend/radar_platform/comparison.py',
             PROJECT/'backend/radar_platform/ramp_alignment.py',PROJECT/'backend/radar_platform/ramp_comparison.py',
             PROJECT/'scripts/verify_unified_ramp.py',PROJECT/'scripts/validate_unified.py',
             PROJECT/'scripts/explain_weekly_inputs.py',Path(__file__)]
    write(OUT/'metrics/source_manifest.json',{'date':'2026-10-10','source_hashes':{str(p.relative_to(PROJECT)):sha(p) for p in sources},'data_cache':str(CACHE),'model_changed':False})
    with zipfile.ZipFile(OUT/'复现脚本.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in [*sources,PROJECT/'scripts/render_scene.py',PROJECT/'scripts/check_weekly_evidence_report.py']:
            z.write(p,p.relative_to(PROJECT))
    with zipfile.ZipFile(OUT.parent/'雷达周报_2026-10-10_图表与参数.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in OUT.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(OUT))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['ramp','motion','angle','terrain','inputs','report','all'],default='all');ap.add_argument('--cached',action='store_true');args=ap.parse_args()
    for p in [OUT/'images',OUT/'configs',OUT/'metrics',CACHE]:p.mkdir(parents=True,exist_ok=True)
    configure_font()
    from explain_weekly_inputs import main as input_figures
    actions={'motion':motion,'angle':angle_demo,'terrain':terrain,'ramp':lambda:ramp(not args.cached),'inputs':input_figures,'report':report}
    for k,f in actions.items():
        if args.stage in ('all',k):print('Starting',k,flush=True);f()

if __name__=='__main__':main()
