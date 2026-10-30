#!/usr/bin/env python3
"""Systematic analytic (NOT real-data) motion and array acceptance campaign."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
from itertools import product
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from radar_platform.models import RadarConfig,PlatformConfig,SceneConfig,SequenceConfig,SimulationRequest,TargetConfig,C0
from radar_platform.simulator import simulate_frame
from radar_platform.fmcw import FMCWConfig
from radar_platform.array_fmcw import TDMArray,simulate_motion_adc,process_array_adc
from radar_platform.comparison import compare_power,representative_indices
from radar_platform.repeat_metrics import distribution


MODES=['radar_moves','target_moves','both_move','same_velocity','transverse_target','both_static']


def motion(mode,speed):
    vr=np.zeros(3);vt=np.zeros(3)
    if mode=='radar_moves':vr[0]=speed
    elif mode=='target_moves':vt[0]=-speed
    elif mode=='both_move':vr[0]=.4*speed;vt[0]=-.6*speed
    elif mode=='same_velocity':vr[0]=speed;vt[0]=speed
    elif mode=='transverse_target':vt[1]=speed
    return vr,vt


def independent_truth(position,vr,vt,t):
    # Evaluation only; these values never enter peak detection or motion solver.
    delta=position+(vt-vr)*t
    r=float(np.linalg.norm(delta));closing=float(np.dot(vr-vt,delta/r))
    return r,closing


def save_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    # Selection is persisted BEFORE scores. These are deterministic parameter cases,
    # never independent physical acquisitions.
    plan={'modes':MODES,'pulsed_ranges_m':[1200,2400,3600],'pulsed_speeds_mps':[-30,0,30],
          'pulsed_angles_deg':[-5,0,5],'fmcw_ranges_m':[6,12,18],'fmcw_speeds_mps':[-2,0,2],
          'fmcw_angles_deg':[-25,0,25],'range_fractional_bins':[0,.37],
          'array_angles_deg':[-60,-40,-20,0,20,40,60],'array_layouts':['1Tx4Rx','2Tx4Rx'],
          'seed':42,'data_origin':'analytic_and_simulated_only'}
    (a.output/'selection.json').write_text(json.dumps(plan,indent=2)+'\n')
    rows=[]
    pulsed=RadarConfig(carrier_hz=10e9,bandwidth_hz=10e6,sample_rate_hz=25e6,pulse_width_s=8e-6,prf_hz=10000,pulses_per_cpi=64,beam_half_angle_deg=10)
    for i,(mode,r0,speed,angle,offset) in enumerate(product(MODES,plan['pulsed_ranges_m'],plan['pulsed_speeds_mps'],plan['pulsed_angles_deg'],plan['range_fractional_bins'])):
        step=C0/(2*pulsed.sample_rate_hz);r0+=offset*step
        pos=r0*np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)),0.])
        vr,vt=motion(mode,speed)
        platform=PlatformConfig(position_ned_m=(0,0,0),speed_mps=float(np.linalg.norm(vr)),heading_deg=0 if vr[0]>=0 else 180,dive_angle_deg=0,boresight_ned=(1,0,0))
        target=TargetConfig(id='T',position_ned_m=tuple(pos),velocity_ned_mps=tuple(vt))
        request=SimulationRequest(radar=pulsed,platform=platform,scene=SceneConfig(targets=[target]),sequence=SequenceConfig(frame_count=1))
        result=simulate_frame(request,0)
        truth_r,truth_v=independent_truth(pos,vr,vt,.5*pulsed.pulses_per_cpi/pulsed.prf_hz)
        peak=result.metadata.blind_detections[0]
        rerr=abs(peak.range_using_apparent_doppler_m-truth_r);verr=abs(peak.apparent_velocity_mps-truth_v)
        vstep=C0/pulsed.carrier_hz*pulsed.prf_hz/(2*pulsed.pulses_per_cpi)
        rows.append({'case_id':f'pulsed_{i:04d}','waveform':'pulsed_lfm','mode':mode,'range_m':r0,'speed_parameter_mps':speed,
            'angle_deg':angle,'fractional_bin':offset,'true_mid_range_m':truth_r,'true_closing_mps':truth_v,
            'range_error_m':rerr,'velocity_error_mps':verr,'range_bin_error':rerr/step,'velocity_bin_error':verr/vstep,
            'passed':bool(rerr<=step and verr<=vstep)})
        if i%100==0:print('pulsed',i,flush=True)
    fcfg=FMCWConfig(76.5e9,77.5e9,50e-6,60e-6,128,64,4e6,2e-6)
    array=TDMArray.ideal_ula(fcfg,1,1)
    for i,(mode,r0,speed,angle,offset) in enumerate(product(MODES,plan['fmcw_ranges_m'],plan['fmcw_speeds_mps'],plan['fmcw_angles_deg'],plan['range_fractional_bins'])):
        rstep=fcfg.native_range_resolution_m;r0+=offset*rstep
        pos=r0*np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)),0.]);vr,vt=motion(mode,speed)
        target=TargetConfig(id='T',position_ned_m=tuple(pos),velocity_ned_mps=tuple(vt))
        adc=simulate_motion_adc(fcfg,[target],array,platform_velocity_ned_mps=vr)
        result=process_array_adc(adc,fcfg,array,azimuth_deg=[0.])
        ir,idop=np.unravel_index(np.argmax(abs(result.rd_channels[:,:,0,0])),result.rd_channels.shape[:2])
        tmid=(fcfg.chirps-1)*array.loop_period_s/2+fcfg.adc_start_s+(fcfg.samples-1)/fcfg.sample_rate_hz/2
        truth_r,truth_v=independent_truth(pos,vr,vt,tmid)
        # Up-chirp tx*conj(rx): beat = slope*delay - closing fd. Use ONLY detected fd.
        detected_range=result.range_m[ir]+C0*result.closing_doppler_hz[idop]/(2*fcfg.slope_hz_s)
        rerr=abs(detected_range-truth_r);verr=abs(result.velocity_mps[idop]-truth_v)
        vstep=fcfg.wavelength_m/(2*fcfg.chirps*array.loop_period_s)
        rows.append({'case_id':f'fmcw_{i:04d}','waveform':'fmcw','mode':mode,'range_m':r0,'speed_parameter_mps':speed,
            'angle_deg':angle,'fractional_bin':offset,'true_mid_range_m':truth_r,'true_closing_mps':truth_v,
            'range_error_m':rerr,'velocity_error_mps':verr,'range_bin_error':rerr/rstep,'velocity_bin_error':verr/vstep,
            'passed':bool(rerr<=rstep and verr<=vstep)})
        if i%100==0:print('fmcw',i,flush=True)
    save_csv(a.output/'motion_cases.csv',rows)
    arows=[]
    for i,(ntx,r,angle,speed) in enumerate(product([1,2],[6.,12.,18.],plan['array_angles_deg'],[-2.,0.,2.])):
        array=TDMArray.ideal_ula(fcfg,ntx,4)
        direction=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)),0])
        target=TargetConfig(id='T',position_ned_m=tuple(r*direction),velocity_ned_mps=tuple(-speed*direction))
        adc=simulate_motion_adc(fcfg,[target],array)
        result=process_array_adc(adc,fcfg,array)
        ir,iv,ia=np.unravel_index(np.argmax(abs(result.rad)),result.rad.shape)
        error=abs(result.azimuth_deg[ia]-angle)
        uncomp=process_array_adc(adc,fcfg,array,tdm_compensation=False)
        ua=uncomp.azimuth_deg[np.unravel_index(np.argmax(abs(uncomp.rad)),uncomp.rad.shape)[2]]
        arows.append({'case_id':f'array_{i:04d}','tx':ntx,'rx':4,'range_m':r,'angle_deg':angle,'closing_mps':speed,
                      'detected_angle_deg':float(result.azimuth_deg[ia]),'angle_error_deg':float(error),
                      'uncompensated_angle_error_deg':float(abs(ua-angle)),'passed':bool(error<=1.0)})
    save_csv(a.output/'array_cases.csv',arows)
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
    for ax,key,title in zip(axes[:2],['range_bin_error','velocity_bin_error'],['距离误差 / 原生单元','速度误差 / 原生单元']):
        for j,w in enumerate(['pulsed_lfm','fmcw']):
            y=[x[key] for x in rows if x['waveform']==w]
            ax.scatter(np.full(len(y),j)+np.linspace(-.15,.15,len(y)),y,s=8,alpha=.4)
        ax.axhline(1,ls='--',c='red');ax.set_xticks([0,1],['脉冲 LFM','FMCW']);ax.set_title(title)
    for ntx in [1,2]:
        y=[x['angle_error_deg'] for x in arows if x['tx']==ntx]
        axes[2].scatter(np.full(len(y),ntx)+np.linspace(-.1,.1,len(y)),y,s=12,alpha=.4)
    axes[2].set_xticks([1,2],['1Tx4Rx','2Tx4Rx']);axes[2].set_title('角度误差 / °（1° 扫描网格）')
    fig.suptitle('批量解析检查：数值仿真，不是真实数据保真结论')
    fig.savefig(a.output/'analytic_summary.png',dpi=160);plt.close(fig)
    # Dynamic ideal sequence, complete complex data persisted, no fake real-data label.
    array=TDMArray.ideal_ula(fcfg,2,4)
    target=TargetConfig(id='moving',position_ned_m=(12,2,0),velocity_ned_mps=(-2,.5,0))
    maps=[];frames=[]
    for t in [0.,.5,1.,1.5]:
        adc=simulate_motion_adc(fcfg,[target],array,frame_start_s=t)
        result=process_array_adc(adc,fcfg,array)
        maps.append(np.mean(abs(result.rd_channels)**2,axis=(2,3)))
        frames.append({'frame_start_s':t,'target_position_ned_m':(np.array(target.position_ned_m)+np.array(target.velocity_ned_mps)*t).tolist()})
        np.savez_compressed(a.output/f'ideal_motion_{t:g}s.npz',adc=adc,rd=result.rd_channels,rad=result.rad,
                            range_m=result.range_m,velocity_mps=result.velocity_mps,azimuth_deg=result.azimuth_deg)
    fig,axes=plt.subplots(1,4,figsize=(15,4),layout='constrained')
    reference=max(float(maps[0].max()),1e-30)
    for ax,power,meta in zip(axes,maps,frames):
        im=ax.imshow(np.maximum(10*np.log10(np.maximum(power.T/reference,1e-6)),-60),origin='lower',aspect='auto',
                     extent=[result.range_m[0],result.range_m[-1],result.velocity_mps[0],result.velocity_mps[-1]],vmin=-60,vmax=6,cmap='turbo')
        ax.set(xlim=(6,16),ylim=(-4,4),xlabel='距离 / m',ylabel='接近速度 / (m/s)',title=f"纯仿真 t={meta['frame_start_s']:g} s")
    fig.colorbar(im,ax=axes.ravel().tolist(),label='统一参考功率 / dB',shrink=.7)
    fig.suptitle('新增目标匀速平移：雷达静止，目标接近并横向移动；无实测配对')
    fig.savefig(a.output/'ideal_target_motion.png',dpi=160);plt.close(fig)
    sources=['models.py','physics.py','simulator.py','kinematics.py','detection.py','fmcw.py','array_fmcw.py','array_api.py','comparison.py']
    project=Path(__file__).resolve().parents[1]
    summary={'generated_utc':datetime.now(timezone.utc).isoformat(),'is_real':False,
        'motion_cases':len(rows),'motion_passed':sum(r['passed'] for r in rows),'array_cases':len(arows),'array_passed':sum(r['passed'] for r in arows),
        'by_waveform':{w:{'count':sum(r['waveform']==w for r in rows),'range_error_m':distribution([r['range_error_m'] for r in rows if r['waveform']==w]),
                         'velocity_error_mps':distribution([r['velocity_error_mps'] for r in rows if r['waveform']==w])} for w in ['pulsed_lfm','fmcw']},
        'source_hashes':{f'backend/radar_platform/{s}':hashlib.sha256((project/'backend/radar_platform'/s).read_bytes()).hexdigest() for s in sources},
        'bounds':'no noise, no ambiguity, small motion per CPI; independent peak extraction; synthetic analytic checks only',
        'sequence':frames}
    (a.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['motion_cases','motion_passed','array_cases','array_passed']},indent=2))


if __name__=='__main__':main()
