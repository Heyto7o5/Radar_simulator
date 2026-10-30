#!/usr/bin/env python3
"""Explain numerical checks and spatial resolution with small scientific plots."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

from radar_platform.unified.geometry import build_layout, sensor_basis, scan_directions
from radar_platform.unified.engine import simulate_frame
from radar_platform.unified.processing import array_factor
from radar_platform.unified.storage import write_json
from validate_unified import case_request
from render_scene import configure_font


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);configure_font()
    with (args.validation/'cases.csv').open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,key,title in zip(axes,['range_error_bins','velocity_error_bins','elevation_error_deg'],
                            ['距离峰误差 / 原生采样单元','速度峰误差 / 原生 Doppler 单元','俯仰角峰误差 / °（含非整数角）']):
        for index,waveform in enumerate(['pulsed_lfm','fmcw']):
            values=[float(r[key]) for r in rows if r['waveform']==waveform and r[key]!='']
            ax.scatter(index+np.linspace(-.12,.12,len(values)),values,s=7,alpha=.45)
        ax.axhline(1,color='red',ls='--',label='本轮解析验收阈值')
        ax.set_xticks([0,1],['脉冲 LFM','FMCW']);ax.set_title(title);ax.set_ylim(-.03,1.1);ax.grid(alpha=.15)
    fig.suptitle(f'{len(rows)} 组理想数值检查：两波形 × 多阵列 × 距离/运动 × 前视/下视/斜视\n不依赖公开数据拟合，也不等于实测散射保真',fontsize=12)
    fig.savefig(args.output/'analytic_checks.png',dpi=160);plt.close(fig)

    case={'id':'two_targets','waveform':'fmcw','pose':'nadir','range_m':5000.,'speed_mps':0.,
          'array':'MIMO_ULA','azimuth_deg':-12.,'elevation_deg':0.,'motion':'target_moves'}
    request=case_request(case)
    direction=scan_directions([12.],[0.])[0,0]@sensor_basis(request.platform).T
    point=np.array(request.platform.position_ned_m)+5000*direction
    request.scene.objects.append(request.scene.objects[0].model_copy(update={'id':'second','position_ned_m':tuple(point)}))
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    scan=np.linspace(-60,60,2401);results=[]
    for tx,label in [(1,'1Tx4Rx'),(2,'2Tx4Rx TDM-MIMO')]:
        request.array.num_tx=tx
        layout=build_layout(request.array,request.waveform,request.platform)
        factor=abs(array_factor(layout,request.waveform.wavelength_m,scan,[0]))[0]
        width=float(np.ptp(scan[factor>=1/np.sqrt(2)]))
        axes[0].plot(scan,20*np.log10(np.maximum(factor,1e-4)),label=f'{label}：3 dB 宽度 {width:.1f}°')
        frame=simulate_frame(request);p=frame.products
        r,d=np.unravel_index(np.argmax(p.rd_power),p.rd_power.shape)
        pattern=abs(p.rad[r,d,0]).astype(float)**2
        peaks,_=find_peaks(pattern,prominence=.1*pattern.max())
        axes[1].plot(p.azimuth_deg,10*np.log10(np.maximum(pattern/pattern.max(),1e-6)),label=label)
        left=pattern[(p.azimuth_deg<0)&(p.azimuth_deg>-20)].max()
        right=pattern[(p.azimuth_deg>0)&(p.azimuth_deg<20)].max()
        valley=pattern[np.argmin(abs(p.azimuth_deg))]
        results.append({'array':label,'array_factor_3db_width_deg':width,'detected_angular_peaks_deg':p.azimuth_deg[peaks].tolist(),
                        'known_target_angles_deg':[-12.,12.],'target_range_m':5000.,'target_velocity_mps':0.,
                        'central_valley_below_smaller_peak_db':float(10*np.log10(min(left,right)/valley))})
    axes[0].axhline(-3,color='gray',ls='--');axes[0].set(xlim=(-40,40),ylim=(-45,1),xlabel='方位扫描角 / °',ylabel='归一化阵列因子 / dB',title='几何增大虚拟孔径，主瓣变窄')
    for angle in [-12,12]:axes[1].axvline(angle,color='gray',ls='--')
    axes[1].set(xlim=(-32,32),ylim=(-35,1),xlabel='方位扫描角 / °',ylabel='角度切片相对功率 / dB',title='同距离、同速度的两个理想目标')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.suptitle('MIMO 功能验证：不仅增加通道，还检查角度分辨能力\n等幅、固定相位、无噪声示例；3 dB 主瓣宽度不等于任意场景的测角误差',fontsize=12)
    fig.savefig(args.output/'mimo_resolution.png',dpi=160);plt.close(fig)
    # Preserve a small separation sweep, including the weak 20-degree case.
    sweep=[]
    for separation,tx in [(s,t) for s in [16.,20.,24.,30.] for t in [1,2]]:
        case['azimuth_deg']=-separation/2
        sample=case_request(case);sample.array.num_tx=tx
        direction=scan_directions([separation/2],[0.])[0,0]@sensor_basis(sample.platform).T
        sample.scene.objects.append(sample.scene.objects[0].model_copy(update={'id':'second','position_ned_m':tuple(np.array(sample.platform.position_ned_m)+5000*direction)}))
        f=simulate_frame(sample);p=f.products
        r,d=np.unravel_index(np.argmax(p.rd_power),p.rd_power.shape)
        pattern=abs(p.rad[r,d,0]).astype(float)**2
        peaks,_=find_peaks(pattern,prominence=.1*pattern.max())
        near=peaks[abs(p.azimuth_deg[peaks])<25]
        lower=min(pattern[p.azimuth_deg<0].max(),pattern[p.azimuth_deg>0].max())
        dip=float(10*np.log10(lower/pattern[np.argmin(abs(p.azimuth_deg))]))
        sweep.append({'separation_deg':separation,'tx':tx,'rx':4,'peak_angles_deg':p.azimuth_deg[near].tolist(),
                      'central_dip_db':dip,'resolved_by_declared_rule':bool(len(near)>=2 and dip>=3)})
    write_json(args.output/'resolution_metrics.json',{'results':results,'separation_sweep':sweep,'is_real':False,
        'resolution_rule':'two local peaks, central valley at least 3 dB below the lower peak; demonstration criterion not universal Rayleigh resolution',
        'two_target_test':'equal amplitude, fixed zero reflection phase, same range and zero velocity; not a universal resolution guarantee'})
    print(json.dumps(results,ensure_ascii=False))


if __name__=='__main__':main()
