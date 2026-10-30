#!/usr/bin/env python3
"""Illustrate what angle adds: two ideal targets sharing range and velocity."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from radar_platform.fmcw import FMCWConfig
from radar_platform.array_fmcw import TDMArray,simulate_motion_adc,process_array_adc
from radar_platform.models import TargetConfig


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    cfg=FMCWConfig(76.5e9,77.5e9,50e-6,60e-6,128,64,4e6,2e-6);array=TDMArray.ideal_ula(cfg,2,4)
    targets=[]
    for i,angle in enumerate([-20.,20.]):
        direction=np.array([np.cos(np.deg2rad(angle)),np.sin(np.deg2rad(angle)),0])
        targets.append(TargetConfig(id=f'T{i+1}',position_ned_m=tuple(10*direction),velocity_ned_mps=tuple(-direction)))
    adc=simulate_motion_adc(cfg,targets,array);result=process_array_adc(adc,cfg,array)
    rd=np.mean(abs(result.rd_channels)**2,axis=(2,3));ra=np.mean(abs(result.rad)**2,axis=1)
    r,d=np.unravel_index(np.argmax(rd),rd.shape);cut=abs(result.rad[r,d])**2
    reference=float(max(rd.max(),ra.max(),cut.max()))
    db=lambda x:10*np.log10(np.maximum(x/reference,1e-6))
    np.savez_compressed(a.output/'two_target_complex.npz',adc=adc,rd=result.rd_channels,rad=result.rad,
                         range_m=result.range_m,velocity_mps=result.velocity_mps,azimuth_deg=result.azimuth_deg)
    (a.output/'request.json').write_text(json.dumps({'is_real':False,'radar':cfg.__dict__,
        'targets':[t.model_dump(mode='json') for t in targets],'tx':array.tx_ned_m.tolist(),'rx':array.rx_ned_m.tolist(),
        'scope':'two ideal point targets, same range and speed; not measured hardware validation'},indent=2)+'\n')
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':10})
    fig,axes=plt.subplots(1,3,figsize=(15,4.2),layout='constrained')
    im=axes[0].imshow(db(rd.T),origin='lower',aspect='auto',extent=[result.range_m[0],result.range_m[-1],result.velocity_mps[0],result.velocity_mps[-1]],vmin=-60,vmax=0,cmap='turbo')
    axes[0].set(xlim=(7,13),ylim=(-3,3),xlabel='距离 / m',ylabel='接近速度 / (m/s)',title='RD：同距离、同速度难以分开')
    axes[1].imshow(db(ra.T),origin='lower',aspect='auto',extent=[result.range_m[0],result.range_m[-1],result.azimuth_deg[0],result.azimuth_deg[-1]],vmin=-60,vmax=0,cmap='turbo')
    axes[1].set(xlim=(7,13),xlabel='距离 / m',ylabel='方位扫描角 / °',title='RA：增加角度后出现两个能量峰')
    axes[2].plot(result.azimuth_deg,db(cut),c='#d87526')
    for angle in [-20,20]:axes[2].axvline(angle,ls='--',c='gray')
    axes[2].set(xlabel='方位扫描角 / °',ylabel='共用参考功率 / dB',ylim=(-40,3),title='同一距离/速度单元的角度切片')
    axes[2].grid(alpha=.2);fig.colorbar(im,ax=axes[:2].tolist(),label='dB',shrink=.7)
    fig.suptitle('理想 2Tx4Rx：两个目标均约 10 m、接近速度 1 m/s，方位 ±20°\n仅说明新增角度维的作用，不是真实场景保真或角分辨率结论')
    fig.savefig(a.output/'array_angle_demo.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
