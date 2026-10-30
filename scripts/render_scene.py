#!/usr/bin/env python3
"""Render saved unified products without re-running simulation or launching a UI."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from radar_platform.unified.schema import Simulation
from radar_platform.unified.geometry import build_layout
from radar_platform.unified.processing import array_factor


def configure_font():
    path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    if Path(path).exists():
        font_manager.fontManager.addfont(path)
        plt.rcParams['font.family']=font_manager.FontProperties(fname=path).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':9})


def load_product(root,meta,name):
    item=meta['products'][name]
    path=root/f'frame_{meta["frame_index"]:04d}'/item['file']
    return np.memmap(path,dtype=item['dtype'],mode='r',shape=tuple(item['shape']))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    configure_font()
    request=Simulation.model_validate_json((args.run/'request.json').read_text())
    scene=json.loads((args.run/'scene.json').read_text())
    frames=[json.loads(p.read_text()) for p in sorted(args.run.glob('frame_*/metadata.json'))]
    if not frames:raise ValueError('no completed frames to render')
    scatterers=np.load(args.run/'scatterers.npz',allow_pickle=False)
    positions=scatterers['position_ned_m']
    objects=scatterers['object_index']
    terrain_path=args.run/'terrain.json'
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    if terrain_path.exists():
        terrain=json.loads(terrain_path.read_text())
        im=axes[0].pcolormesh(terrain['north_m'],terrain['east_m'],terrain['elevation_up_m'],shading='auto',cmap='terrain')
        fig.colorbar(im,ax=axes[0],label='地形高程 / m')
    for index,descriptor in enumerate(scene['objects']):
        mask=objects==index
        label={'terrain':'地形散射点','building':'简化建筑','moving_target':'运动点目标','reference_plate':'参考板'}.get(descriptor['id'],descriptor['id'])
        axes[0].scatter(positions[mask,0],positions[mask,1],s=6 if descriptor['kind'] in ('flat','hills','dem') else 14,label=label)
        axes[1].scatter(positions[mask,0],-positions[mask,2],s=6)
        if descriptor['kind'] not in ('flat','hills','dem'):
            center=positions[mask].mean(axis=0)
            axes[0].annotate(label,center[:2],fontsize=8)
    p=np.array(request.platform.position_ned_m)
    direction=np.array(request.platform.boresight_ned);direction/=np.linalg.norm(direction)
    endpoint=p+direction*(-p[2]/direction[2]) if direction[2]>0 else p+direction*5000
    axes[1].scatter(p[0],-p[2],marker='v',s=75,label='雷达')
    axes[1].plot([p[0],endpoint[0]],[-p[2],-endpoint[2]],'--',label='波束中心方向')
    axes[0].set(xlabel='North / m',ylabel='East / m',title='同一世界场景：地形 + 建筑 + 参考板 + 运动点')
    axes[0].set_aspect('equal');axes[0].legend(fontsize=7,loc='upper left')
    axes[1].set(xlabel='North / m',ylabel='高程 / m',title='高空侧视（横纵等比例）')
    axes[1].set_aspect('equal',adjustable='box')
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.suptitle('通用场景输入，不使用实测 RD 拟合目标；面目标为散射点近似')
    fig.savefig(args.output/'scene.png',dpi=160);plt.close(fig)

    selected=frames if len(frames)<=4 else [frames[i] for i in np.linspace(0,len(frames)-1,4,dtype=int)]
    reference=max(float(load_product(args.run,frames[0],'rd_power').max()),1e-30)
    fig,axes=plt.subplots(1,len(selected),figsize=(4*len(selected),4.3),layout='constrained',squeeze=False)
    for axis,meta in zip(axes.ravel(),selected):
        power=np.array(load_product(args.run,meta,'rd_power'))
        r,v=np.array(meta['axes']['range_m']),np.array(meta['axes']['velocity_mps'])
        db=10*np.log10(np.maximum(power/reference,1e-6))
        im=axis.imshow(db.T,origin='lower',aspect='auto',extent=[r[0],r[-1],v[0],v[-1]],cmap='turbo',vmin=-60,vmax=6)
        axis.set(xlabel='斜距 / m',ylabel='接近速度 / (m/s)',title=f"t={meta['frame_start_s']:.1f}s · {meta['elapsed_s']:.1f}s/帧")
    fig.colorbar(im,ax=axes.ravel().tolist(),label='相对第一帧峰值 / dB',shrink=.8)
    fig.suptitle(f"10 GHz 高空组合场景 · {request.waveform.kind} · {request.array.num_tx}Tx{request.array.num_rx}Rx\n纯仿真；各帧共用功率参考；没有真实数据保真声明",fontsize=12)
    fig.savefig(args.output/'rd_sequence.png',dpi=160);plt.close(fig)

    first=frames[0];rad=load_product(args.run,first,'rad');rd=load_product(args.run,first,'rd_power')
    r,d=np.unravel_index(np.argmax(rd),rd.shape)
    az=np.array(first['axes']['azimuth_deg']);el=np.array(first['axes']['elevation_deg'])
    power=abs(np.asarray(rad[r,d],dtype=np.complex128))**2
    angular=10*np.log10(np.maximum(power/max(float(power.max()),1e-30),1e-6))
    layout=build_layout(request.array,request.waveform,request.platform)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    pos=layout.virtual_local_m
    axes[0].scatter(layout.tx_local_m[:,1],-layout.tx_local_m[:,2],marker='^',s=90,label='Tx',alpha=.6)
    axes[0].scatter(layout.rx_local_m[:,1],-layout.rx_local_m[:,2],marker='s',s=55,label='Rx',alpha=.6)
    axes[0].scatter(pos[:,1],-pos[:,2],marker='x',s=45,label='虚拟位置 Tx+Rx')
    axes[0].set(xlabel='雷达局部右向 / m',ylabel='雷达局部上向 / m',title='阵列几何与通道映射');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    if len(el)>1 and len(az)>1:
        im=axes[1].imshow(angular,origin='lower',aspect='auto',extent=[az[0],az[-1],el[0],el[-1]],vmin=-40,vmax=0,cmap='turbo')
        axes[1].set(xlabel='局部方位角 / °',ylabel='局部俯仰角 / °')
        fig.colorbar(im,ax=axes[1],label='该切片相对峰值 / dB')
    else:
        axes[1].plot(az,angular[0]);axes[1].set(xlabel='局部方位角 / °',ylabel='该切片相对功率 / dB')
    axes[1].set_title(f"最强 RD 单元角度切片\nR={first['axes']['range_m'][r]:.1f}m, v={first['axes']['velocity_mps'][d]:.2f}m/s")
    fig.suptitle('URA 支持两维角度；扫描格点不等于角分辨率，混合目标切片不等于单目标真值')
    fig.savefig(args.output/'array_and_angle.png',dpi=160);plt.close(fig)
    print(json.dumps({'output':str(args.output),'rendered_frames':len(selected),'run':str(args.run)},ensure_ascii=False))


if __name__=='__main__':main()
