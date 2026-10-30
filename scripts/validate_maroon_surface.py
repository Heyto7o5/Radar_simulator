#!/usr/bin/env python3
"""Evaluate separately saved platform predictions against a real geometry pair.

Calibration: Tx index divisible by five. Evaluation: other Tx rows (same
static capture, NOT independent acquisitions). Only one complex scalar per
model is fitted. Geometry, materials, phase and paths are not fitted.
"""
from __future__ import annotations
import argparse,csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from radar_platform.maroon_dataset import DEFAULT_ROOT,load_measurement,geometry_and_radar
from radar_platform.surface_echo import range_response
from radar_platform.validation_metrics import fit_complex_gain,complex_metrics,profile_metrics
from radar_platform.models import C0


def style():
    f=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
    if f.exists():
        font_manager.fontManager.addfont(str(f))
        plt.rcParams['font.family']=font_manager.FontProperties(fname=str(f)).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--forward',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise SystemExit('choose new output directory')
    real,f,valid=load_measurement(args.root)
    mesh,radar,coordinates=geometry_and_radar(args.root)
    with np.load(args.forward/'forward.npz',allow_pickle=False) as z:
        simulated=z['signal']; bounce2=z['bounce_1']+z['bounce_2']
        if not np.array_equal(z['frequencies_hz'],f):raise ValueError('physical frequencies do not agree')
        if not np.allclose(z['vertices_ned_m'],mesh.vertices_m,atol=1e-12):raise ValueError('geometry provenance mismatch')
        paths={k:z[f'path_count_{k}'] for k in (1,2,3)}
    if simulated.shape!=real.shape:raise ValueError('signal axes disagree')
    # Geometry-only point baseline: the vertex common to all three faces.
    common=set(mesh.faces[0]).intersection(*map(set,mesh.faces[1:]))
    if len(common)!=1:raise ValueError('expected one common corner vertex')
    vertex=mesh.vertices_m[common.pop()]
    rt=np.linalg.norm(radar.tx_ned_m-vertex,axis=1)
    rr=np.linalg.norm(radar.rx_ned_m-vertex,axis=1)
    length=rt[:,None]+rr[None,:]
    point=(np.exp(-2j*np.pi*length[:,:,None]*f/C0)/(rt[:,None]*rr[None,:])[:,:,None]).astype(np.complex64)
    calibration=valid&(np.arange(94)[:,None]%5==0)
    evaluation=valid&~calibration
    if np.any(calibration&evaluation):raise AssertionError('split overlap')
    real_range,ranges=range_response(real,f,512)
    real_profile=np.mean(abs(real_range[evaluation])**2,axis=0)
    labels={'point':'等效点（几何顶点）','two_bounce':'最多二次反射','mesh':'三角面 + 最多三次反射'}
    all_pred={'point':point,'two_bounce':bounce2,'mesh':simulated}
    rows=[];responses={}; gains={};fitted={}
    for name,signal in all_pred.items():
        gain=fit_complex_gain(signal,real,calibration)
        pred=(gain*signal).astype(np.complex64); fitted[name]=pred
        response,_=range_response(pred,f,512);responses[name]=response
        profile=np.mean(abs(response[evaluation])**2,axis=0)
        row={'model':name,**complex_metrics(pred,real,evaluation),**profile_metrics(profile,real_profile,ranges),
             'predicted_nonzero_channels':int((np.any(abs(signal)>0,axis=-1)&evaluation).sum()),
             'evaluation_channels':int(evaluation.sum())}
        haspath=np.any(abs(signal)>0,axis=-1)&evaluation
        # Conditional per-channel peak error must always be shown with coverage.
        gate=(ranges>=.2)&(ranges<=.6)
        mpeak=ranges[gate][np.argmax(abs(real_range[haspath][:,gate])**2,axis=-1)]
        speak=ranges[gate][np.argmax(abs(response[haspath][:,gate])**2,axis=-1)]
        row['conditional_channel_peak_mae_m']=float(np.mean(abs(mpeak-speak))) if len(mpeak) else None
        gains[name]={'real':gain.real,'imag':gain.imag};rows.append(row)
    args.output.mkdir(parents=True)
    with (args.output/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as handle:
        w=csv.DictWriter(handle,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(args.output/'comparison.npz',measured_spectrum=real,
        simulated_spectrum=simulated,scaled_simulated_spectrum=fitted['mesh'],frequencies_hz=f,
        calibration_mask=calibration,evaluation_mask=evaluation,range_axis_m=ranges,
        measured_mean_power=real_profile,
        simulated_mean_power=np.mean(abs(responses['mesh'][evaluation])**2,axis=0))
    sources=['backend/radar_platform/surface_scene.py','backend/radar_platform/surface_echo.py',
             'backend/radar_platform/maroon_dataset.py','backend/radar_platform/validation_metrics.py',
             'scripts/simulate_maroon_geometry.py','scripts/validate_maroon_surface.py']
    project=Path(__file__).resolve().parents[1]
    forward_meta=json.loads((args.forward/'forward.json').read_text())
    result={'generated_utc':datetime.now(timezone.utc).isoformat(),'dataset_manifest':str(args.root/'manifest.json'),
        'source_commit':json.loads((args.root/'manifest.json').read_text())['commit'],
        'model':'independent platform scalar PEC image-path solver',
        'calibration_rule':'Tx index % 5 == 0; one global complex gain per model',
        'evaluation_rule':'Tx index % 5 != 0; same static capture, spatial channel holdout only',
        'calibration_channels':int(calibration.sum()),'evaluation_channels':int(evaluation.sum()),
        'source_zeroed_channels':int((~valid).sum()),'fitted_gains':gains,
        'material':'ideal scalar PEC, reflection coefficient -1 per reflection',
        'global_reference':'calibration receiver power only; no per-channel or per-plot peak normalization',
        'native_range_resolution_m':C0/(2*(f[-1]-f[0])),
        'range_sample_spacing_m':float(ranges[1]),'max_unambiguous_equivalent_range_m':C0/(2*(f[1]-f[0])),
        'shape':list(real.shape),'has_slow_time':False,'rows':rows,'forward':forward_meta,'coordinates':coordinates,
        'excluded_effects':['vector polarization','finite-aperture diffraction','diffuse scattering','antenna patterns','registration refitting','absolute radiometric calibration'],
        'source_hashes':{s:digest(project/s) for s in sources},'forward_sha256':digest(args.forward/'forward.npz')}
    (args.output/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    # Common dB reference from calibration channels, used unchanged for all images.
    reference=max(float(np.mean(abs(real_range[calibration])**2,axis=0).max()),1e-30)
    db=lambda power:np.maximum(10*np.log10(np.maximum(power,1e-30)/reference),-60.)
    style()
    fig=plt.figure(figsize=(16,9),layout='constrained')
    grid=fig.add_gridspec(2,3)
    ax=fig.add_subplot(grid[0,0],projection='3d')
    xyz=mesh.vertices_m[:,[1,0,2]].copy();xyz[:,2]*=-1
    coll=Poly3DCollection(xyz[mesh.faces],alpha=.55,facecolor=['#2477a9','#31a88d','#eba24a'],edgecolor='#20344c')
    ax.add_collection3d(coll)
    ax.scatter(radar.tx_ned_m[:,1],radar.tx_ned_m[:,0],-radar.tx_ned_m[:,2],s=6,c='#ba3c4a',label='94 Tx')
    ax.scatter(radar.rx_ned_m[:,1],radar.rx_ned_m[:,0],-radar.rx_ned_m[:,2],s=6,c='#2477a9',label='94 Rx')
    ax.set(xlim=(-.12,.12),ylim=(-.02,.43),zlim=(-.1,.1),xlabel='East / m',ylabel='North / m',zlabel='Up / m',title='真实 OBJ 输入：三个反射面')
    ax.view_init(25,-45);ax.legend(fontsize=8)
    ax=fig.add_subplot(grid[0,1]);ax.plot(ranges,db(real_profile),color='black',lw=2,label='实测')
    for name,color in [('point','#999999'),('mesh','#e88425')]:
        profile=np.mean(abs(responses[name][evaluation])**2,axis=0)
        ax.plot(ranges,db(profile),label=labels[name],color=color)
    ax.set(xlim=(.20,.58),ylim=(-50,5),xlabel='半双程等效距离 / m',ylabel='相对功率 / dB',title='未拟合通道：平均距离剖面');ax.legend(fontsize=8);ax.grid(alpha=.15)
    ax=fig.add_subplot(grid[0,2])
    tx_indices=np.flatnonzero(np.arange(94)%5!=0)
    channel_power=lambda x: np.array([np.mean(abs(x[t,valid[t]])**2) for t in tx_indices])
    pref=max(float(np.mean(abs(real[calibration])**2)),1e-30)
    ax.plot(tx_indices,10*np.log10(channel_power(real)/pref),c='black',label='实测')
    ax.plot(tx_indices,10*np.log10(np.maximum(channel_power(fitted['mesh']),1e-30)/pref),c='#e88425',label='平台仿真')
    ax.set(xlabel='Tx 编号（未参与校准）',ylabel='平均接收功率 / dB',title='通道能量：显示模型尚未对齐部分',ylim=(-45,12));ax.legend(fontsize=8);ax.grid(alpha=.15)
    range_mask=(ranges>=.20)&(ranges<=.58)
    def channel_image(response):
        return np.array([np.mean(abs(response[t,valid[t]][:,range_mask])**2,axis=0) for t in tx_indices])
    measured_img=db(channel_image(real_range));sim_img=db(channel_image(responses['mesh']))
    for col,arr,title,cmap,lo,hi in [(0,measured_img,'实测距离–通道图（不是 RD）','turbo',-50,0),
                                  (1,sim_img,'平台三角面仿真：相同物理轴','turbo',-50,0),
                                  (2,abs(measured_img-sim_img),'绝对 dB 差异：无逐通道归一化','magma',0,40)]:
        ax=fig.add_subplot(grid[1,col]);im=ax.imshow(arr,origin='lower',aspect='auto',extent=[ranges[range_mask][0],ranges[range_mask][-1],0,len(tx_indices)],cmap=cmap,vmin=lo,vmax=hi)
        ax.set(xlabel='半双程等效距离 / m',ylabel='评估 Tx 排序',title=title)
        fig.colorbar(im,ax=ax,label='dB',shrink=.8)
    fig.suptitle('MAROON 角反射器：输入真实几何，由我们的平台计算回波并与实测比较',fontsize=16)
    fig.savefig(args.output/'geometry_signal_comparison.png',dpi=180)
    fig.savefig(args.output/'geometry_signal_comparison.pdf');plt.close(fig)
    # Compact ablation: predictions remain evaluated on exactly the same channels.
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    for ax,key,title in zip(axes,['complex_coherence','profile_power_correlation','profile_wasserstein_m'],['复信号相关（越高越好）','平均距离剖面相关（越高越好）','全无模糊距离 W1 / m（越低越好）']):
        values=[r[key] for r in rows]
        ax.bar(range(len(rows)),values,color=['#8c9ca9','#4b90b5','#e88425'])
        ax.set_xticks(range(len(rows)),['等效点','≤2 次反射','≤3 次反射']);ax.set_title(title,fontsize=11)
        for i,v in enumerate(values):ax.text(i,v,f'{v:.3f}',ha='center',va='bottom',fontsize=10)
        ax.set_ylim(0,max(values)*1.20)
    fig.savefig(args.output/'model_ablation.png',dpi=180);plt.close(fig)
    print(json.dumps({'rows':rows,'calibration_channels':int(calibration.sum()),'evaluation_channels':int(evaluation.sum()),'output':str(args.output)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
