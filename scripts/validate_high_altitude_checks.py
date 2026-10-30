#!/usr/bin/env python3
"""Deterministic applicability/motion/visibility acceptance, not real-data validation."""
import argparse,base64,copy,csv,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radar_platform.unified.schema import Simulation
from radar_platform.unified.session import SimulationSession
from radar_platform.unified.motion_check import motion_error_report
from radar_platform.unified.applicability import inspect_configuration

ROOT=Path(__file__).resolve().parents[1]


def motion_case(kind,carrier,distance,speed,direction):
    if kind=='pulsed_lfm':
        wave=dict(kind=kind,carrier_hz=carrier,bandwidth_hz=2e6,pulse_width_s=2e-6,
                  sample_rate_hz=4e6,samples=256,adc_start_s=0,slot_period_s=100e-6,loops=8)
    else:
        wave=dict(kind=kind,start_hz=carrier,slope_hz_s=1e10,ramp_s=120e-6,
                  sample_rate_hz=2e6,samples=128,adc_start_s=45e-6,slot_period_s=150e-6,loops=8)
    velocity=[speed,0,0] if direction=='radial' else [0,speed,0]
    return Simulation.model_validate(dict(waveform=wave,
        platform=dict(position_ned_m=[0,0,0],velocity_ned_mps=velocity,boresight_ned=[1,0,0],beam_half_angle_deg=89),
        array=dict(layout='siso',num_tx=1,num_rx=1),
        scene=dict(objects=[dict(kind='point',id='target',position_ned_m=[distance,0,0])]),
        processing=dict(range_limits_m=[0,6000],azimuth_deg=[0.],elevation_deg=[0.]),sequence=dict(frame_count=1)))


def target_coverage(frame,name):
    c=frame.metadata['visibility']['coverage'];obj=c['objects'].index(name)
    i=c['object_index'].index(obj)
    return {k:c[k][i] for k in ['illuminated_tx_slots','unblocked_channel_slots','sampled_echo_channel_slots']}


def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    configs=out/'configs';configs.mkdir()
    rows=[];requests=[]
    for kind in ['pulsed_lfm','fmcw']:
        for carrier in [1e9,10e9]:
            for distance in [10.,100.,5000.]:
                for speed in [0.,300.,1200.,2500.]:
                    for direction in ['radial','transverse']:
                        r=motion_case(kind,carrier,distance,speed,direction)
                        check=motion_error_report(r,max_slots=8,fast_samples=9)
                        applicability=inspect_configuration(r.model_dump(mode='json'))
                        row=dict(case=len(rows),kind=kind,carrier_hz=carrier,distance_m=distance,
                                 speed_mps=speed,direction=direction,status=check['status'],
                                 applicability_status=applicability['status'],
                                 applicability_risks=';'.join(x['code'] for x in applicability['checks'] if x['status']!='usable'),
                                 active_sample_pairs=check['coverage']['active_sample_pairs'],**check['metrics'])
                        rows.append(row);requests.append(r)
                        dump(configs/f'motion_{row["case"]:03d}.json',r.model_dump(mode='json'))
    with (out/'motion_sweep.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    worst=max(rows,key=lambda x:x['max_unwrapped_phase_error_rad'])
    convergence=[dict(fast_samples=n,**motion_error_report(requests[worst['case']],max_slots=8,fast_samples=n)['metrics']) for n in [2,5,9,33,128]]
    dump(out/'motion_refinement.json',{'case':worst,'refinement':convergence,
        'scope':'nested endpoint-inclusive fast-time checks; not a bound on other trajectories'})
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for kind in ['pulsed_lfm','fmcw']:
        selected=[r for r in rows if r['kind']==kind and r['carrier_hz']==10e9 and r['speed_mps']==1200 and r['direction']=='transverse']
        axes[0].loglog([r['distance_m'] for r in selected],[max(r['max_unwrapped_phase_error_rad'],1e-12) for r in selected],'o-',label=kind)
    axes[0].axhline(np.pi/8,color='r',ls='--',label='pi/8 screening threshold')
    axes[0].set(xlabel='Slant range (m)',ylabel='Sampled maximum phase error (rad)',title='1200 m/s transverse motion, 10 GHz');axes[0].legend()
    axes[1].plot([r['fast_samples'] for r in convergence],[r['max_unwrapped_phase_error_rad'] for r in convergence],'o-')
    axes[1].set(xlabel='Fast-time samples checked',ylabel='Maximum phase error (rad)',title='Worst sampled case: refinement')
    fig.tight_layout();fig.savefig(out/'motion.png',dpi=150);plt.close(fig)
    base=json.loads((ROOT/'configs/unified/high_altitude_ridge_visibility.json').read_text())
    coverage=[];main_off=main_on=None
    for height in [0,100,300,600,900]:
        for speed in [0,1200]:
            raw=copy.deepcopy(base)
            raw['scene']['terrain']['elevation_up_m']=[[0,0,0,height,0,0]]*3
            raw['platform']['velocity_ned_mps']=(np.array(base['platform']['velocity_ned_mps'])*speed/1200).tolist()
            session=SimulationSession(Simulation.model_validate(raw));frame=session.frame(0)
            directory=out/f'ridge_h{height}_v{speed}';session.save_inputs(directory);session.save_frame(directory,frame)
            front=target_coverage(frame,'front');behind=target_coverage(frame,'behind')
            # At t=0 the back-target ray crosses the ridge apex at z=-384.615 m.
            expected_clear=height<5000*(1-1200/1300)
            assert front['unblocked_channel_slots']==32
            assert behind['unblocked_channel_slots']==(32 if expected_clear else 0)
            coverage.append(dict(height_m=height,speed_mps=speed,front=front,behind=behind,passed=True))
            if height==600 and speed==1200:
                main_on=frame
                raw['visibility']['occlusion_enabled']=False
                off_session=SimulationSession(Simulation.model_validate(raw));main_off=off_session.frame(0)
                off_session.save_inputs(out/'ridge_disabled');off_session.save_frame(out/'ridge_disabled',main_off)
    dump(out/'coverage_cases.json',coverage)
    p=main_on.products;q=main_off.products;ref=max(float(q.rd_power.max()),1e-30)
    off_db=10*np.log10(np.maximum(q.rd_power/ref,1e-6));on_db=10*np.log10(np.maximum(p.rd_power/ref,1e-6))
    c=main_on.metadata['visibility']['coverage'];points=np.array(c['positions_at_frame_start_ned_m'])
    beam=np.array(c['illuminated_tx_slots'])>0;clear=np.array(c['unblocked_channel_slots'])>0
    fig,ax=plt.subplots(2,3,figsize=(15,8))
    n=np.array(base['scene']['terrain']['north_m']);z=np.array(base['scene']['terrain']['elevation_up_m'][0])
    ax[0,0].plot(n,z,'k-',label='Ideal ridge');ax[0,0].scatter([0],[5000],marker='^',label='Radar')
    for name,x in [('front',1000),('behind',1300)]:
        ax[0,0].plot([0,x],[5000,0],'--',label=name)
    ax[0,0].set(xlabel='North (m)',ylabel='Height (m)',title='Input geometry: synthetic, not real DEM');ax[0,0].legend()
    for mask,color,label in [(~beam,'gray','outside beam'),(beam&clear,'tab:blue','LOS clear'),(beam&~clear,'tab:red','blocked')]:
        ax[0,1].scatter(points[mask,1],points[mask,0],c=color,label=label)
    ax[0,1].set(xlabel='East (m)',ylabel='North (m)',title='Sample coverage, not continuous area');ax[0,1].legend()
    ax[0,2].axis('off');ax[0,2].text(0,.9,'5 km height; 1200 m/s; 1 GHz\n\nFront target: 32 / 32 clear slots\nBehind ridge: 0 / 32 clear slots\n\nOpaque surfaces, no diffraction\nVisibility frozen within each Tx slot\nRD panels share one power reference',va='top')
    dr=p.range_m[1]-p.range_m[0];dv=p.velocity_mps[1]-p.velocity_mps[0]
    extent=[p.range_m[0]-dr/2,p.range_m[-1]+dr/2,p.velocity_mps[0]-dv/2,p.velocity_mps[-1]+dv/2]
    for a,data,title in zip(ax[1],[off_db,on_db,on_db-off_db],['Occlusion OFF (dB)','Occlusion ON (dB)','ON minus OFF (dB)']):
        diff='minus' in title
        im=a.imshow(data.T,origin='lower',aspect='auto',extent=extent,cmap='RdBu_r' if diff else 'viridis',vmin=-60,vmax=60 if diff else 0)
        a.set(xlabel='Range (m)',ylabel='Apparent closing velocity (m/s)',title=title);fig.colorbar(im,ax=a)
    fig.tight_layout();fig.savefig(out/'coverage_rd.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for a,data,title in zip(axes,[off_db,on_db,on_db-off_db],['Occlusion OFF: target region','Occlusion ON: target region','ON minus OFF: target region']):
        diff='minus' in title
        im=a.imshow(data.T,origin='lower',aspect='auto',extent=extent,cmap='RdBu_r' if diff else 'viridis',vmin=-60,vmax=60 if diff else 0)
        a.set(xlim=(5000,5250),ylim=(900,1450),xlabel='Range (m)',ylabel='Apparent closing velocity (m/s)',title=title)
        fig.colorbar(im,ax=a)
    fig.tight_layout();fig.savefig(out/'rd_zoom.png',dpi=150);plt.close(fig)
    raw=json.loads((ROOT/'configs/unified/high_altitude_1ghz.json').read_text())
    default_check=inspect_configuration(raw);default_check['motion']=motion_error_report(Simulation.model_validate(raw))
    dump(out/'default_parameter_check.json',default_check)
    regressions=[]
    oldroot=ROOT/'.runtime/consolidation_20261016/acceptance_matrix'
    for name in ['high_altitude_1ghz','fmcw_siso','fmcw_simo','fmcw_mimo','moving_plate_po','terrain_flat','terrain_hills','terrain_dem']:
        folder=oldroot/name
        if not (folder/'request.json').exists():
            regressions.append(dict(case=name,available=False));continue
        request=Simulation.model_validate_json((folder/'request.json').read_text());current=SimulationSession(request).frame(0)
        meta=json.loads((folder/'frame_0000/metadata.json').read_text())
        spec=meta['products']['adc'];old=np.fromfile(folder/'frame_0000'/spec['file'],dtype=spec['dtype']).reshape(spec['shape'])
        error=float(np.max(abs(current.adc-old)));assert error==0,name
        regressions.append(dict(case=name,available=True,adc_max_absolute_difference=error))
    summary={'motion_parameter_cases':len(rows),'motion_cases_over_phase_threshold':sum(r['max_unwrapped_phase_error_rad']>np.pi/8 for r in rows),
        'coverage_parameter_cases':len(coverage),'all_coverage_checks_passed':True,'disabled_regression':regressions,
        'scope':'synthetic analytic and integration checks; not real-data validation',
        'limitations':['No retarded propagation reference','Sampled trajectory/path diagnostics, not rigorous extrema bounds',
                       'Opaque per-Tx-slot visibility; no within-slot shadow motion, diffraction or environmental multipath',
                       'Brute-force ray/triangle budget limits large scenes; no GPU/BVH acceleration yet']}
    dump(out/'summary.json',summary)
    sources=[*sorted((ROOT/'backend/radar_platform').rglob('*.py')),Path(__file__).resolve(),ROOT/'configs/unified/high_altitude_ridge_visibility.json']
    dump(out/'source_sha256.json',{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    images=''.join('<img style="width:100%" src="data:image/png;base64,'+base64.b64encode((out/name).read_bytes()).decode()+'">' for name in ['motion.png','coverage_rd.png','rd_zoom.png'])
    text=f'<h1>参数适用性、运动误差、地形遮挡与覆盖</h1><p>{len(rows)} 组理想运动参数检查；{len(coverage)} 组山脊高度/速度检查。不是独立实测场景，也不是高空保真证明。</p><p>默认 1 GHz、1200 m/s 无 Doppler 折叠，但 CPI 距离走动约 4.61 bin。运动图对比的是时隙内线性路径与配置轨迹的瞬时距离，两者都未考虑传播迟滞。π/8 为工程筛查阈值。</p><p>遮挡图使用显式生成的陡峭山脊测试几何，并非真实 DEM。关闭遮挡与开启遮挡的 RD 使用同一功率参考，差值为开启减关闭；不是实测与仿真的差值。</p>'
    page='<html lang="zh"><meta charset="utf-8"><title>高空雷达检查</title><body style="max-width:1400px;margin:30px auto;font:16px sans-serif">'+text+images+'<p>最后一行是固定目标区域放大图，仅显示 5000–5250 m、900–1450 m/s，完整 RD 仍在上一行。山脊演示保持 1200 m/s，采用 1 GHz、20 kHz PRF、10 μs 脉宽，让目标离速度边界更远；它不同于默认参数检查用的配置，原默认配置未修改。原生速度单元约 93.7 m/s，峰值只能落在离散单元上。</p><p>96 组运动检查衡量几何近似，不保证每组都适合实际采集。CSV 同时保存接收窗、拍频、模糊等适用性风险。每组配置、CSV、原生复数数据和 coverage.json 均保留在结果目录。关闭遮挡的历史 ADC 回归结果见 summary.json。</p></body></html>'
    (out/'report.html').write_text(page)
    print(json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
