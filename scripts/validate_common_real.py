#!/usr/bin/env python3
"""Re-evaluate existing real data using one common-reference MAE definition."""
import argparse,csv,hashlib,json,os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from radar_platform.comparison import compare_power,average_rd_power,representative_indices
from radar_platform.repeat_metrics import distribution,common_safe_gates
from radar_platform.surface_echo import range_response
from radar_platform.distributed_clutter import ClutterRecipe,simulate_clutter_iq,rd_blocks
from radar_platform.ipix_dataset import load_ipix,IQCorrection
from radar_platform.physical_optics import sfcw_range_profile,range_gate
from radar_platform.maqu_dataset import load_background_subtracted
from radar_platform.ramp_dataset import load_real_frame,load_labels
from radar_platform.ramp_comparison import SimulationTuning,simulate_label_conditioned_adc,official_rv_slices,official_range_axis_m,official_velocity_axis_mps

DATA=Path('/nas/datasets/hyt/datasets/radar_open')


def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def save_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def plot_curves(output,cases,title):
    fig,axes=plt.subplots(len(cases),3,figsize=(14,3*len(cases)),layout='constrained',squeeze=False)
    for row,case in zip(axes,cases):
        comp=case['comparison'];x=case['axis'];roi=case.get('plot_roi',np.ones(len(x),dtype=bool))
        row[0].plot(x[roi],comp.measured_db[roi],c='black');row[1].plot(x[roi],comp.simulated_db[roi],c='#df7b25')
        row[2].plot(x[roi],comp.difference_db[roi],c='#517f99');row[2].axhline(0,c='gray',lw=.8)
        for ax in row:ax.set(xlabel='距离 / m',ylabel='dB');ax.grid(alpha=.2)
        row[0].set_title(case['name']+'：实测');row[1].set_title('独立仿真（冻结校准）')
        row[2].set_title(f"仿真−实测；全域 MAE={comp.metrics['mae_db']:.2f} dB")
        common_min=min(float(comp.measured_db[roi].min()),float(comp.simulated_db[roi].min()))
        common_max=max(float(comp.measured_db[roi].max()),float(comp.simulated_db[roi].max()))
        for ax in row[:2]:ax.set_ylim(common_min-2,common_max+2)
    fig.suptitle(title);fig.savefig(output,dpi=155);plt.close(fig)


def plot_maps(output,cases,title):
    fig,axes=plt.subplots(len(cases),3,figsize=(14,3.15*len(cases)),layout='constrained',squeeze=False)
    for row,case in zip(axes,cases):
        comp=case['comparison'];mask=case.get('mask',np.ones(comp.measured_db.shape,dtype=bool))
        for j,(array,label) in enumerate([(comp.measured_db,'实测'),(comp.simulated_db,'仿真'),(comp.difference_db,'仿真−实测')]):
            arr=np.where(mask,array,np.nan)
            im=row[j].imshow(arr,origin='lower',aspect='auto',extent=case['extent'],cmap='RdBu_r' if j==2 else 'turbo',
                             vmin=-30 if j==2 else -60,vmax=30 if j==2 else 6,interpolation='nearest')
            row[j].set(xlabel='距离 / m',ylabel=case['ylabel'],title=(case['name']+'：'+label) if j<2 else f"MAE={comp.metrics['mae_db']:.2f} dB")
            if case.get('ylim'):row[j].set_ylim(case['ylim'])
            fig.colorbar(im,ax=row[j],label='dB',shrink=.75)
    fig.suptitle(title);fig.savefig(output,dpi=155);plt.close(fig)


def maroon(output):
    root=DATA/'maroon_corner/processed/validation_20260929_final'
    with np.load(root/'comparison.npz') as z:
        real=z['measured_spectrum'];sim=z['scaled_simulated_spectrum'];f=z['frequencies_hz'];cal=z['calibration_mask'];ev=z['evaluation_mask']
    measured,axis=range_response(real,f);predicted,_=range_response(sim,f)
    rp,sp=abs(measured)**2,abs(predicted)**2
    reference=float(rp[cal].mean(axis=0).max());roi=(axis>=.2)&(axis<=.6)
    rows=[]
    for tx,rx in zip(*np.where(ev)):
        comparison=compare_power(rp[tx,rx],sp[tx,rx],reference_power=reference,target_mask=roi)
        rows.append({'case_id':f'tx{tx:02d}_rx{rx:02d}','tx':int(tx),'rx':int(rx),
                     'nonzero_prediction':bool(sp[tx,rx].sum()>0),**comparison.metrics})
    save_csv(output/'maroon_channel_mae.csv',rows)
    cases=[]
    for index in representative_indices([r['mae_db'] for r in rows]):
        r=rows[index];tx,rx=r['tx'],r['rx']
        cases.append({'name':r['case_id'],'axis':axis,'plot_roi':roi,
                      'comparison':compare_power(rp[tx,rx],sp[tx,rx],reference_power=reference,target_mask=roi)})
    plot_curves(output/'maroon_common_mae.png',cases,'MAROON 既有小样：静态距离响应，不是 RD；3 个固定通道 + 最大误差通道')
    return {'rows':len(rows),'independent_captures':1,'calibration_channels':int(cal.sum()),
            'mae_db':distribution([r['mae_db'] for r in rows]),'target_mae_db':distribution([r['target_mae_db'] for r in rows]),
            'nonzero_prediction_fraction':sum(r['nonzero_prediction'] for r in rows)/len(rows),
            'reference_power':reference,'selected_cases':[x['name'] for x in cases],
            'scope':'same static averaged capture, existing frozen scalar gain; all zero predictions retained'}


def ipix(output):
    root=DATA/'ipix_dartmouth';frozen=json.loads((root/'processed/validation_20260930_final/recipe.json').read_text())
    cfg=ClutterRecipe(**frozen['model']);corr=IQCorrection(**frozen['iq_correction'])
    first=load_ipix(root/'raw/19931107_135603_starea.cdf');cal=corr.apply(first.iq[:32768])
    calrd,f=rd_blocks(cal,cfg.prf_hz);safe=common_safe_gates(14,[8,9,10,11],[8,9,10,11])
    reference=float(average_rd_power(calrd)[:,safe].max())
    specs=[(17,'19931107_135603_starea.cdf',65536,[8,9,10,11]),(18,'19931107_141630_starea.cdf',0,[8,9,10,11]),
           (19,'19931107_145028_starea.cdf',0,[7,8,9]),(25,'19931108_213827_starea.cdf',0,[6,7,8])]
    rows=[];cases=[];record_summaries=[];simulation_cache={}
    for rid,file,start,excluded in specs:
        rec=load_ipix(root/'raw'/file);iq=corr.apply(rec.iq[start:]);n=len(iq)
        same=np.allclose(rec.range_near_edge_m,cfg.range_near_edge_m,atol=.002,rtol=0)
        model=cfg if same else replace(cfg,range_near_edge_m=tuple(rec.range_near_edge_m),mean_power=tuple(np.ones(14)))
        mask=common_safe_gates(14,[8,9,10,11],excluded)
        key=(n,same)
        if key not in simulation_cache:
            def seed_power(seed):
                rd,_=rd_blocks(simulate_clutter_iq(model,n,seed),cfg.prf_hz)
                return np.mean(abs(rd.reshape(-1,16,512,14).astype(np.complex128))**2,axis=1)
            with ThreadPoolExecutor(max_workers=4) as pool:powers=list(pool.map(seed_power,range(42,50)))
            simulation_cache[key]=np.mean(powers,axis=0)
        sim=simulation_cache[key];rd,_=rd_blocks(iq,cfg.prf_hz)
        real=np.mean(abs(rd.reshape(-1,16,512,14).astype(np.complex128))**2,axis=1)
        valid=np.broadcast_to(mask,(512,14));current=[]
        np.savez_compressed(output/f'ipix_{rid}_mean_rd.npz',measured_power=real,simulated_power=sim,valid_mask=valid,
                            range_m=rec.range_near_edge_m,doppler_hz=f,reference_power=reference,is_same_range=same)
        for w,(m,s) in enumerate(zip(real,sim)):
            # Farther range has no calibrated range-power law; amplitude is NOT scored.
            if not same:m=m/m[valid].mean();s=s/s[valid].mean()
            comp=compare_power(m,s,reference_power=reference if same else 1.,valid_mask=valid)
            name=f'#{rid} 窗 {w}'+('（仅形状）' if not same else '')
            row={'record_id':rid,'window':w,'start_s':(start+w*8192)/cfg.prf_hz,'cpi_count':16,
                 'comparison_mode':'common_calibration_power' if same else 'unit_mean_shape_only',**comp.metrics}
            rows.append(row);current.append(row)
            cases.append({'name':name,'comparison':comp,'mask':valid,'extent':[rec.range_near_edge_m[0],rec.range_near_edge_m[-1],f[0],f[-1]],
                          'ylabel':'Doppler / Hz','ylim':(-150,150)})
        record_summaries.append({'record_id':rid,'windows':len(real),'measured_cpis':len(rd),'comparison_mode':current[0]['comparison_mode'],
                                 'mae_db':distribution([r['mae_db'] for r in current]),'receiver_controls':rec.metadata['receiver_controls_volts']})
        print('IPIX common-reference',rid,len(real),'windows',flush=True)
    save_csv(output/'ipix_window_mae.csv',rows)
    # Fixed three positions can illustrate both modes, but the added worst case
    # must not rank shape-only scores against amplitude-preserving scores.
    indices=representative_indices([r['mae_db'] if r['comparison_mode']=='common_calibration_power' else -1. for r in rows])
    plot_maps(output/'ipix_common_mae.png',[cases[i] for i in indices],
              'IPIX：16 CPI 线性功率平均；仿真取 8 个种子均值\n没有逐图峰值归一化；#19 若出现则仅比较单位均值谱形')
    return {'records':record_summaries,'real_cpis':sum(r['measured_cpis'] for r in record_summaries),'windows':len(rows),
            'seeds':list(range(42,50)),'reference_power':reference,'selected_cases':[cases[i]['name'] for i in indices],
            'scope':'statistical averages, not time-synchronized sea wave prediction; record #19 shape-only is not an amplitude score'}


def maqu(output):
    root=DATA/'maqu_scatterometer';saved=root/'processed/validation_20260930_final'
    recipe=json.loads((saved/'recipe.json').read_text());data=np.load(saved/'comparison_arrays.npz')
    f,reference,_=load_background_subtracted(root,'20180815',13,42)
    rows=[];cases=[]
    for pol in ('11','22'):
        delay=recipe['instrument_equivalent_range_offset_m'][pol]
        gate=range_gate(reference[pol],f,27.7+delay,.65,.2)
        profile,_=sfcw_range_profile(gate,f);pref=float(np.max(abs(profile)**2))
        for config in recipe['validation_geometries']:
            key=f"{config['name']}_{pol}_";m=abs(data[key+'measured_profile'])**2;s=abs(data[key+'simulated_profile'])**2;axis=data[key+'range_m']
            roi=abs(axis-config['range_m'])<1
            comp=compare_power(m,s,reference_power=pref,target_mask=roi)
            name=config['name']+(' VV' if pol=='11' else ' HH')
            rows.append({'case_id':name,'polarization':pol,**comp.metrics})
            cases.append({'name':name,'axis':axis,'plot_roi':roi,'comparison':comp})
    data.close();save_csv(output/'maqu_mae.csv',rows)
    indices=representative_indices([r['mae_db'] for r in rows])
    plot_curves(output/'maqu_common_mae.png',[cases[i] for i in indices],'玛曲金属板：全带宽静态距离响应；统一参考板功率，不重对齐测试峰')
    return {'validation_measurements':3,'polarization_comparisons':len(rows),'mae_db':distribution([r['mae_db'] for r in rows]),
            'target_mae_db':distribution([r['target_mae_db'] for r in rows]),'selected_cases':[cases[i]['name'] for i in indices],
            'scope':'existing frozen six-parameter calibration; three geometries/sessions, not six independent scenes'}


def ramp(output):
    os.environ['RADAR_DATA_ROOT']=str(DATA)
    base=DATA/'ramp_adc_77ghz/extracted'
    frames=sorted(base.rglob('radar_raw_frame/*.mat'))
    if not frames:raise ValueError('no real RAMP ADC available')
    prior=json.loads((DATA/'ramp_adc_77ghz/processed/comparison_20260923T073129Z/comparison.json').read_text())
    calibration=set(prior['calibration_frames'])
    tuning=SimulationTuning(name='ideal_geometry_only',noise_std=0.,scatterers_per_target=1)
    prepared=[];calreal=[];calsim=[]
    for index,path in enumerate(frames):
        frame_id=str(path.relative_to(base));adc,_=load_real_frame(frame_id);labels=load_labels(path)
        simulated=simulate_label_conditioned_adc(labels,tuning,42+index)
        # Official RV is mean channel amplitude. Square it consistently for power.
        measured=official_rv_slices(adc).astype(float)**2;predicted=official_rv_slices(simulated).astype(float)**2
        if frame_id in calibration:calreal.append(measured);calsim.append(predicted)
        prepared.append((frame_id,labels,measured,predicted))
    if not calreal:raise ValueError('frozen calibration list missing')
    # One global power scale determined only on existing calibration frames.
    power_gain=float(np.sum(calreal)/np.sum(calsim));reference=float(np.max(np.mean(calreal,axis=(0,1))))
    dump(output/'ramp_calibration.json',{'frame_ids':sorted(calibration),'new_ideal_model':tuning.to_dict(),'global_power_gain':power_gain,
                                       'reference_power':reference,'no_per_frame_fit':True,'status':'development re-evaluation, not final blind test'})
    raxis,vaxis=official_range_axis_m(),official_velocity_axis_mps();rows=[];cases=[]
    for frame_id,labels,measured,predicted in prepared:
        for slice_index in range(2):
            roi=np.zeros(measured[slice_index].shape,dtype=bool)
            for label in labels:
                r=float(label['range_m']);v=label.get('radial_velocity_mps')
                # Geometry annotation is evaluation-only. Missing speed labels do
                # not become a fabricated exact zero-speed ROI.
                distance_band=abs(raxis-r)<=max(1.,float(label.get('length_m',1.))/2)
                speed_band=np.ones(len(vaxis),dtype=bool) if v is None else abs(vaxis-float(v))<=1.
                roi|=distance_band[:,None]&speed_band[None,:]
            comp=compare_power(measured[slice_index],predicted[slice_index]*power_gain,reference_power=reference,target_mask=roi)
            role='calibration' if frame_id in calibration else 'development_holdout_same_record'
            rows.append({'frame_id':frame_id,'slice':slice_index,'role':role,'labels':len(labels),**comp.metrics})
            if role!='calibration':
                transposed=compare_power(measured[slice_index].T,(predicted[slice_index]*power_gain).T,reference_power=reference,target_mask=roi.T)
                cases.append({'name':Path(frame_id).stem+f' 段 {slice_index}','comparison':transposed,'extent':[raxis[0],raxis[-1],vaxis[0],vaxis[-1]],'ylabel':'表观速度 / (m/s)'})
    save_csv(output/'ramp_frame_mae.csv',rows)
    scored=[r for r in rows if r['role']!='calibration']
    indices=representative_indices([r['mae_db'] for r in scored])
    plot_maps(output/'ramp_common_mae.png',[cases[i] for i in indices],
              'RAMP：真实 ADC → RD 与标签点目标仿真\n共用一个校准期功率比例；不含真实汽车表面/材料，属于开发对比')
    return {'real_frames':len(frames),'calibration_frames':len(calreal),'development_frames':len(frames)-len(calreal),
            'overlapping_rv_slices':len(rows),'independent_recordings':len(set(str(p.parent) for p in frames)),
            'mae_db':distribution([r['mae_db'] for r in scored]),'target_mae_db':distribution([r['target_mae_db'] for r in scored]),
            'scope':'label-conditioned ideal points; speed from adjacent labels is an estimate, raw Tx/Rx geometry calibration pending',
            'selected_cases':[cases[i]['name'] for i in indices]}


def tsms_audit(output):
    from radar_platform.tsms_validation import load_processed_rd
    root=DATA/'tsms_drone_sample';rows=[]
    for path in sorted((root/'extracted').rglob('*.mat')):
        try:
            rd=load_processed_rd(path)
            rows.append({'file':str(path.relative_to(root)),'rows':rd.shape[0],'cols':rd.shape[1],
                         'finite':bool(np.isfinite(rd).all()),'eligible_for_strict_physical_mae':False,
                         'reason':'batch configuration / velocity axis not confirmed against representative raw sample'})
        except Exception as e:rows.append({'file':str(path.relative_to(root)),'rows':0,'cols':0,'finite':False,'eligible_for_strict_physical_mae':False,'reason':str(e)})
    save_csv(output/'tsms_eligibility.csv',rows)
    return {'processed_frames_audited':len(rows),'finite_frames':sum(r['finite'] for r in rows),'strict_physical_mae_eligible':0,
            'scope':'all local frames audited, not claimed as independently validated truth'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':9})
    result={'metric_policy':{'reference':'calibration only, never per-image peak','floor_db':-60,'difference':'simulated minus measured',
                             'random_clutter':'linear average before dB, not paired instantaneous pixels'},'datasets':{}}
    for name,fn in [('MAROON',maroon),('IPIX',ipix),('Maqu',maqu),('RAMP',ramp),('TSMS',tsms_audit)]:
        result['datasets'][name]=fn(a.output);dump(a.output/'summary.json',result)
        print(name,json.dumps(result['datasets'][name],ensure_ascii=False),flush=True)
    project=Path(__file__).resolve().parents[1]
    source_files=[Path(__file__).relative_to(project),*[p.relative_to(project) for p in sorted((project/'backend/radar_platform').glob('*.py'))]]
    result['source_hashes']={str(s):hashlib.sha256((project/s).read_bytes()).hexdigest() for s in source_files}
    dump(a.output/'summary.json',result)


if __name__=='__main__':main()
