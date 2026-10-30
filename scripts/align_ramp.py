#!/usr/bin/env python3
"""Sequence-split RAMP ablations on one shared complex RD/RAD processing grid."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from radar_platform.ramp_alignment import AnnotationTracks,read_ramp_adc,aligned_config,aligned_array,process_aligned_adc,simulate_track_adc,annotation_masks
from radar_platform.ramp_comparison import SimulationTuning,simulate_label_conditioned_adc,simulate_label_conditioned_adc_legacy
from radar_platform.array_fmcw import process_array_adc
from radar_platform.alignment_metrics import fit_target_power
from radar_platform.comparison import compare_power,representative_indices
from radar_platform.repeat_metrics import distribution
from validate_common_real import plot_maps
from matplotlib import font_manager
import matplotlib.pyplot as plt

MODELS=['legacy_random','stable_uid','physical_raw_track','physical_smooth_track','physical_smooth_roi_gain']


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as h:
        writer=csv.DictWriter(h,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/nas/datasets/hyt/datasets/radar_open/ramp_adc_77ghz/alignment_subset_20261005'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    selection=json.loads((a.root/'selection.json').read_text());records=selection['frames']
    tracks={seq:AnnotationTracks(a.root/'Automotive'/seq/'text_labels',window_frames=11) for seq in {r['sequence'] for r in records}}
    cfg=aligned_config();angles=np.linspace(-80,80,81)
    tuning=SimulationTuning(name='point_only',scatterers_per_target=1,noise_std=0.)
    project=Path(__file__).resolve().parents[1]
    sources=[*sorted((project/'backend/radar_platform').glob('*.py')),Path(__file__)]
    dump(a.output/'protocol.json',{'input_selection':str(a.root/'selection.json'),'models':MODELS,'track_window_frames':11,
        'source_sha256':{str(p.relative_to(project)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        'radar_config':cfg.__dict__,'rad_shape':[128,255,81],'windows':['hamming','hamming','hamming'],
        'scope':'no CAD; class amplitude ratios remain assumptions; trajectory labels are input estimates, not precise radar truth',
        'full_scoring_range_m':[1,24],'power_roi':'annotation range/angle plus fixed 1.5m/15deg and box extent; all Doppler',
        'calibration_only_orientation':True,'no_per_channel_hardware_fit':True,'closing_sign':-1,
        'display_reference':'median of per-calibration-frame target-region peak powers; exclude near-zero-range leakage',
        'anti_false_good_score':'report fraction of jointly clipped cells and union signal-support MAE as well as full/ROI MAE'})
    # Nominal sensor-coordinate handedness is the ONLY selected angular parameter.
    # This is empirical label registration, not surveyed hardware calibration.
    orientation_rows=[];calrecords=[r for r in records if r['role']=='calibration']
    for i,r in enumerate(calrecords):
        labels=tracks[r['sequence']].labels(r['frame_index'],smooth=True)
        if len(labels)!=1:continue
        adc=read_ramp_adc(a.root/r['frame_id'])
        rd=process_array_adc(adc,cfg,aligned_array(1),azimuth_deg=[0.],closing_doppler_sign=-1,spatial_phase_sign=-1,
                              range_window='hamming',doppler_window='hamming',spatial_window='hamming')
        label=labels[0];r0=label['range_m'];theta=np.degrees(np.arctan2(label['x_m'],label['y_m']))
        valid_range=np.flatnonzero(abs(rd.range_m-r0)<=1.5+max(label['length_m'],label['width_m'])/2)
        if not len(valid_range) or abs(theta)>75:continue
        power=np.mean(abs(rd.rd_channels)**2,axis=(2,3))
        ir,iv=np.unravel_index(np.argmax(power[valid_range]),(len(valid_range),255));ir=int(valid_range[ir])
        vector=rd.rd_channels[ir,iv].T.reshape(-1)
        raw_frequency=-rd.closing_doppler_hz[iv]
        vector=vector*np.exp(-2j*np.pi*raw_frequency*np.repeat(aligned_array(1).tx_offsets_s,4))
        for sign in [-1,1]:
            array=aligned_array(sign);directions=np.c_[np.cos(np.deg2rad(angles)),np.sin(np.deg2rad(angles)),np.zeros(len(angles))]
            steering=np.exp(-2j*np.pi*directions@array.virtual_positions_ned_m.T/cfg.wavelength_m)
            response=abs(steering.conj()@(vector*np.hamming(8)))**2
            estimated=float(angles[response.argmax()])
            orientation_rows.append({'frame_id':r['frame_id'],'east_sign':sign,'label_angle_deg':float(theta),'estimated_angle_deg':estimated,
                'angle_abs_error_deg':abs(estimated-theta),'label_closing_mps':label['radial_velocity_mps'],
                'fft_closing_mps_assuming_minus_sign':float(rd.velocity_mps[iv])})
        if i%60==0:print('orientation calibration',i,'/',len(calrecords),flush=True)
    medians={s:float(np.median([x['angle_abs_error_deg'] for x in orientation_rows if x['east_sign']==s])) for s in [-1,1]}
    east_sign=min(medians,key=medians.get)
    csv_write(a.output/'orientation_calibration.csv',orientation_rows)
    registration={'east_sign':east_sign,'calibration_median_angle_errors_deg':medians,'source':'nominal TI virtual geometry plus cal-label axis registration',
                  'hardware_channel_amplitude_phase_calibrated':False,'closing_sign':-1,
                  'frequency_origin_note':'published start=77GHz, ADC-start unknown; observed sweep segment only'}
    dump(a.output/'registration.json',registration)

    def predictions(record):
        raw=tracks[record['sequence']].labels(record['frame_index'],smooth=False)
        smooth=tracks[record['sequence']].labels(record['frame_index'],smooth=True)
        seed=42+records.index(record)
        scene_seed=int.from_bytes(hashlib.sha256(record['sequence'].encode()).digest()[:4],'little')
        return raw,smooth,{
            'legacy_random':simulate_label_conditioned_adc_legacy(raw,tuning,seed),
            'stable_uid':simulate_label_conditioned_adc(raw,tuning,seed,scene_seed=scene_seed),
            'physical_raw_track':simulate_track_adc(raw,east_sign=east_sign),
            'physical_smooth_track':simulate_track_adc(smooth,east_sign=east_sign),
        }

    def powers(result):return abs(result.rad).astype(np.float64)**2,np.mean(abs(result.rd_channels).astype(np.float64)**2,axis=(2,3))
    calibrations=[];cal_max=[];cal_rd_max=[];cache=a.output/'calibration_cache';cache.mkdir()
    for i,record in enumerate(calrecords):
        real=process_aligned_adc(read_ramp_adc(a.root/record['frame_id']),east_sign=east_sign)
        mp,mrd=powers(real);raw,smooth,models=predictions(record)
        target,background,valid=annotation_masks(raw,real.range_m,real.velocity_mps,angles)
        if not target.any() or not background.any():continue
        cal_max.append(float(np.max(mp[target])));cal_rd_max.append(float(np.max(mrd[target.any(axis=2)])))
        for name,adc in models.items():
            pred=process_aligned_adc(adc,east_sign=east_sign);sp,_=powers(pred)
            try:fit=fit_target_power([mp],[sp],[target],[background])
            except ValueError:continue
            calibrations.append({'frame_id':record['frame_id'],'model':name,'total_measured_power':float(mp[valid].sum()),
                'total_simulated_power':float(sp[valid].sum()),'roi_power_gain':fit.power_gain,'background_mean_power':fit.background_power,
                'target_cells':fit.calibration_target_cells,'background_cells':fit.calibration_background_cells})
        if i%60==0:print('power calibration',i,'/',len(calrecords),flush=True)
    shared_frames=set.intersection(*[set(r['frame_id'] for r in calibrations if r['model']==name) for name in MODELS[:-1]])
    if len(shared_frames)<3:raise ValueError('too few common identifiable calibration frames')
    calibrations=[r for r in calibrations if r['frame_id'] in shared_frames]
    csv_write(a.output/'power_calibration.csv',calibrations)
    gains={}
    for name in MODELS[:-1]:
        group=[r for r in calibrations if r['model']==name]
        if not group:raise ValueError('no valid calibration for '+name)
        gains[name]=sum(r['total_measured_power'] for r in group)/sum(r['total_simulated_power'] for r in group)
    gains['physical_smooth_roi_gain']=float(np.median([r['roi_power_gain'] for r in calibrations if r['model']=='physical_smooth_track']))
    ref=float(np.median(cal_max));ref_rd=float(np.median(cal_rd_max))
    dump(a.output/'power_alignment.json',{'gains':gains,'reference_RAD_power':ref,'reference_RD_power':ref_rd,
        'only_calibration_records_used':True,'common_calibration_frame_ids':sorted(shared_frames),
        'calibration_frames_with_targets':len(cal_max),'no_background_added_to_prediction':True})
    # This file is finalized before the first development or held-out raw ADC is read.
    frozen_hash=hashlib.sha256((a.output/'power_alignment.json').read_bytes()+(a.output/'registration.json').read_bytes()).hexdigest()
    dump(a.output/'frozen_before_evaluation.json',{'parameter_sha256':frozen_hash,'source_selection_sha256':hashlib.sha256((a.root/'selection.json').read_bytes()).hexdigest()})
    rows=[];angles_rows=[];cases=[];plot_data=[];inputs=[]
    evalrecords=[r for r in records if r['role']!='calibration']
    for i,record in enumerate(evalrecords):
        real=process_aligned_adc(read_ramp_adc(a.root/record['frame_id']),east_sign=east_sign)
        mp,mrd=powers(real);raw,smooth,models=predictions(record)
        target,background,valid=annotation_masks(raw,real.range_m,real.velocity_mps,angles)
        rd_target=target.any(axis=2);rd_valid=valid.any(axis=2)
        inputs.append({'frame_id':record['frame_id'],'role':record['role'],'raw_labels':raw,'smoothed_labels':smooth})
        for name,adc in models.items():
            pred=process_aligned_adc(adc,east_sign=east_sign);sp,srd=powers(pred)
            variants=[name,'physical_smooth_roi_gain'] if name=='physical_smooth_track' else [name]
            for variant in variants:
                gain=gains[variant]
                comp=compare_power(mp,sp*gain,reference_power=ref,target_mask=target,valid_mask=valid)
                rcomp=compare_power(mrd,srd*gain,reference_power=ref_rd,target_mask=rd_target,valid_mask=rd_valid)
                row={'frame_id':record['frame_id'],'sequence':record['sequence'],'role':record['role'],'model':variant,
                     'target_count':len(raw),'classes':','.join(map(str,sorted({x['class_id'] for x in raw}))),
                     'ambiguous_identity_count':sum(x['tracking_ambiguous'] for x in raw),
                     'unknown_velocity_count':sum(x['radial_velocity_mps'] is None for x in raw),
                     'rad_mae_db':comp.metrics['mae_db'],'rad_target_mae_db':comp.metrics['target_mae_db'],
                     'rd_mae_db':rcomp.metrics['mae_db'],'rd_target_mae_db':rcomp.metrics['target_mae_db'],
                     'rd_signal_support_mae_db':rcomp.metrics['signal_support_mae_db'],
                     'rad_signal_support_mae_db':comp.metrics['signal_support_mae_db'],
                     'rd_target_support_mae_db':rcomp.metrics['target_signal_support_mae_db'],
                     'rad_target_support_mae_db':comp.metrics['target_signal_support_mae_db'],
                     'rd_joint_floor_fraction':rcomp.metrics['joint_floor_fraction'],
                     'rad_joint_floor_fraction':comp.metrics['joint_floor_fraction'],
                     'simulation_zero_fraction':comp.metrics['simulation_zero_power_fraction']}
                rows.append(row)
                if variant=='physical_smooth_roi_gain':
                    # Cache only projections for every frame; full complex cubes can
                    # be reproduced from saved inputs and fixed config/gain.
                    key=record['sequence']+'_'+str(record['frame_index'])
                    path=a.output/(key+'_projections.npz')
                    np.savez_compressed(path,real_rd=mrd,sim_rd=srd*gain,real_ra=mp.mean(axis=1),sim_ra=(sp*gain).mean(axis=1),
                        rd_target=rd_target,rd_valid=rd_valid,range_m=real.range_m,velocity_mps=real.velocity_mps,angle_deg=angles,
                        reference_RD=ref_rd,reference_RAD=ref)
                    plot_data.append({'record':record,'path':str(path),'mae_db':rcomp.metrics['mae_db'],'target_mae_db':rcomp.metrics['target_mae_db']})
            del pred,sp,srd
        # Diagnostic association: only single-label frames, range gate only;
        # angle is searched over the full scan, never restricted around truth.
        if len(raw)==1 and not raw[0]['tracking_ambiguous']:
            label=raw[0];R=label['range_m'];rr=np.flatnonzero(abs(real.range_m-R)<=1.5+max(label['width_m'],label['length_m'])/2)
            if len(rr):
                ri,di=np.unravel_index(np.argmax(mrd[rr]),(len(rr),len(real.velocity_mps)));ri=int(rr[ri])
                estimated=float(angles[np.argmax(mp[ri,di])]);truth=float(np.degrees(np.arctan2(label['x_m'],label['y_m'])))
                angles_rows.append({'frame_id':record['frame_id'],'role':record['role'],'label_angle_deg':truth,
                    'rd_conditioned_peak_angle_deg':estimated,'absolute_difference_deg':abs(estimated-truth),
                    'observed_peak_range_m':float(real.range_m[ri]),'observed_peak_closing_mps':float(real.velocity_mps[di]),
                    'scope':'annotation-range-conditioned diagnostic; scattering-center association and hardware calibration not guaranteed'})
        if i%20==0:
            print('evaluation',i,'/',len(evalrecords),record['role'],flush=True)
            csv_write(a.output/'frame_metrics.partial.csv',rows)
    csv_write(a.output/'frame_metrics.csv',rows)
    if angles_rows:csv_write(a.output/'angle_diagnostics.csv',angles_rows)
    dump(a.output/'simulation_inputs.json',inputs)
    summaries=[]
    for role in ['development','heldout_record']:
        for name in MODELS:
            group=[r for r in rows if r['role']==role and r['model']==name]
            summaries.append({'role':role,'model':name,'frames':len(group),
                'frames_with_ambiguous_identity':sum(r['ambiguous_identity_count']>0 for r in group),
                'frames_with_unknown_velocity':sum(r['unknown_velocity_count']>0 for r in group),
                **{key:distribution([r[key] for r in group]) for key in ['rad_mae_db','rad_target_mae_db','rd_mae_db','rd_target_mae_db',
                  'rd_signal_support_mae_db','rad_signal_support_mae_db','rd_target_support_mae_db','rad_target_support_mae_db',
                  'rd_joint_floor_fraction','rad_joint_floor_fraction']}})
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':9})
    selected=representative_indices([r['mae_db'] for r in plot_data])
    for index in selected:
        item=plot_data[index]
        with np.load(item['path']) as z:
            comparison=compare_power(z['real_rd'].T,z['sim_rd'].T,reference_power=float(z['reference_RD']),target_mask=z['rd_target'].T,valid_mask=z['rd_valid'].T)
            cases.append({'name':item['record']['sequence']+'/'+str(item['record']['frame_index']),
                'comparison':comparison,'extent':[z['range_m'][0],z['range_m'][-1],z['velocity_mps'][0],z['velocity_mps'][-1]],
                'ylabel':'接近速度 / (m/s)','mask':z['rd_valid'].T})
    plot_maps(a.output/'aligned_rd_examples.png',cases,'RAMP 对齐后：真实 RD / 仿真 RD / 差值\n同一处理窗、物理轴与固定校准参考；无逐帧增益、移峰或测试帧拟合')
    fig,axes=plt.subplots(1,2,figsize=(13,4),layout='constrained')
    labels=['旧随机点','固定 UID','物理混频/原轨迹','平滑标签轨迹','目标区标定']
    for ax,key,title in zip(axes,['rd_target_mae_db','rad_target_mae_db'],['目标区 RD MAE / dB','目标区 RAD MAE / dB']):
        for role in ['development','heldout_record']:
            values=[next(s[key]['mean'] for s in summaries if s['role']==role and s['model']==m) for m in MODELS]
            ax.plot(range(len(MODELS)),values,'o-',label=role)
        ax.set_xticks(range(len(MODELS)),labels,rotation=15);ax.set_title(title);ax.grid(alpha=.2);ax.legend()
    fig.savefig(a.output/'ablation.png',dpi=160);plt.close(fig)
    result={'selected_frames':len(records),'calibration_frames':len(calrecords),'evaluation_frames':len(evalrecords),
        'independent_sequences':len(selection['inventory']),'models':MODELS,'registration':registration,'power_gains':gains,
        'metrics':summaries,'selected_examples':[plot_data[i]['record']['frame_id'] for i in selected],
        'full_RAD_comparison_performed':True,'grid':[128,255,81],'no_hardware_angular_accuracy_claim':True,
        'parameter_sha256_before_evaluation':frozen_hash,'scope':'one-cycle record-heldout development; ideal point targets lack real extended scattering and background'}
    dump(a.output/'summary.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
