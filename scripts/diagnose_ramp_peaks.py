#!/usr/bin/env python3
"""Conditional peak correspondence and array-response identifiability audit.

No evaluation waveform is used to tune geometry, gain, timing or phase. Peak
differences are diagnostics inside annotation range regions, NOT blind detection
accuracy or true physical target-speed labels. Overlapping range regions are
excluded with an explicit reason, rather than matching one peak to several objects.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from radar_platform.ramp_alignment import AnnotationTracks,read_ramp_adc,aligned_config,aligned_array,simulate_track_adc
from radar_platform.ramp_comparison import SimulationTuning,simulate_label_conditioned_adc_legacy,simulate_label_conditioned_adc
from radar_platform.array_fmcw import process_array_adc
from radar_platform.alignment_metrics import fit_relative_channel_response
from radar_platform.repeat_metrics import distribution


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/nas/datasets/hyt/datasets/radar_open/ramp_adc_77ghz/alignment_subset_20261005'))
    p.add_argument('--registration',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    selection=json.loads((a.root/'selection.json').read_text())['frames'];registration=json.loads(a.registration.read_text());east=registration['east_sign']
    cfg=aligned_config();array=aligned_array(east);angles=np.linspace(-80,80,81)
    tracks={seq:AnnotationTracks(a.root/'Automotive'/seq/'text_labels') for seq in {r['sequence'] for r in selection}}
    direction=np.c_[np.cos(np.deg2rad(angles)),np.sin(np.deg2rad(angles)),np.zeros(len(angles))]
    steering=np.exp(-2j*np.pi*direction@array.virtual_positions_ned_m.T/cfg.wavelength_m)
    cal_vectors=[];expected=[];cal_keys=[];rows=[];excluded=[]
    tuning=SimulationTuning('point',noise_std=0.)
    def rd_only(adc):return process_array_adc(adc,cfg,array,azimuth_deg=[0.],closing_doppler_sign=-1,spatial_phase_sign=-1,
                                             range_window='hamming',doppler_window='hamming',spatial_window='hamming')
    def peak(result,label):
        delta=1.5+.5*(abs(label['x_m']/label['range_m'])*label['width_m']+abs(label['y_m']/label['range_m'])*label['length_m'])
        rindex=np.flatnonzero((abs(result.range_m-label['range_m'])<=delta)&(result.range_m>=1)&(result.range_m<=24))
        if not len(rindex):return None
        power=np.mean(abs(result.rd_channels)**2,axis=(2,3))
        r,d=np.unravel_index(np.argmax(power[rindex]),(len(rindex),cfg.chirps));r=int(rindex[r])
        if power[r,d]<=0:return None
        vector=result.rd_channels[r,d].T.reshape(8)
        raw_frequency=-result.closing_doppler_hz[d]
        vector=vector*np.exp(-2j*np.pi*raw_frequency*np.repeat(array.tx_offsets_s,4))
        angular=abs(steering.conj()@(vector*np.hamming(8)))**2
        return {'range_m':float(result.range_m[r]),'closing_mps':float(result.velocity_mps[d]),'angle_deg':float(angles[np.argmax(angular)]),
                'vector':vector,'peak_power':float(power[r,d]),'background_median':float(np.median(power[rindex]))}
    # Calibrating empirical channel response uses only designated cal recordings.
    for i,record in enumerate(r for r in selection if r['role']=='calibration'):
        labels=tracks[record['sequence']].labels(record['frame_index'],smooth=True)
        if len(labels)!=1 or labels[0]['tracking_ambiguous']:continue
        result=rd_only(read_ramp_adc(a.root/record['frame_id']));found=peak(result,labels[0])
        if found is None or found['peak_power']<10*found['background_median']:continue
        theta=np.arctan2(labels[0]['x_m'],labels[0]['y_m']);unit=np.array([np.cos(theta),np.sin(theta),0.])
        ref=np.exp(-2j*np.pi*(array.virtual_positions_ned_m@unit)/cfg.wavelength_m)
        spatial_coherence=abs(np.vdot(ref,found['vector']))**2/(np.sum(abs(ref)**2)*np.sum(abs(found['vector'])**2))
        if spatial_coherence<.25:continue
        cal_vectors.append(found['vector']);expected.append(ref);cal_keys.append(record['frame_id'])
    response=fit_relative_channel_response(cal_vectors,expected)
    response['calibration_frame_ids']=cal_keys;response['calibration_sequence_count']=len({Path(k).parts[1] for k in cal_keys})
    response['eligible_for_trial']=response['stable_enough_for_empirical_trial'] and response['calibration_sequence_count']>=3 and len(cal_keys)>=50
    (a.output/'channel_response_diagnostic.json').write_text(json.dumps(response,indent=2)+'\n')
    print('channel response consistency',response['phase_consistency'],'eligible',response['eligible_for_trial'],flush=True)
    for i,record in enumerate(r for r in selection if r['role']!='calibration'):
        raw=tracks[record['sequence']].labels(record['frame_index']);smooth=tracks[record['sequence']].labels(record['frame_index'],smooth=True)
        real=rd_only(read_ramp_adc(a.root/record['frame_id']))
        seed=42+selection.index(record);scene_seed=int.from_bytes(hashlib.sha256(record['sequence'].encode()).digest()[:4],'little')
        generated={'legacy_random':simulate_label_conditioned_adc_legacy(raw,tuning,seed),
                   'stable_uid':simulate_label_conditioned_adc(raw,tuning,seed,scene_seed=scene_seed),
                   'physical_raw_track':simulate_track_adc(raw,east_sign=east),
                   'physical_smooth_track':simulate_track_adc(smooth,east_sign=east)}
        predictions={name:rd_only(x) for name,x in generated.items()}
        for label in raw:
            if label['tracking_ambiguous']:
                excluded.append({'frame_id':record['frame_id'],'uid':label['uid'],'reason':'ambiguous annotation identity'});continue
            own=1.5+max(label['width_m'],label['length_m'])/2
            overlap=any(abs(other['range_m']-label['range_m'])<=own+1.5+max(other['width_m'],other['length_m'])/2 for other in raw if other['uid']!=label['uid'])
            if overlap:
                excluded.append({'frame_id':record['frame_id'],'uid':label['uid'],'reason':'range gates overlap other labeled targets; no forced association'});continue
            measured=peak(real,label)
            if measured is None:
                excluded.append({'frame_id':record['frame_id'],'uid':label['uid'],'reason':'outside scored range or no observed power'});continue
            for name,pred in predictions.items():
                found=peak(pred,label)
                rows.append({'frame_id':record['frame_id'],'sequence':record['sequence'],'role':record['role'],'uid':label['uid'],'class_id':label['class_id'],'model':name,
                    'prediction_present':found is not None,'range_peak_difference_m':None if found is None else abs(found['range_m']-measured['range_m']),
                    'doppler_peak_difference_mps':None if found is None else abs(found['closing_mps']-measured['closing_mps']),
                    'angle_peak_difference_deg':None if found is None else abs(found['angle_deg']-measured['angle_deg']),
                    'observed_peak_range_m':measured['range_m'],'observed_peak_closing_mps':measured['closing_mps'],'observed_peak_angle_deg':measured['angle_deg'],
                    'annotation_velocity_available':label['radial_velocity_mps'] is not None,
                    'not_blind_detection_or_surveyed_truth':True})
        if i%50==0:print('conditional peak audit',i,flush=True)
    for name,values in [('peak_differences.csv',rows),('association_exclusions.csv',excluded)]:
        with (a.output/name).open('w',encoding='utf-8-sig',newline='') as h:
            w=csv.DictWriter(h,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    summaries=[]
    for role in ['development','heldout_record']:
        for name in generated:
            group=[r for r in rows if r['role']==role and r['model']==name]
            summaries.append({'role':role,'model':name,'target_frame_pairs':len(group),'predicted_pairs':sum(r['prediction_present'] for r in group),
                **{key:distribution([r[key] for r in group]) for key in ['range_peak_difference_m','doppler_peak_difference_mps','angle_peak_difference_deg']}})
    (a.output/'summary.json').write_text(json.dumps({'results':summaries,'excluded_target_frame_pairs':len(excluded),
        'scope':'supplementary conditional peak correspondence; not a new preregistered blind benchmark; no model refit using these peaks',
        'calibration_response_eligible':response['eligible_for_trial']},indent=2)+'\n')


if __name__=='__main__':main()
