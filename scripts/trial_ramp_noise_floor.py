#!/usr/bin/env python3
"""Calibration-record-only diffuse background floor experiment; ideal mode unchanged."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from radar_platform.ramp_alignment import AnnotationTracks,read_ramp_adc,aligned_config,aligned_array,process_aligned_adc,simulate_track_adc
from radar_platform.array_fmcw import process_array_adc
from radar_platform.alignment_metrics import estimate_adc_noise_variance,add_calibrated_noise,apply_channel_phase_response
from radar_platform.comparison import compare_power,representative_indices
from radar_platform.repeat_metrics import distribution
from validate_common_real import plot_maps
from matplotlib import font_manager
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-run',type=Path,required=True)
    p.add_argument('--phase-calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('/nas/datasets/hyt/datasets/radar_open/ramp_adc_77ghz/alignment_subset_20261005'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    selected=json.loads((a.root/'selection.json').read_text())['frames'];inputs=json.loads((a.base_run/'simulation_inputs.json').read_text())
    power_cal=json.loads((a.base_run/'power_alignment.json').read_text());registration=json.loads((a.base_run/'registration.json').read_text());east=registration['east_sign']
    phase_cal=json.loads(a.phase_calibration.read_text());phase=np.array(phase_cal['phase_rad'])
    if not phase_cal['stable_enough_for_trial']:raise ValueError('phase candidate failed prespecified consistency gate')
    cfg=aligned_config();array=aligned_array(east);tracks={};calibration_rows=[]
    for i,record in enumerate(r for r in selected if r['role']=='calibration'):
        seq=record['sequence']
        if seq not in tracks:tracks[seq]=AnnotationTracks(a.root/'Automotive'/seq/'text_labels')
        labels=tracks[seq].labels(record['frame_index'])
        result=process_array_adc(read_ramp_adc(a.root/record['frame_id']),cfg,array,azimuth_deg=[0.],
            closing_doppler_sign=-1,spatial_phase_sign=-1,range_window='hamming',doppler_window='hamming',spatial_window='hamming')
        rmask=(result.range_m>=1)&(result.range_m<=24)
        for label in labels:
            width=2.25+max(label['width_m'],label['length_m'])/2
            rmask&=abs(result.range_m-label['range_m'])>width
        mask=rmask[:,None]&(abs(result.velocity_mps)>.25)[None,:]
        if mask.sum()<1000:continue
        variance=estimate_adc_noise_variance(result.rd_channels,mask,np.hamming(128),np.hamming(255))
        calibration_rows.append({'frame_id':record['frame_id'],'off_target_cells':int(mask.sum()),'variance':variance.tolist()})
    if len(calibration_rows)<50:raise ValueError('not enough calibration-only background samples')
    variance=np.median(np.array([r['variance'] for r in calibration_rows]),axis=0)
    gain=power_cal['gains']['physical_smooth_roi_gain']
    protocol={'noise_variance_ADC_units_tx_major':variance.tolist(),'noise_std_ADC_units_tx_major':np.sqrt(variance).tolist(),
              'calibration_frames':calibration_rows,'phase_rad':phase.tolist(),'signal_power_gain_kept_fixed':gain,
              'definition':'independent circular complex Gaussian floor, variance inferred from off-target and away-from-zero-Doppler calibration cells',
              'ideal_default_unchanged':True,'no_static_clutter_or_multipath_added':True,'not_physical_noise_figure':True,
              'no_evaluation_frame_fit':True,'fixed_main_seed_rule':'42 + index in fixed evaluation list',
              'seed_sensitivity_rule':'extra seeds 142+i and 242+i at fixed 25/50/75 percent list positions',
              'scope':'experimental observation-noise alignment; not a full receiver or environment model'}
    (a.output/'frozen_calibration.json').write_text(json.dumps(protocol,indent=2)+'\n')
    rows=[];cases=[];sensitivity=[]
    anchors={round((len(inputs)-1)*q) for q in [.25,.5,.75]}
    for i,item in enumerate(inputs):
        real=process_aligned_adc(read_ramp_adc(a.root/item['frame_id']),east_sign=east)
        signal=apply_channel_phase_response(simulate_track_adc(item['smoothed_labels'],east_sign=east),phase)*np.sqrt(gain)
        pred=process_aligned_adc(add_calibrated_noise(signal,variance,42+i),east_sign=east)
        labels=item['raw_labels'];rvalid=(real.range_m>=1)&(real.range_m<=24)
        target=np.zeros((128,81),dtype=bool)
        for label in labels:
            r=label['range_m'];theta=np.rad2deg(np.arctan2(label['x_m'],label['y_m']))
            extent=.5*(abs(label['x_m']/r)*label['width_m']+abs(label['y_m']/r)*label['length_m'])
            da=15+np.rad2deg(np.arctan2(max(label['width_m'],label['length_m'])/2,r))
            target|=(abs(real.range_m-r)<=1.5+extent)[:,None]&(abs(real.azimuth_deg-theta)<=da)[None,:]
        target=np.broadcast_to((target&rvalid[:,None])[:,None,:],real.rad.shape);valid=np.broadcast_to(rvalid[:,None,None],real.rad.shape)
        rp=abs(real.rad).astype(float)**2;sp=abs(pred.rad).astype(float)**2
        rdreal=np.mean(abs(real.rd_channels).astype(float)**2,axis=(2,3));rdsim=np.mean(abs(pred.rd_channels).astype(float)**2,axis=(2,3))
        rad=compare_power(rp,sp,reference_power=power_cal['reference_RAD_power'],target_mask=target,valid_mask=valid)
        rd=compare_power(rdreal,rdsim,reference_power=power_cal['reference_RD_power'],target_mask=target.any(axis=2),valid_mask=valid.any(axis=2))
        rows.append({'frame_id':item['frame_id'],'role':item['role'],'rd_mae_db':rd.metrics['mae_db'],'rd_target_mae_db':rd.metrics['target_mae_db'],
                     'rad_mae_db':rad.metrics['mae_db'],'rad_target_mae_db':rad.metrics['target_mae_db'],
                     'rd_joint_floor_fraction':rd.metrics['joint_floor_fraction'],'rad_joint_floor_fraction':rad.metrics['joint_floor_fraction'],
                     'seed':42+i,'ambiguous_identity_count':sum(x['tracking_ambiguous'] for x in labels)})
        display=compare_power(rdreal.T,rdsim.T,reference_power=power_cal['reference_RD_power'],target_mask=target.any(axis=2).T,valid_mask=valid.any(axis=2).T)
        cases.append({'name':Path(item['frame_id']).parts[1]+'/'+Path(item['frame_id']).stem,'comparison':display,
                      'extent':[real.range_m[0],real.range_m[-1],real.velocity_mps[0],real.velocity_mps[-1]],'mask':valid.any(axis=2).T,'ylabel':'接近速度 / (m/s)'})
        if i in anchors:
            for seed in [142+i,242+i]:
                other=process_aligned_adc(add_calibrated_noise(signal,variance,seed),east_sign=east)
                error=compare_power(rp,abs(other.rad).astype(float)**2,reference_power=power_cal['reference_RAD_power'],target_mask=target,valid_mask=valid)
                sensitivity.append({'frame_id':item['frame_id'],'seed':seed,'rad_mae_db':error.metrics['mae_db'],'rad_target_mae_db':error.metrics['target_mae_db']})
        if i%50==0:print('calibrated noise floor evaluation',i,'/',len(inputs),flush=True)
    for filename,data in [('frame_metrics.csv',rows),('seed_sensitivity.csv',sensitivity)]:
        with (a.output/filename).open('w',encoding='utf-8-sig',newline='') as h:
            w=csv.DictWriter(h,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    summaries=[]
    for role in ['development','heldout_record']:
        group=[r for r in rows if r['role']==role]
        summaries.append({'role':role,'frames':len(group),**{k:distribution([r[k] for r in group]) for k in ['rd_mae_db','rd_target_mae_db','rad_mae_db','rad_target_mae_db']}})
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':9})
    choose=representative_indices([c['comparison'].metrics['mae_db'] for c in cases])
    plot_maps(a.output/'rd_examples.png',[cases[j] for j in choose],
        'RAMP 对照：理想目标 + 校准期估计的固定通道相位与统计噪声底\n随机噪声不是实测纹理复制；静态杂波、材料和目标扩展仍未复现')
    (a.output/'summary.json').write_text(json.dumps({'results':summaries,'noise_variance':variance.tolist(),'calibration_frames':len(calibration_rows),
         'selected_examples':[cases[j]['name'] for j in choose],'scope':protocol['scope'],'ideal_default_unchanged':True},indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
