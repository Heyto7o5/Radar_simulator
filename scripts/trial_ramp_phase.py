#!/usr/bin/env python3
"""Frozen calibration-only phase-curvature response trial on record-heldout RAD."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from radar_platform.ramp_alignment import read_ramp_adc,process_aligned_adc,simulate_track_adc,annotation_masks
from radar_platform.alignment_metrics import apply_channel_phase_response
from radar_platform.comparison import compare_power,representative_indices
from radar_platform.repeat_metrics import distribution
from validate_common_real import plot_maps
from matplotlib import font_manager
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-run',type=Path,required=True)
    p.add_argument('--phase-calibration',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('/nas/datasets/hyt/datasets/radar_open/ramp_adc_77ghz/alignment_subset_20261005'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    calibration=json.loads(a.phase_calibration.read_text())
    if not calibration['stable_enough_for_trial']:raise ValueError('calibration phase curvature is not identifiable enough for a trial')
    phase=np.array(calibration['phase_rad']);registration=json.loads((a.base_run/'registration.json').read_text())
    power=json.loads((a.base_run/'power_alignment.json').read_text());gain=power['gains']['physical_smooth_roi_gain']
    records=json.loads((a.base_run/'simulation_inputs.json').read_text());east=registration['east_sign']
    policy={'base_run':str(a.base_run),'gain_kept_fixed':gain,'phase_calibration':calibration,
            'phase_fit_from_calibration_only':True,'no_test_phase_fit':True,'absolute_angle_zero_not_identified':True,
            'forward_response_not_image_shift':True,'new_receiver_nonlinearity':False,
            'source_calibration_sha256':hashlib.sha256(a.phase_calibration.read_bytes()).hexdigest()}
    (a.output/'frozen_protocol.json').write_text(json.dumps(policy,indent=2)+'\n')
    rows=[];plot_cases=[]
    for i,record in enumerate(records):
        real=process_aligned_adc(read_ramp_adc(a.root/record['frame_id']),east_sign=east)
        adc=simulate_track_adc(record['smoothed_labels'],east_sign=east)
        target,bg,valid=annotation_masks(record['raw_labels'],real.range_m,real.velocity_mps,real.azimuth_deg)
        real_power=abs(real.rad).astype(float)**2
        baseline_rd=None
        for mode,cube in [('before',adc),('phase_response',apply_channel_phase_response(adc,phase))]:
            pred=process_aligned_adc(cube,east_sign=east);sim_power=abs(pred.rad).astype(float)**2*gain
            comp=compare_power(real_power,sim_power,reference_power=power['reference_RAD_power'],valid_mask=valid,target_mask=target)
            rd=np.mean(abs(pred.rd_channels).astype(float)**2,axis=(2,3))*gain
            if baseline_rd is None:baseline_rd=rd
            drift=float(np.max(abs(rd-baseline_rd))/max(float(baseline_rd.max()),1e-30))
            if drift>1e-5:raise AssertionError('phase-only response changed incoherent RD power')
            rows.append({'frame_id':record['frame_id'],'role':record['role'],'mode':mode,
                         'mae_db':comp.metrics['mae_db'],'target_mae_db':comp.metrics['target_mae_db'],
                         'support_mae_db':comp.metrics['signal_support_mae_db'],
                         'target_support_mae_db':comp.metrics['target_signal_support_mae_db'],
                         'joint_floor_fraction':comp.metrics['joint_floor_fraction'],'rd_relative_power_change':drift})
            if mode=='phase_response':
                # RA is an explicit linear-power Doppler average of the full RAD.
                mean_real=real_power.mean(axis=1).T;mean_sim=sim_power.mean(axis=1).T
                vmask=valid.any(axis=1).T;tmask=target.any(axis=1).T
                ra=compare_power(mean_real,mean_sim,reference_power=power['reference_RAD_power'],valid_mask=vmask,target_mask=tmask)
                plot_cases.append({'name':Path(record['frame_id']).parts[1]+'/'+Path(record['frame_id']).stem,
                    'comparison':ra,'mask':vmask,'extent':[real.range_m[0],real.range_m[-1],real.azimuth_deg[0],real.azimuth_deg[-1]],
                    'ylabel':'方位扫描角 / °'})
        if i%50==0:print('fixed phase response evaluation',i,'/',len(records),flush=True)
    with (a.output/'frame_metrics.csv').open('w',encoding='utf-8-sig',newline='') as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summaries=[]
    for role in ['development','heldout_record']:
        for mode in ['before','phase_response']:
            group=[r for r in rows if r['role']==role and r['mode']==mode]
            summaries.append({'role':role,'mode':mode,'frames':len(group),
                **{key:distribution([r[key] for r in group]) for key in ['mae_db','target_mae_db','support_mae_db','target_support_mae_db','rd_relative_power_change']}})
    font='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc';font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family':font_manager.FontProperties(fname=font).get_name(),'axes.unicode_minus':False,'font.size':9})
    indices=representative_indices([c['comparison'].metrics['mae_db'] for c in plot_cases])
    plot_maps(a.output/'aligned_ra_examples.png',[plot_cases[i] for i in indices],
              'RAMP：真实 / 仿真 / 差值（RA 为 RAD 的 Doppler 功率平均）\n仅加入校准记录估计的一组固定通道相位；绝对角度零点仍未知')
    (a.output/'summary.json').write_text(json.dumps({'rows':summaries,'fixed_phase_deg':np.rad2deg(phase).tolist(),
        'selected_examples':[plot_cases[i]['name'] for i in indices],
        'scope':'non-affine phase trial, not certified hardware calibration; original raw measurements and global power gain unchanged'},indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
