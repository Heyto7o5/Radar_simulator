#!/usr/bin/env python3
"""All local RAMP ADC -> coherent RAD under explicitly assumed ULA geometry.

This checks a processing interface, NOT calibrated real-array angular accuracy.
Conditional angles at labeled ranges are diagnostics, not blind detections.
"""
import argparse,csv,json,os
from pathlib import Path
import numpy as np
from radar_platform.array_fmcw import TDMArray,process_array_adc
from radar_platform.fmcw import FMCWConfig
from radar_platform.ramp_dataset import load_real_frame,load_labels
from radar_platform.ramp_comparison import OFFICIAL_SWEEP_SLOPE_HZ_S
from radar_platform.repeat_metrics import distribution


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    root=Path('/nas/datasets/hyt/datasets/radar_open');os.environ['RADAR_DATA_ROOT']=str(root)
    base=root/'ramp_adc_77ghz/extracted';paths=sorted(base.rglob('radar_raw_frame/*.mat'))
    # Processing depends on slope, sampling and same-Tx interval. Absolute ramp
    # start is unavailable here; this nominal 48us ramp is not an observed fact.
    ramp=48e-6;band=OFFICIAL_SWEEP_SLOPE_HZ_S*ramp
    cfg=FMCWConfig(77e9-band/2,77e9+band/2,ramp,60e-6,128,255,4e6)
    array=TDMArray.ideal_ula(cfg,2,4);angles=np.linspace(-80,80,81)
    frozen=json.loads((root/'ramp_adc_77ghz/processed/comparison_20260923T073129Z/comparison.json').read_text())
    calibration=set(frozen['calibration_frames'])
    selected=set(np.array(paths,dtype=object)[[round((len(paths)-1)*q) for q in [.25,.5,.75]]].tolist())
    # Save complete cubes for fixed presentation frames; all other frames retain
    # raw source ADC plus RA projections, enabling deterministic full reconstruction.
    rows=[];label_rows=[];ras=[]
    for i,path in enumerate(paths):
        frame_id=str(path.relative_to(base));adc,_=load_real_frame(frame_id)
        result=process_array_adc(adc,cfg,array,azimuth_deg=angles,closing_doppler_sign=1,spatial_phase_sign=1)
        power=abs(result.rad)**2;ra=power.mean(axis=1);ras.append(ra)
        rows.append({'frame_id':frame_id,'role':'calibration' if frame_id in calibration else 'development',
                     'adc_shape':str(adc.shape),'rad_shape':str(result.rad.shape),'finite':bool(np.isfinite(result.rad).all()),
                     'hardware_array_geometry_verified':False,'absolute_angle_calibration_verified':False})
        for label in load_labels(path):
            r=label['range_m'];index=int(np.argmin(abs(result.range_m-r)))
            estimated=float(angles[np.argmax(ra[index])]);expected=float(np.rad2deg(np.arctan2(label['x_m'],label['y_m'])))
            label_rows.append({'frame_id':frame_id,'uid':label['uid'],'role':rows[-1]['role'],
                              'annotation_range_m':r,'annotation_azimuth_deg':expected,'conditional_peak_angle_deg':estimated,
                              'absolute_difference_deg':abs(estimated-expected),'eligible_as_calibrated_angle_truth':False})
        if path in selected:
            np.savez_compressed(a.output/(path.stem+'_complex_rad.npz'),rad=result.rad,rd_channels=result.rd_channels,
                                 range_m=result.range_m,velocity_mps=result.velocity_mps,azimuth_deg=angles)
        if i%15==0:print('real RAMP coherent processing',i,'/',len(paths),flush=True)
    for name,values in [('frame_processing.csv',rows),('conditional_label_diagnostics.csv',label_rows)]:
        with (a.output/name).open('w',encoding='utf-8-sig',newline='') as h:
            w=csv.DictWriter(h,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    np.savez_compressed(a.output/'all_ra_power.npz',power=np.stack(ras),range_m=result.range_m,azimuth_deg=angles)
    summary={'processed_real_frames':len(rows),'finite_cubes':sum(x['finite'] for x in rows),
             'observed_hardware_angular_accuracy_verified':False,'assumptions':['ideal 2Tx4Rx half-wavelength virtual ULA',
                 'channel order tx-major/rx-minor','spatial phase sign +1 not calibrated','no phase/amplitude channel calibration',
                 'nominal ramp only specifies known slope; start frequency and ADC start not independently verified'],
             'source_truth':'camera/label centers do not necessarily equal radar scattering centers',
             'conditional_angle_difference_deg':distribution([x['absolute_difference_deg'] for x in label_rows if x['role']=='development']),
             'complete_cubes_retained_for':[p.name for p in sorted(selected)],
             'all_other_cubes':'reconstructible from original ADC; all RA projections and configurations retained',
             'radar_processing_config':cfg.__dict__,'array':{'tx':array.tx_ned_m.tolist(),'rx':array.rx_ned_m.tolist(),'offsets_s':array.tx_offsets_s.tolist()},
             'metadata':result.metadata}
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ['processed_real_frames','finite_cubes','observed_hardware_angular_accuracy_verified']},indent=2))


if __name__=='__main__':main()
