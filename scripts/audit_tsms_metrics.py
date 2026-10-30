#!/usr/bin/env python3
"""Recompute the old TSMS mean-of-errors versus error-of-means distinction."""
import argparse,csv,json
from pathlib import Path
import numpy as np
from radar_platform.tsms_validation import load_processed_rd,processed_axes,rd_metrics,load_raw_sample


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/nas/datasets/hyt/datasets/radar_open/tsms_drone_sample'))
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise SystemExit('choose a new output directory')
    prior=args.root/'processed/platform_fmcw_20260924T034318Z/comparison.json'
    prior_data=json.loads(prior.read_text());rows=[];all_errors=[]
    for row in prior_data['rows']:
        d=row['distance_m'];errors=[]
        files=sorted((args.root/f'extracted/corner_reflector_{d:g}m').rglob('*.mat'))[30:]
        for path in files:
            rd=load_processed_rd(path);r,v=processed_axes(d,rd.shape)
            metrics=rd_metrics(rd,r,v,.6,.19,expected_range_m=d,expected_velocity_mps=0)
            errors.append(abs(metrics['peak_range_m']-row['platform_peak_range_m']))
        all_errors.extend(errors)
        rows.append({'distance_m':d,'frames':len(errors),'error_of_mean_m':row['platform_to_measured_peak_error_m'],
                     'per_frame_mae_m':float(np.mean(errors)),
                     'physical_vs_labeled_velocity_axis_max_error_mps':row['velocity_axis_max_error_mps']})
    sample=load_raw_sample(args.root/'raw/FMCW_RawData.mat')
    result={'prior_run':str(prior),'old_error_of_means_m':float(np.mean([r['error_of_mean_m'] for r in rows])),
        'recomputed_per_frame_mae_m':float(np.mean(all_errors)),'rows':rows,'raw_sample_config':sample.config.__dict__,
        'paper_configuration':{'samples':256,'chirps':128,'ramp_s':256e-6,'source':'https://www.nature.com/articles/s41597-026-06802-6'},
        'conclusion':'Historical calibrated point-target check; sample vs batch configuration is unresolved. These viewed frames are validation, not a new final test set.'}
    args.output.mkdir(parents=True)
    (args.output/'tsms_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (args.output/'tsms_audit.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
