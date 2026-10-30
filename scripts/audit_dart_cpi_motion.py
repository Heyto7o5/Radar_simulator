#!/usr/bin/env python3
"""Check constant pose/velocity approximation over the documented 256 ms CPI."""
import csv,json
from pathlib import Path
import h5py
import numpy as np

ROOT=Path('/nas/datasets/hyt/datasets/radar_open/dart_lab/extracted/lab-1')
OUT=Path(__file__).resolve().parents[1]/'docs/validation_diagnostics/2026-10-15/dart'

def main():
    with h5py.File(ROOT/'trajectory.h5') as f:
        t=f['t'][:];pos=f['pos'][:];vel=f['vel'][:];rot=f['rot'][:];valid=f['valid'][:]
    if np.any(np.diff(t)<=0):raise ValueError('nonmonotonic trajectory timestamps')
    rows=[]
    offsets=np.linspace(-.1275,.1275,9)
    for i in np.flatnonzero(valid):
        times=t[i]+offsets
        if times[0]<t[0] or times[-1]>t[-1]:continue
        measured=np.column_stack([np.interp(times,t,pos[:,j]) for j in range(3)])
        predicted=pos[i]+offsets[:,None]*vel[i]
        errors=np.linalg.norm(measured-predicted,axis=1)
        # Use actual samples within CPI, no unvalidated rotation interpolation.
        inside=np.flatnonzero((t>=times[0])&(t<=times[-1]))
        relative=np.einsum('ij,njk->nik',rot[i].T,rot[inside])
        angle=np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)))
        rows.append({'trajectory_index':int(i),'max_linear_position_error_m':float(errors.max()),
                     'rms_linear_position_error_m':float(np.sqrt(np.mean(errors**2))),
                     'max_observed_rotation_deg':float(angle.max()),'pose_samples':len(inside)})
    with (OUT/'cpi_motion_samples.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={'scene_count':1,'frames_checked':len(rows),'cpi_s':.256,
             'reference':'published smoothed trajectory, not independent motion capture',
             'first_valid_frame':rows[0],
             'linear_error_is_not_direct_range_error':True,
             'position_check_uses_interpolation_between_published_poses':True}
    for key in ['max_linear_position_error_m','max_observed_rotation_deg']:
        values=np.array([row[key] for row in rows]);summary[key]={'median':float(np.median(values)),'p90':float(np.quantile(values,.9)),'max':float(values.max())}
    (OUT/'cpi_motion_audit.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
