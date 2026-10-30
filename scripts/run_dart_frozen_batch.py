#!/usr/bin/env python3
"""Fixed development batch; temporal holdout is NOT independent-scene validation."""
import concurrent.futures
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import h5py
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
DATA=Path('/nas/datasets/hyt/datasets/radar_open/dart_lab')
TAG='batch_v1'
OUT=PROJECT/'docs/validation_diagnostics/2026-10-15/dart'/TAG
CALIBRATION=[0,8,16]
EVALUATION=[32,40,48,56,64,72,80,88]

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    global TAG,OUT
    parser=argparse.ArgumentParser();parser.add_argument('--expanded-geometry',action='store_true');args=parser.parse_args()
    if args.expanded_geometry:
        TAG='batch_v2_world_geometry';OUT=OUT.parent/TAG
    OUT.mkdir(parents=True,exist_ok=True)
    snapshots={str(p.relative_to(PROJECT)):sha(p) for p in sorted((PROJECT/'backend/radar_platform').rglob('*.py'))}
    snapshots['scripts/simulate_dart_surface_baseline.py']=sha(PROJECT/'scripts/simulate_dart_surface_baseline.py')
    inputs=['processed/lab-1_supported_planes_cell_0.25.npz','extracted/lab-1/trajectory.h5','extracted/lab-1/sensor.json']
    if args.expanded_geometry:inputs[0]='processed/lab-1_supported_planes_cell_0.25_maxplanes_32_trajectory_bounds.npz'
    with h5py.File(DATA/'extracted/lab-1/trajectory.h5') as f:
        valid=np.flatnonzero(f['valid'][:]);raw=np.flatnonzero(f['mask'][:]);timestamps=f['t'][:]
        records=[{'valid_frame_index':i,'trajectory_index':int(valid[i]),'raw_frame_index':int(raw[valid[i]]),
                  'timestamp_s':float(timestamps[valid[i]]),'role':'calibration' if i in CALIBRATION else 'temporal_check'} for i in CALIBRATION+EVALUATION]
    manifest={'batch_tag':TAG,'scene_count':1,'independent_heldout_scene_count':0,'records':records,
              'same_capture_temporal_check_only':True,'frames_previously_inspected':[0,1,2],
              'selection':'fixed valid-index stride of 8; 16 indices gap after calibration; no RD-value selection',
              'fixed_geometry':'partial LiDAR geometry fitted at first valid pose; NOT refreshed per radar frame',
              'model':'sampled platform motion, 65deg cone with 5deg cosine edge, diffuse area cells .25m, seed42, no noise',
              'display_selection':'first three temporal-check frames plus worst MAE after all rows finish',
              'planned_metrics':'shared calibration-only amplitude reference; full-map and fixed geometry-region dB MAE, floor -60 dB; no per-frame fit',
              'known_processing_mismatch':'real data has upstream zero-Doppler magnitude subtraction; exclude center three bins from primary score and separately report full map',
              'hashes':snapshots,'input_sha256':{p:sha(DATA/p) for p in inputs}}
    if args.expanded_geometry:
        manifest.update(fixed_geometry='one fixed world mesh from trajectory-expanded bounding box; 22 fitted planes, 19090 triangles; not regenerated per frame',
                        frames_previously_inspected=CALIBRATION+EVALUATION,
                        development_followup_not_new_holdout=True,
                        comparison_caveat='world crop and fitted geometry changed; diffuse phases are deterministic per mesh partition, not one-to-one paired across changed meshes',
                        planned_metrics='same v1 physical grid, reference and masks; show both old fixed gain and calibration-only refitted gain; unchanged measured noise PSD and seeds')
    path=OUT/'manifest.json'
    if path.exists() and json.loads(path.read_text())!=manifest:raise ValueError('frozen manifest differs; use a new batch tag')
    path.write_text(json.dumps(manifest,indent=2))
    logroot=DATA/'processed'/TAG/'logs';logroot.mkdir(parents=True,exist_ok=True)
    state={'pid':os.getpid(),'started_unix':time.time(),'status':'running','jobs':{}}
    def save():
        temp=OUT/'state.json.part';temp.write_text(json.dumps(state,indent=2));temp.replace(OUT/'state.json')
    def run(record):
        i=record['valid_frame_index']
        for name,expected in snapshots.items():
            if sha(PROJECT/name)!=expected:raise ValueError('code changed during frozen batch: '+name)
        prefix=f'surface_sampled_motion_smooth_beam_{TAG}_f{i:04d}'
        result=OUT/f'{prefix}_status.json'
        if result.exists():raise FileExistsError('result already exists; refusing unverified overwrite')
        with (logroot/f'f{i:04d}.log').open('w') as log:
            completed=subprocess.run([sys.executable,str(PROJECT/'scripts/simulate_dart_surface_baseline.py'),
                '--sampled-trajectory','--smooth-beam','--valid-frame-index',str(i),'--batch-tag',TAG,
                '--geometry','trajectory_bounds' if args.expanded_geometry else 'initial'],
                cwd=PROJECT,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'})
        if completed.returncode!=0:raise RuntimeError(f'frame {i} failed; see {logroot}/f{i:04d}.log')
        status=json.loads(result.read_text())
        if status['raw_frame_index']!=record['raw_frame_index']:raise ValueError('frame correspondence changed')
        return {'status':'complete','result':str(result),'elapsed_s':status['elapsed_s']}
    save()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        jobs={executor.submit(run,r):r['valid_frame_index'] for r in records}
        for future in concurrent.futures.as_completed(jobs):
            i=jobs[future]
            try:state['jobs'][str(i)]=future.result()
            except Exception as exc:state['jobs'][str(i)]={'status':'failed','error':str(exc)}
            save();print(json.dumps({'frame':i,**state['jobs'][str(i)]}),flush=True)
    state['status']='complete' if all(v['status']=='complete' for v in state['jobs'].values()) else 'failed'
    save();print('batch',state['status'],flush=True)

if __name__=='__main__':main()
