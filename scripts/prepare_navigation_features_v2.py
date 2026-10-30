#!/usr/bin/env python3
"""Cache alternative RDA-only preprocessing separately from frozen raw data."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
import time
import numpy as np

from radar_platform.navigation.io import local_path,read_json,write_json,digest
from radar_platform.navigation.observation import Observation
from radar_platform.navigation.features_v2 import encode_v2,coordinate_grid


def prepare(job):
    source,destination,case_id=job;source=local_path(source);destination=local_path(destination)
    destination.mkdir(parents=True,exist_ok=True)
    key=hashlib.sha256((source/'input.npy').read_bytes()).hexdigest()
    if (destination/'complete.json').exists():
        cached=read_json(destination/'complete.json')
        if cached['source_feature_sha256']!=key:raise ValueError('source feature cache changed')
        return cached
    observation=Observation.load(source/'rda.npz');features,baseline=encode_v2(observation)
    np.save(destination/'input.tmp.npy',features.astype(np.float16));(destination/'input.tmp.npy').replace(destination/'input.npy')
    np.save(destination/'velocity_baseline.npy',baseline)
    for filename in ['truth.json','rda.npz']:
        link=destination/filename
        if not link.exists():link.symlink_to(source/filename)
    metadata=dict(id=case_id,source_feature_sha256=key,shape=list(features.shape),
                  velocity_baseline_mps=baseline.tolist(),physical_data_unchanged=True)
    write_json(destination/'complete.json',metadata);return metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=8)
    args=p.parse_args();source=local_path(args.data);root=local_path(args.output);root.mkdir(parents=True,exist_ok=True)
    if read_json(source/'status.json')['status']!='complete':raise ValueError('physical data must be complete first')
    protocol=read_json(source/'protocol.json')
    source_files=[local_path(__file__),local_path('backend/radar_platform/navigation/features_v2.py'),local_path('backend/radar_platform/navigation/observation.py')]
    hashes={str(p.relative_to(local_path('.'))):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    contract=dict(parent=str(source),parent_protocol_sha256=digest(protocol),source_sha256=hashes,
        preprocessing='Original 16 spectral channels plus log integrated power, log peak power, native Doppler mean/std; native-power velocity least-squares baseline. Numeric radar data only.')
    if (root/'preprocessing.json').exists() and read_json(root/'preprocessing.json')!=contract:raise ValueError('feature preprocessing changed; use new output')
    write_json(root/'preprocessing.json',contract);write_json(root/'protocol.json',protocol)
    for filename in ['base_config.json','scene_provenance.json']:write_json(root/filename,read_json(source/filename))
    records=protocol['records'];started=time.time()
    write_json(root/'status.json',dict(status='running',pid=os.getpid(),total=len(records)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(prepare,(str(source/'cases'/r['id']),str(root/'cases'/r['id']),r['id'])) for r in records]
        for done,future in enumerate(as_completed(futures),1):
            result=future.result();write_json(root/'status.json',dict(status='running',pid=os.getpid(),done=done,total=len(records)))
            if done%32==0:print(json.dumps(dict(done=done,total=len(records),last=result['id'])),flush=True)
    observation=Observation.load(source/'cases'/records[0]['id']/'rda.npz')
    np.save(root/'coordinates.npy',coordinate_grid(observation.range_m,observation.elevation_deg,observation.azimuth_deg))
    np.savez(root/'acquisition_axes.npz',**{k:getattr(observation,k) for k in ['range_m','velocity_mps','elevation_deg','azimuth_deg']})
    write_json(root/'status.json',dict(status='complete',cases=len(records),elapsed_s=time.time()-started))


if __name__=='__main__':main()
