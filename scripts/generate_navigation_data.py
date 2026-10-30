#!/usr/bin/env python3
"""Resumable physical two-frame radar generation, no image transformations."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

from radar_platform.navigation.io import local_path,write_json,read_json
from radar_platform.navigation.scene import navigation_scene
from radar_platform.navigation.dataset import pilot_records,learning_records,generate_case


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--pilot',action='store_true')
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--limit',type=int)
    p.add_argument('--terrain-only',action='store_true')
    p.add_argument('--train-count',type=int,default=4096)
    args=p.parse_args();root=local_path(args.output);root.mkdir(parents=True,exist_ok=True)
    cfg,provenance=navigation_scene(terrain_only=args.terrain_only)
    records=pilot_records() if args.pilot else learning_records({'train':args.train_count,'val':512,'test':1024})
    if args.limit:records=records[:args.limit]
    protocol=dict(records=records,pilot=args.pilot,terrain_only=args.terrain_only,
        input='Two physical R-D-elevation-azimuth magnitude volumes and fixed acquisition axes/timestamps only',
        label='Ground target relative position and platform velocity in current aircraft forward/right/up axes at second CPI end',
        split='Whole short trajectories, disjoint 3D initial-state cells and independent scene/noise seeds; fixed geography',
        preprocessing='log1p(power / frame_mean_power), numeric resampling, float16 feature cache; raw magnitude float32 retained',
        counts={s:sum(r['split']==s for r in records) for s in sorted({r['split'] for r in records})})
    if (root/'protocol.json').exists() and read_json(root/'protocol.json')!=protocol:
        raise ValueError('existing protocol differs; use a new output')
    write_json(root/'protocol.json',protocol);write_json(root/'base_config.json',cfg.model_dump(mode='json'))
    write_json(root/'scene_provenance.json',provenance)
    files=sorted(local_path('backend/radar_platform').rglob('*.py'))+[local_path(__file__)]
    hashes={str(f.relative_to(local_path('.'))):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    if (root/'source_sha256.json').exists():
        previous=read_json(root/'source_sha256.json')
        # These modules are never imported by generation. Changes to physical
        # dependencies remain forbidden; the original source archive is retained.
        unrelated={f'backend/radar_platform/navigation/{name}.py' for name in
                   ['inference','model','learning','controller','episode']}
        unrelated.add('scripts/generate_navigation_data.py')  # this resume/provenance wrapper
        changes={}
        for name,value in previous.items():
            if hashes.get(name)!=value:
                if name not in unrelated:raise ValueError(f'generation dependency changed: {name}; use a new output')
                changes[name]={'original':value,'current':hashes.get(name)}
        if changes:write_json(root/f'resume_sources_{time.time_ns()}.json',dict(compatible_nonphysical_changes=changes,
            reason='Original physics/scene/trajectory/preprocessing/dataset code unchanged; only unused learning modules or resumption wrapper changed. Cached configuration hashes checked individually.'))
    else:
        write_json(root/'source_sha256.json',hashes)
        with zipfile.ZipFile(root/'generation_source.zip','x',zipfile.ZIP_DEFLATED) as z:
            for f in files:z.write(f,str(f.relative_to(local_path('.'))))
    started=time.time()
    write_json(root/'status.json',dict(status='running',pid=os.getpid(),total=len(records),started_unix_s=started))
    jobs=[(cfg.model_dump(mode='json'),r,str(root/'cases'/r['id']),k==0) for k,r in enumerate(records)]
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(generate_case,j) for j in jobs]
            for done,future in enumerate(as_completed(futures),1):
                result=future.result()
                print(json.dumps(dict(done=done,total=len(jobs),**result)),flush=True)
                write_json(root/'status.json',dict(status='running',pid=os.getpid(),done=done,total=len(jobs),
                    started_unix_s=started,elapsed_s=time.time()-started,last_case=result))
        write_json(root/'status.json',dict(status='complete',cases=len(records),frames=2*len(records),elapsed_s=time.time()-started))
    except BaseException as exc:
        write_json(root/'status.json',dict(status='failed',error=str(exc),pid=os.getpid()))
        raise


if __name__=='__main__':main()
