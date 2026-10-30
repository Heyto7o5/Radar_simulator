#!/usr/bin/env python3
"""Supplement training cells with lateral motion found in bounded turn maneuvers."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
import time
import numpy as np

from radar_platform.navigation.io import local_path,read_json,write_json,digest
from radar_platform.navigation.dynamics import body_to_neu
from radar_platform.navigation.dataset import generate_case


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=16);p.add_argument('--count',type=int,default=512)
    args=p.parse_args();parent=local_path(args.parent);root=local_path(args.output);root.mkdir(parents=True,exist_ok=True)
    base=read_json(parent/'base_config.json');parent_protocol=read_json(parent/'protocol.json')
    available=[r for r in parent_protocol['records'] if r['split']=='train']
    rng=np.random.default_rng(2026103019);records=[]
    for k in range(args.count):
        original=available[int(rng.integers(len(available)))];r=dict(original)
        yaw=float(rng.uniform(-np.pi,np.pi));direction=float(rng.uniform(-np.pi,np.pi));speed=float(rng.uniform(0,12))
        vb=np.array([speed*np.cos(direction),speed*np.sin(direction),rng.uniform(-3,3)])
        r.update(id=f'turn_{k:05d}',yaw_rad=yaw,velocity_neu_mps=(body_to_neu(yaw)@vb).tolist(),
            acceleration_body_mps2=rng.uniform([-2.8,-2.8,-1],[2.8,2.8,1]).tolist(),
            yaw_rate_rad_s=float(rng.uniform(-np.pi/3,np.pi/3)),phase_seed=2100000000+k,noise_seed=2200000000+k)
        records.append(r)
    protocol=dict(records=records,counts={'train':len(records)},parent_protocol_sha256=digest(parent_protocol),
        reason='Before model fitting or test evaluation: bounded 12m/s turn has body-lateral velocity 6.51m/s; augment all horizontal velocity directions within existing training cells')
    if (root/'protocol.json').exists() and read_json(root/'protocol.json')!=protocol:raise ValueError('turn protocol changed')
    write_json(root/'protocol.json',protocol);write_json(root/'base_config.json',base)
    write_json(root/'scene_provenance.json',read_json(parent/'scene_provenance.json'))
    files=sorted(local_path('backend/radar_platform').rglob('*.py'))+[local_path(__file__)]
    hashes={str(f.relative_to(local_path('.'))):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    if (root/'source_sha256.json').exists():
        previous=read_json(root/'source_sha256.json')
        unrelated={f'backend/radar_platform/navigation/{name}.py' for name in ['inference','model','learning','controller','episode']}
        unrelated.add('scripts/generate_navigation_turns.py');changes={}
        for name,value in previous.items():
            if hashes.get(name)!=value:
                if name not in unrelated:raise ValueError(f'generation dependency changed: {name}')
                changes[name]={'original':value,'current':hashes.get(name)}
        write_json(root/f'resume_sources_{time.time_ns()}.json',dict(compatible_nonphysical_changes=changes))
    else:write_json(root/'source_sha256.json',hashes)
    started=time.time();write_json(root/'status.json',dict(status='running',pid=os.getpid(),total=len(records),started_unix_s=started))
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(generate_case,(base,r,str(root/'cases'/r['id']),False)) for r in records]
            for done,future in enumerate(as_completed(futures),1):
                value=future.result();print(json.dumps(dict(done=done,total=len(records),**value)),flush=True)
                write_json(root/'status.json',dict(status='running',pid=os.getpid(),done=done,total=len(records),elapsed_s=time.time()-started))
        write_json(root/'status.json',dict(status='complete',cases=len(records),frames=2*len(records),elapsed_s=time.time()-started))
    except BaseException as exc:
        write_json(root/'status.json',dict(status='failed',error=str(exc)));raise


if __name__=='__main__':main()
