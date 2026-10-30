#!/usr/bin/env python3
"""Add true stationary/terminal states inside already assigned training cells."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
import time
import numpy as np

from radar_platform.navigation.io import local_path,read_json,write_json,digest
from radar_platform.navigation.dataset import generate_case
from radar_platform.navigation.dynamics import State,Command,Limits,body_to_neu,propagate


def prepare_records(parent,count=512):
    protocol=read_json(local_path(parent)/'protocol.json')
    cells={tuple(r['cell']) for r in protocol['records'] if r['split']=='train'}
    forbidden={tuple(r['cell']) for r in protocol['records'] if r['split'] in ['val','test']}
    choices=sorted(c for c in cells if c[0] in [3,4] and c[1] in [3,4] and c[2] in [0,1,2])
    rng=np.random.default_rng(2026103051);low=np.array([-45.,-45.,22.]);width=np.array([11.25,11.25,38/6])
    records=[]
    for k in range(count):
        cell=choices[int(rng.integers(len(choices)))];stationary=k%2==0
        distance=rng.uniform(.005 if stationary else .4, .7 if k%4==0 else 4.,2)
        position=np.r_[distance*np.where(np.array(cell[:2])==3,-1.,1.),0.]
        zlo=max(24.,low[2]+cell[2]*width[2]+.25)
        zhi=min(40.,low[2]+(cell[2]+1)*width[2]-.25)
        position[2]=rng.uniform(zlo,zhi)
        yaw=float(rng.uniform(-np.pi,np.pi))
        velocity=np.zeros(3) if stationary else rng.uniform([-.3,-.3,-.25],[.3,.3,.25])
        acceleration=np.zeros(3) if stationary else rng.uniform(-.15,.15,3)
        yaw_rate=0. if stationary else float(rng.uniform(-np.pi/18,np.pi/18))
        world_velocity=body_to_neu(yaw)@velocity
        initial=State(0.,position,world_velocity,yaw)
        _,trace=propagate(initial,Command(acceleration,yaw_rate),.275569,Limits(12,3,4,2,60))
        all_cells=np.floor((np.array([s.position_neu_m for s in trace])-low)/width).astype(int)
        if not np.all(all_cells==np.array(cell)):raise AssertionError('terminal sample crosses an assigned spatial cell')
        if tuple(cell) in forbidden:raise AssertionError('held-out spatial cell must not enter terminal training')
        records.append(dict(id=f'hover_{k:05d}',split='train',cell=list(cell),stationary=stationary,
            position_neu_m=position.tolist(),velocity_neu_mps=world_velocity.tolist(),yaw_rad=yaw,
            acceleration_body_mps2=acceleration.tolist(),yaw_rate_rad_s=yaw_rate,
            phase_seed=3100000000+k,noise_seed=3200000000+k))
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=8);p.add_argument('--count',type=int,default=512)
    args=p.parse_args();parent=local_path(args.parent);root=local_path(args.output);root.mkdir(parents=True,exist_ok=True)
    base=read_json(parent/'base_config.json');records=prepare_records(parent,args.count)
    protocol=dict(records=records,counts={'train':len(records)},parent_protocol_sha256=digest(read_json(parent/'protocol.json')),
        reason='Terminal control diagnosis: original continuous draws lacked exactly stationary states and excluded the central boundary band. Supplement existing training cells only.',
        stationary_cases=sum(r['stationary'] for r in records),all_trajectory_samples_remain_in_training_cells=True,
        goal_cell_not_added='The exact goal cell (4,4,1) belongs to formal test, so training approaches the origin from neighboring training-owned cells; no held-out-cell reassignment.')
    if (root/'protocol.json').exists() and read_json(root/'protocol.json')!=protocol:raise ValueError('hover protocol changed')
    write_json(root/'protocol.json',protocol);write_json(root/'base_config.json',base)
    write_json(root/'scene_provenance.json',read_json(parent/'scene_provenance.json'))
    files=sorted(local_path('backend/radar_platform').rglob('*.py'))+[local_path(__file__)]
    if not (root/'source_sha256.json').exists():
        write_json(root/'source_sha256.json',{str(f.relative_to(local_path('.'))):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
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
