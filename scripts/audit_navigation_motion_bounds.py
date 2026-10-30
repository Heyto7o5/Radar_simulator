#!/usr/bin/env python3
"""Check all scene scatterers at sampled Tx/fast times across motion limits."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import itertools
import json
import numpy as np

from radar_platform.navigation.scene import navigation_scene
from radar_platform.navigation.dynamics import State,Command,Limits,body_to_neu,propagate,sampled_trajectory
from radar_platform.navigation.timeline import covered_cpi_end
from radar_platform.navigation.io import local_path,write_json
from radar_platform.unified.schema import Simulation
from radar_platform.unified.preflight import preflight
from radar_platform.unified.motion_check import motion_error_report


def inspect_case(job):
    record,base=job;start=32.
    yaw=np.deg2rad(record['yaw_deg']);speed=record['speed_mps']
    state=State(start,np.array([-24.,-18.,record['height_m']]),body_to_neu(yaw)@np.array([speed,0.,0.]),yaw)
    limits=Limits(max(3.,speed),1.,record['acceleration_mps2'],1.,record['yaw_rate_deg_s'])
    command=Command(np.array([0.,record['acceleration_mps2'],1.]),np.deg2rad(record['yaw_rate_deg_s']))
    cfg=Simulation.model_validate(base)
    _,samples=propagate(state,command,covered_cpi_end(cfg,start),limits)
    raw=cfg.model_dump(mode='json');raw['platform']['trajectory']=sampled_trajectory(samples).model_dump(mode='json')
    raw['sequence'].update(start_time_s=start,frame_count=1)
    request=Simulation.model_validate(raw);_,estimate=preflight(request)
    motion=motion_error_report(request,max_scatterers=4096,max_slots=8,fast_samples=5)
    return dict(parameters=record,applicability=estimate['applicability'],motion=motion)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();root=local_path(args.output);root.mkdir(parents=True,exist_ok=True)
    cfg,_=navigation_scene();records=[]
    for h,v,yaw in itertools.product([22.,60.],[3.,12.],[0.,90.,180.,270.]):
        records.append(dict(height_m=h,speed_mps=v,yaw_deg=yaw,acceleration_mps2=4. if v==12 else 2.,yaw_rate_deg_s=60. if v==12 else 45.))
    write_json(root/'protocol.json',dict(records=records,all_scatterers=True,sampled_slots=8,sampled_fast_times=5,
        scope='Numerical within-slot approximation check including combined parameter boundaries; not evidence that guidance succeeds at these combinations'))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(inspect_case,(r,cfg.model_dump(mode='json'))) for r in records]
        for f in as_completed(futures):rows.append(f.result())
    summary=dict(cases=len(rows),motion_usable=sum(r['motion']['status']=='usable' for r in rows),
        max_unwrapped_phase_error_rad=max(r['motion']['metrics']['max_unwrapped_phase_error_rad'] for r in rows),
        phase_tolerance_rad=float(np.pi/8),all_scatterers_checked=all(r['motion']['coverage']['scatterers_checked']==r['motion']['coverage']['scatterers_total'] for r in rows),
        limitations='Time sampling is not an exhaustive bound; both reference and simulator neglect retarded propagation. Applicability risks are retained.')
    write_json(root/'results.json',rows);write_json(root/'summary.json',summary);print(json.dumps(summary))


if __name__=='__main__':main()
