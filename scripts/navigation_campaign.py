#!/usr/bin/env python3
"""Predeclared normal and one-factor stress civil-navigation episodes."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import copy
import json
import os
import numpy as np
import psutil

from radar_platform.navigation.episode import default_episode,run_episode
from radar_platform.navigation.io import local_path,write_json,read_json


def cases():
    records=[]
    for bearing in range(8):
        for repeat in range(3):
            spec=default_episode();phi=bearing*np.pi/4
            error=np.deg2rad((30 if repeat==0 else 60)*(1 if bearing%2 else -1))
            yaw=phi+np.pi+error
            spec.update(id=f'normal_{bearing}_{repeat}',position_neu_m=[30*np.cos(phi),30*np.sin(phi),[25.,30.,40.][repeat]],
                velocity_neu_mps=[3*np.cos(yaw),3*np.sin(yaw),0.],yaw_rad=float(yaw),
                phase_seed=500_000+10*bearing+repeat,noise_seed=600_000+10*bearing+repeat)
            records.append(spec)
    for condition,value in [('speed',6.),('speed',12.),('heading',90.),('heading',120.),
                            ('height',22.),('height',60.),('interval',1.),('noise',10.)]:
        for repeat in range(3):
            spec=default_episode();spec['id']=f'stress_{condition}_{value:g}_{repeat}'
            spec['phase_seed']=700000+len(records);spec['noise_seed']=800000+len(records)
            if condition=='speed':
                yaw=spec['yaw_rad'];spec['velocity_neu_mps']=[value*np.cos(yaw),value*np.sin(yaw),0.]
                spec['limits'].update(horizontal_speed_mps=value,horizontal_acceleration_mps2=4.,yaw_rate_deg_s=60.)
            elif condition=='heading':
                yaw=np.arctan2(18,24)+np.deg2rad(value);spec['yaw_rad']=float(yaw)
                spec['velocity_neu_mps']=[3*np.cos(yaw),3*np.sin(yaw),0.]
            elif condition=='height':spec['position_neu_m'][2]=value
            elif condition=='interval':spec['correction_interval_s']=value
            elif condition=='noise':spec['noise_multiplier']=value
            records.append(spec)
    return records


def worker(job):
    spec,output,model,mode,state_filter,gentle=job
    try:return run_episode(spec,output,mode=mode,model_dir=model,threads=2,state_filter=state_filter,gentle_terminal=gentle)
    except Exception as exc:
        folder=local_path(output)
        if folder.exists() and not (folder/'complete.json').exists():
            write_json(folder/'failure.json',dict(kind='execution_error',exception=type(exc).__name__,message=str(exc)))
            write_json(folder/'status.json',dict(status='failed',message=str(exc)))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--model-dir')
    p.add_argument('--mode',choices=['learned','oracle','no_correction'],default='learned')
    p.add_argument('--workers',type=int,default=16);p.add_argument('--normal-only',action='store_true')
    p.add_argument('--terrain-only',action='store_true');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--filter-state',action='store_true')
    p.add_argument('--gentle-terminal',action='store_true')
    args=p.parse_args();root=local_path(args.output);root.mkdir(parents=True,exist_ok=True)
    records=cases()
    if args.normal_only:records=[r for r in records if r['id'].startswith('normal')]
    if args.terrain_only:
        for r in records:r['terrain_only']=True
    protocol=dict(records=records,mode=args.mode,model_dir=args.model_dir,state_filter=args.filter_state,gentle_terminal=args.gentle_terminal)
    if (root/'protocol.json').exists() and read_json(root/'protocol.json')!=protocol:raise ValueError('campaign protocol changed')
    write_json(root/'protocol.json',protocol)
    if args.prepare_only:print(json.dumps(dict(cases=len(records),status='prepared')));return
    results=[]
    process_time=psutil.Process(os.getpid()).create_time()
    write_json(root/'status.json',dict(status='running',pid=os.getpid(),process_create_time=process_time,done=0,total=len(records)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(worker,(r,str(root/r['id']),args.model_dir,args.mode,args.filter_state,args.gentle_terminal)) for r in records]
        for future in as_completed(futures):
            try:result=future.result()
            except Exception as exc:
                write_json(root/'status.json',dict(status='failed',pid=os.getpid(),process_create_time=process_time,
                    done=len(results),total=len(records),error=str(exc)))
                for pending in futures:pending.cancel()
                raise
            results.append(result);write_json(root/'results.json',results)
            write_json(root/'status.json',dict(status='running',pid=os.getpid(),process_create_time=process_time,done=len(results),total=len(records)))
            print(json.dumps(dict(done=len(results),total=len(records),**result)),flush=True)
    summary={}
    for group in ['normal','stress']:
        selected=[r for r in results if r['episode'].startswith(group)]
        if selected:summary[group]=dict(cases=len(selected),successes=sum(r['success'] for r in selected),
            success_rate=float(np.mean([r['success'] for r in selected])),
            final_horizontal_mean_m=float(np.mean([r['horizontal_error_m'] for r in selected])),
            final_horizontal_p90_m=float(np.percentile([r['horizontal_error_m'] for r in selected],90)))
    write_json(root/'summary.json',summary);write_json(root/'status.json',dict(status='complete',cases=len(results)))


if __name__=='__main__':main()
