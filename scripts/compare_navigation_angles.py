#!/usr/bin/env python3
"""Paired real-DEM signed-position probe for planar versus original linear arrays.

Fresh phase/continuous-state holdouts include mirrored forward coordinates,
heading changes, stationary, 0.2m/s and 3m/s motion. This is an offline diagnostic,
not a substitute for the learned closed-loop campaign.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import itertools
import json
import os
import time
import numpy as np

from radar_platform.navigation.dataset import generate_case
from radar_platform.navigation.dynamics import body_to_neu
from radar_platform.navigation.io import local_path,read_json,write_json


def records():
    rng=np.random.default_rng(2026103023);rows=[]
    groups=list(itertools.product([5.,15.,30.],[0.,8.],[0.,np.pi/2],[0.,.2,3.]))
    for split in ['calibration','test']:
        for group,(forward,right,yaw,speed) in enumerate(groups):
            x=forward;y=right;z=30.;psi=yaw;v=speed
            if split=='test':
                x+=rng.uniform(-1,1);y+=rng.uniform(-.7,.7);z+=rng.uniform(-2,2);psi+=rng.uniform(-.15,.15)
                if speed:v+=rng.uniform(-.1,.1)*min(speed,1)
            for sign in [-1,1]:
                body=np.array([sign*x,y,-z]);p=-body_to_neu(psi)@body
                rows.append(dict(id=f'{split}_{group:03d}_{"p" if sign>0 else "n"}',split=split,cell=None,
                    position_neu_m=p.tolist(),velocity_neu_mps=(body_to_neu(psi)@np.array([v,0.,0.])).tolist(),
                    yaw_rad=float(psi),acceleration_body_mps2=[0.,0.,0.],yaw_rate_rad_s=0.,
                    phase_seed=42 if split=='calibration' else 1800000+group,noise_seed=1900000+100*(split=='test')+2*group+(sign>0),
                    group=group,forward_sign=sign,nominal_speed_mps=speed))
    return rows


def feature(folder):
    x=np.load(folder/'input.npy',allow_pickle=False)[:,::2,::2,::2,::2].astype(np.float32).ravel()
    return x/max(float(np.linalg.norm(x)),1e-10)


def evaluate(root,rows):
    result={}
    for condition in ['planar','linear']:
        cases=root/condition/'cases';train=[r for r in rows if r['split']=='calibration'];test=[r for r in rows if r['split']=='test']
        bank=np.stack([feature(cases/r['id']) for r in train])
        labels=np.array([read_json(cases/r['id']/'truth.json')['state_body'] for r in train]);outputs=[]
        for r in test:
            x=feature(cases/r['id']);distance=np.maximum(0,2-2*(bank@x));ix=np.argsort(distance)[:4]
            weights=1/np.maximum(distance[ix],1e-6)**2;prediction=np.average(labels[ix],axis=0,weights=weights)
            truth=np.array(read_json(cases/r['id']/'truth.json')['state_body'])
            outputs.append(dict(id=r['id'],speed=r['nominal_speed_mps'],prediction=prediction.tolist(),truth=truth.tolist(),
                position_error_m=float(np.linalg.norm(prediction[:3]-truth[:3])),
                forward_sign_correct=bool(np.sign(prediction[0])==np.sign(truth[0]))))
        result[condition]={}
        for speed in [0.,.2,3.]:
            selected=[x for x in outputs if x['speed']==speed]
            result[condition][str(speed)]=dict(cases=len(selected),mean_position_error_m=float(np.mean([x['position_error_m'] for x in selected])),
                p90_position_error_m=float(np.percentile([x['position_error_m'] for x in selected],90)),
                forward_sign_correct=sum(x['forward_sign_correct'] for x in selected))
        write_json(root/condition/'predictions.json',outputs)
    write_json(root/'comparison.json',dict(metrics=result,neighbors=4,
        scope='Matched-state real-DEM offline diagnostic with fresh phase and continuous-state perturbations; unequal aperture/channel counts explicitly retained; no flight-success claim'))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-scene',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=12)
    args=p.parse_args();root=local_path(args.output);root.mkdir(parents=True,exist_ok=True);rows=records()
    planar=read_json(args.base_scene);linear=json.loads(json.dumps(planar))
    linear['array']=dict(layout='ula',num_tx=2,num_rx=8,loop_period_s=.0004,tx_offsets_s=[0.,.0003])
    linear['processing']['elevation_deg']=[0.]
    linear['input_provenance'].append('Matched-CPI original 2Tx8Rx linear-array control; one angle only, unchanged physical states and scene.')
    from radar_platform.unified.schema import Simulation
    configs={name:Simulation.model_validate(raw).model_dump(mode='json') for name,raw in [('planar',planar),('linear',linear)]}
    protocol=dict(records=rows,configurations=configs,neighbors=4,
        calibration_cases=72,heldout_cases=72,split='new continuous states and phase seeds; mirror pairs share a scene realization',
        equality='same 0.4ms same-Tx period, CPI end, bandwidth, range/Doppler grids, states, terrain and noise seeds',
        difference='planar 64 virtual channels / two angles versus original linear 16 virtual channels / one angle')
    if (root/'protocol.json').exists() and read_json(root/'protocol.json')!=protocol:raise ValueError('probe protocol changed')
    write_json(root/'protocol.json',protocol);started=time.time()
    write_json(root/'status.json',dict(status='running',pid=os.getpid(),total=288))
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(generate_case,(cfg,r,str(root/condition/'cases'/r['id']),k==0)):condition
                     for condition,cfg in configs.items() for k,r in enumerate(rows)}
            for done,future in enumerate(as_completed(futures),1):
                value=future.result();print(json.dumps(dict(condition=futures[future],done=done,total=len(futures),**value)),flush=True)
                write_json(root/'status.json',dict(status='running',pid=os.getpid(),done=done,total=len(futures),elapsed_s=time.time()-started))
        metrics=evaluate(root,rows)
        write_json(root/'status.json',dict(status='complete',cases=288,elapsed_s=time.time()-started,metrics=metrics))
    except BaseException as exc:
        write_json(root/'status.json',dict(status='failed',error=str(exc)));raise


if __name__=='__main__':main()
