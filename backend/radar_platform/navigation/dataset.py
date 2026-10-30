"""Physical short trajectories, labels kept apart from radar observations."""
import hashlib
import itertools
import time
import numpy as np

from .scene import navigation_scene
from .dynamics import State,Command,Limits,propagate,body_to_neu
from .acquisition import RadarWorld,pair
from .observation import encode
from .io import local_path,write_json,read_json,digest


def pilot_records():
    records=[]
    for k,(bearing,radius,height,speed) in enumerate(itertools.product(range(8),[2.,30.],[30.,45.],[0.,3.])):
        phi=bearing*np.pi/4
        position=np.array([radius*np.cos(phi),radius*np.sin(phi),height])
        yaw=phi+np.pi+np.deg2rad(60 if k%2 else -60)
        records.append(dict(id=f'pilot_{k:04d}',split='pilot',cell=None,
            position_neu_m=position.tolist(),velocity_neu_mps=(body_to_neu(yaw)@np.array([speed,0,0])).tolist(),
            yaw_rad=float(yaw),acceleration_body_mps2=[0.,0.,0.],yaw_rate_rad_s=0.,
            phase_seed=100_000+k,noise_seed=200_000+k))
    return records


def learning_records(counts=None,seed=2026103001):
    counts=counts or {'train':4096,'val':512,'test':1024}
    rng=np.random.default_rng(seed)
    cells=np.array(list(itertools.product(range(8),range(8),range(6))))
    order=rng.permutation(len(cells))
    partitions={'train':order[:256],'val':order[256:304],'test':order[304:]}
    low=np.array([-45.,-45.,22.]);width=np.array([90/8,90/8,38/6])
    result=[]
    for split,count in counts.items():
        indices=partitions[split]
        centers=low+(cells[indices]+.5)*width
        # Additional near-hover training without sharing spatial cells across splits.
        weight=.3+np.exp(-np.sum(centers[:,:2]**2,axis=1)/350-(centers[:,2]-30)**2/90)
        for k in range(count):
            index=int(rng.choice(indices,p=weight/weight.sum()))
            position=low+(cells[index]+rng.uniform(.12,.88,3))*width
            yaw=float(rng.uniform(-np.pi,np.pi))
            mode=k%5
            speed=float(rng.uniform(0,3) if mode<3 else rng.uniform(3,12))
            if mode==0: speed=float(rng.uniform(0,.3))
            velocity=np.array([speed,rng.uniform(-.6,.6),rng.uniform(-1,1)])
            if mode==4:velocity[2]=rng.uniform(-3,3)
            acceleration=rng.uniform([-1.2,-1.2,-.5],[1.2,1.2,.5])
            yaw_rate=float(rng.uniform(-45,45))
            if mode==0:acceleration*=.1;yaw_rate*=.1
            result.append(dict(id=f'{split}_{k:05d}',split=split,cell=cells[index].tolist(),
                position_neu_m=position.tolist(),velocity_neu_mps=(body_to_neu(yaw)@velocity).tolist(),
                yaw_rad=yaw,acceleration_body_mps2=acceleration.tolist(),yaw_rate_rad_s=float(np.deg2rad(yaw_rate)),
                phase_seed=int(rng.integers(1,2**32-1)),noise_seed=int(rng.integers(1,2**32-1))))
    return result


def generate_case(job):
    base_raw,record,folder,save_full=job
    folder=local_path(folder);folder.mkdir(parents=True,exist_ok=True)
    key=digest(dict(base=base_raw,record=record))
    if (folder/'complete.json').exists():
        complete=read_json(folder/'complete.json')
        if complete['configuration_sha256']!=key:raise ValueError('cached configuration mismatch')
        return complete
    from ..unified.schema import Simulation
    raw=dict(base_raw)
    raw['scene']={**base_raw['scene'],'seed':record['phase_seed']}
    raw['receiver_noise']={**base_raw['receiver_noise'],'seed':record['noise_seed']}
    cfg=Simulation.model_validate(raw)
    world=RadarWorld(cfg)
    state=State(0.,np.array(record['position_neu_m']),np.array(record['velocity_neu_mps']),record['yaw_rad'])
    command=Command(np.array(record['acceleration_body_mps2']),record['yaw_rate_rad_s'])
    limits=Limits(12.,3.,4.,2.,60.)
    started=time.perf_counter();frames=[];diagnostics=[];trajectory=[]
    for index in range(2):
        start=index*.25
        if start>state.time_s:
            state,samples=propagate(state,command,start,limits)
            trajectory.extend(samples[1:])
        state,samples=propagate(state,command,start+world.duration_s,limits)
        trajectory.extend(samples if not trajectory else samples[1:])
        frame,diagnostic,request=world.acquire(samples,index,
            save_directory=folder/f'full_{index}' if save_full else None,assessment=True)
        frames.append(frame);diagnostics.append(diagnostic)
        if not save_full:
            # Exact request allows direct replay through the common engine.
            write_json(folder/f'request_{index}.json',request.model_dump(mode='json'))
    observation=pair(*frames)
    np.savez_compressed(folder/'rda.tmp.npz',**observation.arrays())
    (folder/'rda.tmp.npz').replace(folder/'rda.npz')
    np.save(folder/'input.tmp.npy',encode(observation).astype(np.float16))
    (folder/'input.tmp.npy').replace(folder/'input.npy')
    coverage=diagnostics[-1]['visibility']['coverage']
    if 'center_landmark' in coverage['objects']:
        ix=coverage['objects'].index('center_landmark')
        selected=np.array(coverage['object_index'])==ix
        echoes=np.array(coverage['sampled_echo_channel_slots'])[selected]
        visible=int(np.count_nonzero(echoes))
        valid=visible>0
    else:
        visible=None;valid=True
    write_json(folder/'truth.json',dict(record=record,state_body=state.label().tolist(),
        label_time_s=state.time_s,valid_landmark_observation=valid,building_contributing_cells=visible,
        final_position_neu_m=state.position_neu_m.tolist(),final_velocity_neu_mps=state.velocity_neu_mps.tolist(),
        final_yaw_rad=state.yaw_rad))
    # Large per-scatterer visibility arrays stay in the diagnostic sidecar, not inputs.
    write_json(folder/'diagnostics.json',diagnostics)
    complete=dict(id=record['id'],configuration_sha256=key,elapsed_s=time.perf_counter()-started,
        rda_shape=list(observation.magnitude.shape),radar_seconds=sum(d['elapsed_s'] for d in diagnostics),
        valid=valid,building_contributing_cells=visible,
        applicability=[d['applicability']['status'] for d in diagnostics])
    write_json(folder/'complete.json',complete)
    return complete
