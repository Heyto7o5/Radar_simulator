#!/usr/bin/env python3
"""Civil low-speed offline center-landmark localization feasibility experiment.

Uses the agricultural pilot's 30m height and 3m/s horizontal motion. No legacy
high-speed configuration, control policy or release command is used here.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import itertools
import json
from pathlib import Path
import zipfile
import numpy as np

from run_fertilizer_rda_pilot import acquire, make_scene, write
from radar_platform.dem_buildings import BuildingSpec, add_dem_buildings
from radar_platform.unified.schema import Simulation


def center_scene(bare):
    return add_dem_buildings(bare, [BuildingSpec(id='center_landmark', center_ne_m=(0, 0),
        footprint_ne_m=(6, 8), height_above_highest_ground_m=5,
        max_foundation_relief_m=10, relative_reflectivity=.02, divisions=2)])[0]


def feature(rda, mode):
    a = np.asarray(rda, np.float32)
    if mode == 'one_frame_rda':
        a = a[:1]
    elif mode == 'two_frame_rd':
        a = np.sqrt(np.sum(a*a, axis=-1, keepdims=True))
    elif mode != 'two_frame_rda':
        raise ValueError(mode)
    a = np.sqrt(a).reshape(len(a), -1)
    a /= np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-20)
    return (a.ravel()/np.float32(np.sqrt(len(a)))).astype(np.float32)


def predict(bank, labels, observation, mode):
    x = feature(observation, mode)
    distance = np.maximum(0, np.sum(bank*bank, axis=1)+np.dot(x,x)-2*(bank@x))
    ix = np.argsort(distance)[:4]
    weights = 1/np.maximum(distance[ix], 1e-6)**2
    return np.average(labels[ix], axis=0, weights=weights)


def generate(job):
    cfg, state, folder, seed, save = job
    _, meta = acquire(cfg, state, folder, seed=seed, save_full=save)
    return dict(case=str(folder), seconds=meta['elapsed_s'], applicability=meta['applicability']['status'])


def read_rda(folder):
    with np.load(folder/'rda.npz', allow_pickle=False) as f:
        return f['magnitude']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--terrain-only', action='store_true', help='Paired control with identical states and no building')
    args = p.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    bare, _, provenance = make_scene(args.source)
    cfg = bare if args.terrain_only else center_scene(bare)
    provenance.pop('placement', None)  # make_scene's three-building example is not this scene.
    provenance['objects'] = [obj.model_dump(mode='json') for obj in cfg.scene.objects]
    fine, _, _ = make_scene(args.source, spacing=5)
    fine = fine if args.terrain_only else center_scene(fine)
    train = np.array([[n,e,3.,0.] for n,e in itertools.product([-18.,-9.,0.,9.,18.],[-12.,-6.,0.,6.,12.])])
    rng = np.random.default_rng(20261029)
    test = np.c_[rng.uniform([-16,-10],[16,10],(16,2)), np.full(16,3.),np.zeros(16)]
    assert not np.any(np.all(train[:,None,:]==test[None,:,:],axis=-1))
    protocol = dict(train=train.tolist(),test=test.tolist(),terrain_only=args.terrain_only,fixed_altitude_m=30,
        fixed_velocity_ne_mps=[3,0],center_building_size_ne_m=None if args.terrain_only else [6,8],height_above_highest_ground_m=None if args.terrain_only else 5,
        neighbors=4,phase_stress_test_indices=list(range(8)),grid5_test_indices=list(range(4)),
        scope='fixed calibrated civil scene, full signed horizontal position; no controller or fertilizer landing evaluation',
        input='numeric RDA magnitudes only; fixed acquisition and scene are encoded in calibration',
        selection='predeclared 25-state grid, fresh 16 off-grid states; no tuning on test results')
    write(root/'protocol.json',protocol);write(root/'provenance.json',provenance)
    (root/'center_config.json').write_text(cfg.model_dump_json(indent=2))
    files=sorted(Path('backend/radar_platform').rglob('*.py'))+[Path(__file__),Path('scripts/run_fertilizer_rda_pilot.py')]
    with zipfile.ZipFile(root/'source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for path in files:z.write(path,str(path))
    write(root/'source_sha256.json',{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    jobs=[]
    for group,states,scene,seed in [('calibration',train,cfg,42),('test',test,cfg,42),
                                  ('phase',test[:8],cfg,31415),('grid5',test[:4],fine,42)]:
        for k,state in enumerate(states):
            jobs.append((scene,state,root/group/f'{k:03d}',seed,k==0))
    # Symmetric isolated-reflector control: no DEM texture or asymmetric phases.
    raw=bare.model_dump(mode='json')
    raw['scene']['terrain']=None
    raw['scene']['objects']=[dict(kind='point',id='center_reference',position_ned_m=[0,0,0],amplitude=1)]
    point=Simulation.model_validate(raw)
    for k,e in enumerate([-8.,8.]):jobs.append((point,np.array([-12.,e,3.,0.]),root/'mirror'/f'{k:03d}',42,True))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for k,future in enumerate(as_completed([pool.submit(generate,j) for j in jobs]),1):
            print(json.dumps(dict(completed=k,total=len(jobs),**future.result())),flush=True)
    a,b=[read_rda(root/'mirror'/f'{k:03d}') for k in range(2)]
    rd_a,rd_b=np.sum(a*a,axis=-1),np.sum(b*b,axis=-1)
    ambiguity=dict(equal_horizontal_distances=True,
        relative_rd_power_difference=float(np.linalg.norm(rd_a-rd_b)/np.linalg.norm(rd_a)),
        relative_rda_difference=float(np.linalg.norm(a-b)/np.linalg.norm(a)),
        reflected_angle_rda_difference=float(np.linalg.norm(a-b[:,:,:,::-1])/np.linalg.norm(a)))
    write(root/'mirror_control.json',ambiguity)
    labels=np.c_[train[:,:2],np.linalg.norm(train[:,:2],axis=1)]
    modes=['one_frame_rda','two_frame_rd','two_frame_rda']
    banks={m:np.stack([feature(read_rda(root/'calibration'/f'{k:03d}'),m) for k in range(len(train))]) for m in modes}
    np.savez_compressed(root/'calibration.npz',labels=labels,**banks)
    rows=[]
    for group,count in [('test',16),('phase',8),('grid5',4)]:
        for k in range(count):
            observation=read_rda(root/group/f'{k:03d}')
            truth=test[k,:2];radius=np.linalg.norm(truth)
            for mode in modes:
                pred=predict(banks[mode],labels,observation,mode)
                norm=np.linalg.norm(pred[:2])
                angle=np.degrees(np.arccos(np.clip(np.dot(pred[:2],truth)/max(norm*radius,1e-15),-1,1)))
                rows.append(dict(group=group,case=k,mode=mode,true_ne_m=truth.tolist(),predicted_ne_m=pred[:2].tolist(),
                    true_radius_m=float(radius),predicted_radius_m=float(pred[2]),
                    position_error_m=float(np.linalg.norm(pred[:2]-truth)),radius_error_m=float(abs(pred[2]-radius)),
                    bearing_error_deg=float(angle),signs_correct=bool(np.all(np.sign(pred[:2])==np.sign(truth))),
                    improved_over_center_guess=bool(np.linalg.norm(pred[:2]-truth)<radius)))
    write(root/'results.json',rows)
    summary={}
    for group in ['test','phase','grid5']:
        summary[group]={}
        for mode in modes:
            rr=[r for r in rows if r['group']==group and r['mode']==mode]
            summary[group][mode]=dict(n=len(rr),position_mae_m=float(np.mean([r['position_error_m'] for r in rr])),
                position_p90_m=float(np.percentile([r['position_error_m'] for r in rr],90)),
                radius_mae_m=float(np.mean([r['radius_error_m'] for r in rr])),
                bearing_mae_deg=float(np.mean([r['bearing_error_deg'] for r in rr])),
                signs_correct=sum(r['signs_correct'] for r in rr),improved=sum(r['improved_over_center_guess'] for r in rr),
                center_guess_mean_error_m=float(np.mean([r['true_radius_m'] for r in rr])))
    write(root/'summary.json',summary)
    print(json.dumps(dict(summary=summary,mirror_control=ambiguity)),flush=True)


if __name__=='__main__':main()
