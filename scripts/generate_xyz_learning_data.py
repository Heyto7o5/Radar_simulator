#!/usr/bin/env python3
"""Fresh independent XYZ labels and physically generated low-altitude RDA pairs."""
import argparse, hashlib, itertools, json, os, time, zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np

from run_fertilizer_rda_pilot import make_scene, write
from validate_center_landmark import center_scene
from radar_platform.unified.schema import Simulation
from radar_platform.unified.session import SimulationSession
from radar_platform.xyz_features import encode_rda


def prepare_cases():
    rng = np.random.default_rng(2026102901)
    cells = list(itertools.product(range(8), range(6), range(6)))
    order = rng.permutation(len(cells))
    lower = np.array([-18., -12., 22.]); width=np.array([4.5,4.,20/6])
    records=[]
    for split, chosen in [('train',order[:192]),('val',order[192:224]),('test',order[224:])]:
        for index in chosen:
            cell=cells[index]
            for repeat in range(2):
                xyz=lower+width*(np.array(cell)+rng.uniform(.1,.9,3))
                records.append(dict(id=f'{split}_{index:03d}_{repeat}',split=split,cell=list(cell),xyz_m=xyz.tolist(),
                                    phase_seed=int(rng.integers(0,2**31)),noise_seed=int(rng.integers(0,2**31))))
    tests=[r for r in records if r['split']=='test']
    for kind in ['grid5','building_height','noise10']:
        for original in tests[:8]:
            r=dict(original);r.update(id=kind+'_'+original['id'],split=kind,paired_test_id=original['id'])
            records.append(r)
    return records


def acquire_case(job):
    raw, record, folder = job
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    raw['platform']['position_ned_m']=[*record['xyz_m'][:2],-record['xyz_m'][2]]
    raw['scene']['seed']=record['phase_seed']
    raw['receiver_noise']['seed']=record['noise_seed']
    cfg=Simulation.model_validate(raw)
    text=cfg.model_dump_json(indent=2)
    digest=hashlib.sha256(text.encode()).hexdigest()
    if (folder/'complete.json').exists():
        meta=json.loads((folder/'complete.json').read_text())
        if meta['config_sha256']!=digest:raise ValueError('cached config mismatch')
        return meta
    session=SimulationSession(cfg);session.save_inputs(folder)
    start=time.perf_counter();frames=[]
    for k in range(2):
        frame=session.frame(k)
        frames.append(np.abs(frame.products.rad[:,:,0,:]).astype(np.float32))
        if record.get('save_adc'):
            session.save_frame(folder,frame)
    magnitude=np.stack(frames)
    axes=dict(range_m=frame.products.range_m,velocity_mps=frame.products.velocity_mps,
              azimuth_deg=frame.products.azimuth_deg,frame_times_s=np.array([0.,1.]))
    np.savez_compressed(folder/'rda.tmp.npz',magnitude=magnitude,**axes)
    (folder/'rda.tmp.npz').replace(folder/'rda.npz')
    np.save(folder/'input.npy',encode_rda(magnitude))
    meta=dict(id=record['id'],config_sha256=digest,elapsed_s=time.perf_counter()-start,
              applicability=session.estimate['applicability']['status'])
    write(folder/'complete.json',meta)
    return meta


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=16)
    args=p.parse_args();root=args.output;root.mkdir(parents=True,exist_ok=True)
    bare,_,provenance=make_scene(args.source)
    coarse=center_scene(bare)
    fine,_,_=make_scene(args.source,spacing=5);fine=center_scene(fine)
    bases={}
    for kind,cfg in [('normal',coarse),('grid5',fine),('building_height',coarse),('noise10',coarse)]:
        raw=cfg.model_dump(mode='json')
        raw['name']='Civil fixed-scene RDA-to-XYZ learning; independent height'
        raw['waveform'].update(samples=192,sample_rate_hz=3e6)
        raw['processing']['range_limits_m']=[8.,120.]
        raw['receiver_noise']=dict(variance=1e-4 if kind=='noise10' else 1e-5,seed=0,
                                  provenance='Fixed assumed complex ADC noise variance; not signal-normalized SNR')
        raw['input_provenance']=[s for s in raw['input_provenance'] if 'altitude 30m' not in s and 'drop:' not in s and 'setpoint' not in s]
        raw['input_provenance'] += ['XYZ learning labels: x=North,y=East,z=Up relative to DEM center at CPI start.',
                                    'Height independently varies 22..42m. Fixed level attitude and velocity [3,0,0] m/s.',
                                    'Only RDA magnitudes enter the learned predictor; scene encoded by training.']
        if kind=='building_height':
            obj=raw['scene']['objects'][0];obj['size_m'][2]+=2;obj['position_ned_m'][2]-=1
        bases[kind]=Simulation.model_validate(raw).model_dump(mode='json')
    records=prepare_cases()
    for split in sorted({r['split'] for r in records}):
        next(r for r in records if r['split']==split)['save_adc']=True
    protocol=dict(records=records,coordinate_order=['north_m','east_m','up_m'],label_time_s=0,
        ranges_m=[[-18,18],[-12,12],[22,42]],split_policy='whole 3D cells; 0.1-cell margins; two states per cell',
        counts={s:sum(r['split']==s for r in records) for s in sorted({r['split'] for r in records})},
        fixed_velocity_ne_mps=[3,0],fixed_attitude=True,frames=2,frame_interval_s=1,
        scope='offline low-altitude fixed-scene regression; no control or release model',
        stress_scope='paired first eight heldout states; never used to select checkpoints')
    if (root/'protocol.json').exists():
        if json.loads((root/'protocol.json').read_text())!=protocol:raise ValueError('existing protocol mismatch')
    else:write(root/'protocol.json',protocol)
    provenance.pop('placement',None);provenance['objects']=bases['normal']['scene']['objects']
    write(root/'scene_provenance.json',provenance)
    write(root/'base_config.json',bases['normal'])
    files=sorted(Path('backend/radar_platform').rglob('*.py'))+[Path(__file__),Path('scripts/run_fertilizer_rda_pilot.py'),Path('scripts/validate_center_landmark.py')]
    hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    if (root/'source_sha256.json').exists():
        if json.loads((root/'source_sha256.json').read_text())!=hashes:raise ValueError('source changed; use new output')
    else:
        write(root/'source_sha256.json',hashes)
        with zipfile.ZipFile(root/'generation_source.zip','x',zipfile.ZIP_DEFLATED) as archive:
            for f in files:archive.write(f,str(f))
    write(root/'status.json',dict(status='generating',pid=os.getpid(),total=len(records)))
    jobs=[(bases.get(r['split'],bases['normal']),r,root/'cases'/r['id']) for r in records]
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for done,future in enumerate(as_completed([pool.submit(acquire_case,j) for j in jobs]),1):
                meta=future.result();print(json.dumps(dict(done=done,total=len(jobs),**meta)),flush=True)
                write(root/'status.json',dict(status='generating',pid=os.getpid(),done=done,total=len(records)))
        for split in protocol['counts']:
            selected=[r for r in records if r['split']==split]
            inputs=np.stack([np.load(root/'cases'/r['id']/'input.npy') for r in selected])
            labels=np.array([r['xyz_m'] for r in selected],dtype=np.float32)
            np.save(root/f'{split}_x.npy',inputs);np.save(root/f'{split}_y.npy',labels)
        write(root/'status.json',dict(status='complete',cases=len(records),frames=2*len(records)))
    except Exception as exc:
        write(root/'status.json',dict(status='failed',error=str(exc)))
        raise


if __name__=='__main__':main()
