#!/usr/bin/env python3
"""Paired no-building observations: identical terrain, states, seeds and radar."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
import time

from radar_platform.navigation.io import local_path,read_json,write_json,digest
from radar_platform.navigation.dataset import generate_case


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True);p.add_argument('--turns',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=24)
    args=p.parse_args();base=local_path(args.base);turns=local_path(args.turns);root=local_path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    parents=[read_json(x/'protocol.json') for x in [base,turns]]
    records=[r for parent in parents for r in parent['records']]
    cfg=read_json(base/'base_config.json');cfg['scene']['objects']=[]
    cfg['input_provenance']=[x for x in cfg['input_provenance'] if not x.startswith('Synthetic axis-aligned buildings:')]
    cfg['input_provenance'].append('Paired terrain-only control: no synthetic building; same state/terrain/phase/noise acquisition as building experiment.')
    provenance=read_json(base/'scene_provenance.json');provenance['placement']=[]
    protocol=dict(records=records,counts={s:sum(r['split']==s for r in records) for s in ['train','val','test']},
        paired_parent_protocol_sha256=[digest(p) for p in parents],terrain_only=True,
        scope='Fresh training required; building-conditioned weights are not reused',
        input='Numeric dual-angle radar magnitude frames only; same acquisition and held-out cell allocation')
    if (root/'protocol.json').exists() and read_json(root/'protocol.json')!=protocol:raise ValueError('paired protocol changed')
    write_json(root/'protocol.json',protocol);write_json(root/'base_config.json',cfg);write_json(root/'scene_provenance.json',provenance)
    files=sorted(local_path('backend/radar_platform').rglob('*.py'))+[local_path(__file__)]
    hashes={str(f.relative_to(local_path('.'))):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    if (root/'source_sha256.json').exists():
        previous=read_json(root/'source_sha256.json')
        unrelated={f'backend/radar_platform/navigation/{name}.py' for name in ['inference','model','learning','controller','episode']}
        unrelated.add('scripts/generate_navigation_terrain_control.py');changes={}
        for name,value in previous.items():
            if hashes.get(name)!=value:
                if name not in unrelated:raise ValueError(f'generation dependency changed: {name}')
                changes[name]={'original':value,'current':hashes.get(name)}
        write_json(root/f'resume_sources_{time.time_ns()}.json',dict(compatible_nonphysical_changes=changes,
            reason='Original source manifest retained; physics/scene/trajectory/observation/dataset dependencies unchanged; cached configurations checked individually.'))
    else:write_json(root/'source_sha256.json',hashes)
    started=time.time();write_json(root/'status.json',dict(status='running',pid=os.getpid(),total=len(records),started_unix_s=started))
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(generate_case,(cfg,r,str(root/'cases'/r['id']),k==0)) for k,r in enumerate(records)]
            for done,future in enumerate(as_completed(futures),1):
                result=future.result();print(json.dumps(dict(done=done,total=len(records),**result)),flush=True)
                write_json(root/'status.json',dict(status='running',pid=os.getpid(),done=done,total=len(records),elapsed_s=time.time()-started))
        write_json(root/'status.json',dict(status='complete',cases=len(records),frames=2*len(records),elapsed_s=time.time()-started))
    except BaseException as exc:
        write_json(root/'status.json',dict(status='failed',error=str(exc)));raise


if __name__=='__main__':main()
