#!/usr/bin/env python3
"""Continue long-running generation -> training -> development flight locally.

Never restarts a live or failed stage. Validation/test boundaries are explicit.
The agent reviews validation or flight failures before changing the experiment.
"""
import argparse
import json
import os
import subprocess
import sys
import time

from radar_platform.navigation.io import ROOT,local_path,read_json,write_json,digest


def await_generation(path,progress):
    path=local_path(path)
    while True:
        state=read_json(path/'status.json')
        if state['status']=='complete':return
        if state['status']!='running':raise RuntimeError(f'generation terminated: {path}: {state}')
        pid=state.get('pid')
        if pid is None:raise RuntimeError(f'no live process identifier: {path}')
        try:os.kill(pid,0)
        except ProcessLookupError:raise RuntimeError(f'generation process {pid} no longer exists; inspect before resuming')
        write_json(progress,dict(status='waiting_for_generation',pid=os.getpid(),generation=str(path),generation_pid=pid,
                                done=state.get('done',0),total=state['total']))
        time.sleep(30)


def run(script,args,log):
    command=[sys.executable,str(local_path('scripts')/script),*args]
    with local_path(log).open('a') as f:
        f.write('\n'+json.dumps(command)+'\n');f.flush()
        result=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'{script} exited {result.returncode}; see {log}')


def merge_datasets(parents,output):
    output=local_path(output);output.mkdir(parents=True,exist_ok=True);(output/'cases').mkdir(exist_ok=True)
    records=[];base=None;provenance=None;parents_meta=[]
    for parent in parents:
        parent=local_path(parent)
        if read_json(parent/'status.json')['status']!='complete':raise ValueError('cannot merge incomplete generation')
        cfg=read_json(parent/'base_config.json')
        if base is not None and cfg!=base:raise ValueError('generation acquisitions differ')
        base=cfg;provenance=read_json(parent/'scene_provenance.json')
        protocol=read_json(parent/'protocol.json');parents_meta.append(dict(path=str(parent),protocol_sha256=digest(protocol)))
        for record in protocol['records']:
            destination=output/'cases'/record['id'];source=parent/'cases'/record['id']
            if destination.exists():
                if destination.resolve()!=source.resolve():raise ValueError('duplicate case id with different source')
            else:destination.symlink_to(source,target_is_directory=True)
            records.append(record)
    write_json(output/'protocol.json',dict(records=records,parents=parents_meta,
        counts={s:sum(r['split']==s for r in records) for s in ['train','val','test']},
        split='whole initial-state spatial cells and trajectories; supplemental turns use training cells only',
        label='target-relative body position and platform velocity at latest completed CPI',
        scope='same real DSM geography, independent phase/noise realizations'))
    write_json(output/'base_config.json',base);write_json(output/'scene_provenance.json',provenance)
    write_json(output/'status.json',dict(status='complete',cases=len(records),frames=2*len(records)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True);p.add_argument('--turns',required=True);p.add_argument('--combined',required=True)
    p.add_argument('--models',required=True);p.add_argument('--demo',required=True);p.add_argument('--log-root',required=True)
    p.add_argument('--threads',type=int,default=16)
    args=p.parse_args();logs=local_path(args.log_root);logs.mkdir(parents=True,exist_ok=True);progress=logs/'advance_status.json'
    try:
        await_generation(args.base,progress);await_generation(args.turns,progress)
        merge_datasets([args.base,args.turns],args.combined)
        write_json(progress,dict(status='training',pid=os.getpid(),models=args.models))
        models=local_path(args.models)
        if not models.exists():
            run('train_navigation.py',['--data',args.combined,'--output',args.models,'--threads',str(args.threads)],logs/'training_v1.log')
        elif not (models/'status.json').exists() or read_json(models/'status.json')['status']!='complete':
            raise RuntimeError('existing model stage not complete; inspect rather than launch a duplicate trainer')
        metrics=read_json(models/'metrics.json')['val']
        if metrics['position_mean_m']>2. or metrics['position_p90_m']>3. or metrics['velocity_mean_mps']>.6:
            write_json(progress,dict(status='validation_requires_analysis',pid=os.getpid(),validation=metrics,
                                    test_evaluated=False));return
        write_json(progress,dict(status='development_flight',pid=os.getpid(),validation=metrics))
        run('run_navigation.py',['--model-dir',args.models,'--output',args.demo],logs/'learned_demo_v1.log')
        summary=read_json(local_path(args.demo)/'complete.json')
        run('render_navigation.py',['--run',args.demo],logs/'render_demo_v1.log')
        write_json(progress,dict(status='development_flight_complete',validation=metrics,demo=summary,test_evaluated=False,
                                next='Review development flight, then evaluate frozen holdout, normal/stress campaigns and paired controls'))
    except BaseException as exc:
        write_json(progress,dict(status='needs_analysis',error=str(exc)));raise


if __name__=='__main__':main()
