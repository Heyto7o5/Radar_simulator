#!/usr/bin/env python3
"""Train once all declared train/validation cases exist while test simulation continues.

The training view contains the complete planned train/validation subsets and no
test records. A full feature view is joined later without changing training
files or splits. No partial validation set or growing-score checkpointing.
"""
import argparse
import hashlib
import json
import os
import time
import psutil

from advance_navigation import await_generation,merge_datasets,run
from advance_navigation_v2 import existing_training
from radar_platform.navigation.io import local_path,read_json,write_json,digest


def parent_records(parents):
    records=[];configs=[];manifest=[]
    for parent in parents:
        parent=local_path(parent);p=read_json(parent/'protocol.json')
        configs.append(read_json(parent/'base_config.json'))
        records.extend((r,parent) for r in p['records'])
        manifest.append(dict(path=str(parent),protocol_sha256=digest(p)))
    if not all(c==configs[0] for c in configs):raise ValueError('parent acquisition configurations differ')
    if len({r['id'] for r,_ in records})!=len(records):raise ValueError('duplicate physical case id')
    return records,configs[0],manifest


def wait_cases(parents,groups,progress):
    records,_,_=parent_records(parents)
    selected=[(r,p) for r,p in records if r['split'] in groups]
    while True:
        missing=[(r,p) for r,p in selected if not (p/'cases'/r['id']/'complete.json').is_file()]
        if not missing:return
        for parent in {p for _,p in missing}:
            state=read_json(parent/'status.json');pid=state.get('pid')
            if state['status']=='complete':raise RuntimeError(f'producer reports complete but declared cases are missing: {parent}')
            if state['status']!='running' or not pid or not psutil.pid_exists(pid):
                raise RuntimeError(f'physical producer is not live: {parent}')
            process=psutil.Process(pid)
            if process.status()==psutil.STATUS_ZOMBIE:raise RuntimeError(f'physical producer terminated: {parent}')
            command=process.cmdline()
            if '--output' not in command or local_path(command[command.index('--output')+1])!=parent:
                raise RuntimeError('producer PID no longer identifies the declared output')
        write_json(progress,dict(status='waiting_for_declared_splits',pid=os.getpid(),splits=list(groups),
            ready=len(selected)-len(missing),total=len(selected),producer_pids={str(p):read_json(p/'status.json').get('pid') for p in {q for _,q in missing}}))
        time.sleep(30)


def training_view(parents,output):
    records,config,manifest=parent_records(parents);root=local_path(output)
    root.mkdir(parents=True,exist_ok=True);(root/'cases').mkdir(exist_ok=True)
    chosen=[(r,p) for r,p in records if r['split'] in ['train','val']]
    for record,parent in chosen:
        source=parent/'cases'/record['id'];target=root/'cases'/record['id']
        if not (source/'complete.json').is_file():raise ValueError('training view contains an incomplete case')
        if target.is_symlink() or target.exists():
            if target.resolve()!=source.resolve():raise ValueError('training view points to a different source')
        else:target.symlink_to(source,target_is_directory=True)
    protocol=dict(records=[r for r,_ in chosen],parents=manifest,
        counts={s:sum(r['split']==s for r,_ in chosen) for s in ['train','val','test']},
        planned_full_counts={s:sum(r['split']==s for r,_ in records) for s in ['train','val','test']},
        scope='Complete predeclared training and validation subsets; final test records intentionally excluded from training view',
        split='Unchanged complete initial-state spatial cells and independent physical scenes; terminal additions remain in training cells')
    if (root/'protocol.json').exists() and read_json(root/'protocol.json')!=protocol:raise ValueError('training view protocol changed')
    write_json(root/'protocol.json',protocol);write_json(root/'base_config.json',config)
    write_json(root/'scene_provenance.json',read_json(local_path(parents[0])/'scene_provenance.json'))
    write_json(root/'status.json',dict(status='complete',cases=len(chosen),scope='train/validation view only; not complete experimental corpus'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parents',nargs='+',required=True);p.add_argument('--training-view',required=True)
    p.add_argument('--training-features',required=True);p.add_argument('--full-data',required=True);p.add_argument('--full-features',required=True)
    p.add_argument('--spatial-models',required=True);p.add_argument('--models',required=True);p.add_argument('--demo')
    p.add_argument('--status',required=True);p.add_argument('--log-root',required=True);p.add_argument('--terrain-control',action='store_true')
    args=p.parse_args();progress=local_path(args.status);logs=local_path(args.log_root);logs.mkdir(parents=True,exist_ok=True)
    try:
        wait_cases(args.parents,['train','val'],progress)
        training_view(args.parents,args.training_view)
        features=local_path(args.training_features)
        if not (features/'status.json').exists():
            write_json(progress,dict(status='preprocessing_complete_train_val',pid=os.getpid()))
            run('prepare_navigation_features_v2.py',['--data',args.training_view,'--output',args.training_features,'--workers','8'],logs/'features_train_val.log')
        elif read_json(features/'status.json')['status']!='complete':
            raise RuntimeError('existing feature stage is incomplete; inspect rather than duplicate')
        spatial=local_path(args.spatial_models)
        if not spatial.exists():
            write_json(progress,dict(status='training_spatial_models',pid=os.getpid(),test_used=False))
            run('train_navigation_v2.py',['--data',args.training_features,'--output',args.spatial_models,'--threads','16'],logs/'training.log')
        else:existing_training(spatial,progress)
        models=local_path(args.models)
        if not models.exists():
            write_json(progress,dict(status='calibrating_zero_anchored_velocity',pid=os.getpid(),test_used=False))
            run('calibrate_navigation_velocity.py',['--data',args.training_features,'--source-models',args.spatial_models,
                '--output',args.models,'--threads','16'],logs/'velocity_calibration.log')
        else:existing_training(models,progress)
        # Physics continues independently while learning runs. No test result
        # affects weights; complete the full view for later frozen evaluation.
        wait_cases(args.parents,['train','val','test'],progress)
        for parent in args.parents:
            await_generation(parent,progress)
        merge_datasets(args.parents,args.full_data)
        full=local_path(args.full_features);full.mkdir(parents=True,exist_ok=True);(full/'cases').mkdir(exist_ok=True)
        for name,expected in read_json(features/'preprocessing.json')['source_sha256'].items():
            if hashlib.sha256(local_path(name).read_bytes()).hexdigest()!=expected:
                raise RuntimeError('preprocessing source changed after train/validation features froze; do not mix feature versions')
        for record in read_json(features/'protocol.json')['records']:
            target=full/'cases'/record['id'];source=features/'cases'/record['id']
            if target.is_symlink() or target.exists():
                if target.resolve()!=source.resolve():raise ValueError('full features disagree with frozen training features')
            else:target.symlink_to(source,target_is_directory=True)
        if not (full/'status.json').exists():
            run('prepare_navigation_features_v2.py',['--data',args.full_data,'--output',args.full_features,'--workers','8'],logs/'features_test.log')
        elif read_json(full/'status.json')['status']!='complete':raise RuntimeError('existing full feature preparation incomplete')
        metrics=read_json(models/'metrics.json')['val']
        if args.terrain_control:
            write_json(progress,dict(status='terrain_control_ready',validation=metrics,full_features=args.full_features,test_evaluated=False));return
        if metrics['position_mean_m']>2 or metrics['position_p90_m']>3 or metrics['velocity_mean_mps']>.6:
            write_json(progress,dict(status='validation_requires_analysis',validation=metrics,test_evaluated=False));return
        if not args.demo:raise ValueError('main workflow needs a demonstration output')
        write_json(progress,dict(status='development_flight',pid=os.getpid(),validation=metrics))
        run('run_navigation.py',['--model-dir',args.models,'--output',args.demo,'--filter-state','--gentle-terminal'],logs/'flight.log')
        run('render_navigation.py',['--run',args.demo],logs/'render.log')
        write_json(progress,dict(status='development_flight_complete',validation=metrics,
            demo=read_json(local_path(args.demo)/'complete.json'),full_features=args.full_features,test_evaluated=False,
            next='Frozen independent evaluation and all predeclared learned/control campaigns remain'))
    except BaseException as exc:
        write_json(progress,dict(status='needs_analysis',error=str(exc)));raise


if __name__=='__main__':main()
