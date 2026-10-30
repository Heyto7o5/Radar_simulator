#!/usr/bin/env python3
"""Long-running physical generation -> radar heatmap learning -> flight workflow."""
import argparse
import os
import time
import psutil

from advance_navigation import await_generation,merge_datasets,run
from radar_platform.navigation.io import local_path,read_json,write_json


def existing_training(root,progress):
    root=local_path(root)
    while True:
        state=read_json(root/'status.json')
        if state['status']=='complete':return
        if state['status'] not in ['training','calibrating']:raise RuntimeError(f'existing training is not active: {state}')
        pid=state.get('pid')
        if not pid or not psutil.pid_exists(pid):raise RuntimeError('existing trainer is absent; inspect before resuming')
        process=psutil.Process(pid)
        if process.status()==psutil.STATUS_ZOMBIE:raise RuntimeError('existing trainer is a terminated zombie')
        command=process.cmdline()
        if not any('train_navigation_v2.py' in part or 'calibrate_navigation_velocity.py' in part for part in command):raise RuntimeError('trainer PID belongs to another command')
        write_json(progress,dict(status='waiting_for_existing_training',pid=os.getpid(),training_pid=pid,models=str(root)))
        time.sleep(30)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',required=True);p.add_argument('--turns');p.add_argument('--combined')
    p.add_argument('--hover')
    p.add_argument('--features',required=True);p.add_argument('--models',required=True);p.add_argument('--demo')
    p.add_argument('--calibrated-models');p.add_argument('--filter-state',action='store_true')
    p.add_argument('--gentle-terminal',action='store_true')
    p.add_argument('--status',required=True);p.add_argument('--log-root',required=True)
    p.add_argument('--terrain-control',action='store_true');p.add_argument('--development',action='store_true')
    args=p.parse_args();progress=local_path(args.status);logs=local_path(args.log_root);logs.mkdir(parents=True,exist_ok=True)
    try:
        await_generation(args.base,progress)
        physical=args.base
        supplemental=[x for x in [args.turns,args.hover] if x]
        if supplemental:
            if not args.combined:raise ValueError('combined output required for supplemental datasets')
            for extra in supplemental:await_generation(extra,progress)
            merge_datasets([args.base,*supplemental],args.combined);physical=args.combined
        features=local_path(args.features)
        if not (features/'status.json').exists() or read_json(features/'status.json')['status']!='complete':
            if (features/'status.json').exists() and read_json(features/'status.json')['status']=='running':
                await_generation(features,progress)
            else:
                write_json(progress,dict(status='preprocessing_radar',pid=os.getpid(),features=args.features))
                run('prepare_navigation_features_v2.py',['--data',physical,'--output',args.features,'--workers','8'],logs/'features.log')
        models=local_path(args.models)
        if not models.exists():
            write_json(progress,dict(status='training',pid=os.getpid(),models=args.models))
            run('train_navigation_v2.py',['--data',args.features,'--output',args.models,'--threads','16'],logs/'training.log')
        else:existing_training(models,progress)
        if args.calibrated_models:
            calibrated=local_path(args.calibrated_models)
            if not calibrated.exists():
                write_json(progress,dict(status='calibrating_velocity',pid=os.getpid(),models=args.calibrated_models))
                run('calibrate_navigation_velocity.py',['--data',args.features,'--source-models',args.models,
                    '--output',args.calibrated_models,'--threads','16'],logs/'velocity_calibration.log')
            else:existing_training(calibrated,progress)
            models=calibrated
        metrics=read_json(models/'metrics.json')['val']
        if args.terrain_control:
            write_json(progress,dict(status='terrain_control_trained',validation=metrics,test_evaluated=False,
                next='Frozen holdout and paired normal closed loops, retaining failures'));return
        if not args.development and (metrics['position_mean_m']>2 or metrics['position_p90_m']>3 or metrics['velocity_mean_mps']>.6):
            write_json(progress,dict(status='validation_requires_analysis',validation=metrics,test_evaluated=False));return
        if not args.demo:raise ValueError('main workflow needs a development-flight output')
        write_json(progress,dict(status='development_flight',pid=os.getpid(),validation=metrics,development_only=args.development))
        flight_args=['--model-dir',str(models),'--output',args.demo]
        if args.filter_state:flight_args.append('--filter-state')
        if args.gentle_terminal:flight_args.append('--gentle-terminal')
        run('run_navigation.py',flight_args,logs/'flight.log')
        run('render_navigation.py',['--run',args.demo],logs/'render.log')
        write_json(progress,dict(status='development_flight_complete',validation=metrics,
            demo=read_json(local_path(args.demo)/'complete.json'),development_only=args.development,test_evaluated=False,
            next='Inspect actual flight; do not claim completion before independent evaluation and full campaigns'))
    except BaseException as exc:
        write_json(progress,dict(status='needs_analysis',error=str(exc)));raise


if __name__=='__main__':main()
