#!/usr/bin/env python3
"""Train the paired terrain-only control after its physical observations finish."""
import argparse
import os
from advance_navigation import await_generation,run
from radar_platform.navigation.io import local_path,read_json,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--models',required=True);p.add_argument('--log-root',required=True)
    args=p.parse_args();logs=local_path(args.log_root);progress=logs/'advance_terrain_status.json'
    try:
        await_generation(args.data,progress)
        write_json(progress,dict(status='training',pid=os.getpid(),models=args.models))
        models=local_path(args.models)
        if not models.exists():run('train_navigation.py',['--data',args.data,'--output',args.models,'--threads','16'],logs/'training_terrain_v1.log')
        elif not (models/'status.json').exists() or read_json(models/'status.json')['status']!='complete':
            raise RuntimeError('existing terrain trainer incomplete; inspect before resuming')
        write_json(progress,dict(status='terrain_control_trained',validation=read_json(models/'metrics.json')['val'],
            next='Evaluate frozen holdout and paired normal closed loops; keep all failures',test_evaluated=False))
    except BaseException as exc:
        write_json(progress,dict(status='needs_analysis',error=str(exc)));raise


if __name__=='__main__':main()
