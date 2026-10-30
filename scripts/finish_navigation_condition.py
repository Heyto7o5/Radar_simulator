#!/usr/bin/env python3
"""Run the frozen learned campaigns and final holdout after the learning stage."""
import argparse
import json
import os
import time
import psutil

from advance_navigation import run
from navigation_job import inspect_job
from radar_platform.navigation.io import local_path,read_json,write_json
from radar_platform.navigation.episode import model_fingerprint


def wait_for_stage(record,status,progress,terrain):
    wanted='terrain_control_ready' if terrain else 'development_flight_complete'
    while True:
        value=read_json(status)
        if value['status']==wanted:return value
        if value['status'] in ['needs_analysis','validation_requires_analysis']:
            raise RuntimeError(f'learning stage requires analysis: {value}')
        process=inspect_job(record)
        if not process['alive']:raise RuntimeError('recorded learning process has stopped before its terminal result')
        write_json(progress,dict(status='waiting_for_learning_stage',pid=os.getpid(),stage_status=value['status'],stage_pid=process['pid']))
        time.sleep(30)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage-record',required=True);p.add_argument('--stage-status',required=True)
    p.add_argument('--models',required=True);p.add_argument('--features',required=True);p.add_argument('--campaign',required=True)
    p.add_argument('--audits',required=True);p.add_argument('--status',required=True);p.add_argument('--log-root',required=True)
    p.add_argument('--demo');p.add_argument('--terrain-control',action='store_true');p.add_argument('--workers',type=int,default=16)
    args=p.parse_args();progress=local_path(args.status);logs=local_path(args.log_root);logs.mkdir(parents=True,exist_ok=True)
    audit_root=local_path(args.audits);audit_root.mkdir(parents=True,exist_ok=True)
    try:
        stage=wait_for_stage(args.stage_record,args.stage_status,progress,args.terrain_control)
        if not args.terrain_control:
            if not args.demo:raise ValueError('representative learned flight required')
            if not read_json(local_path(args.demo)/'complete.json')['success']:
                raise RuntimeError('final-model development flight did not meet actual arrival criteria; holdout remains unopened')
        campaign=local_path(args.campaign)
        if (campaign/'status.json').exists():
            while read_json(campaign/'status.json')['status']=='running':
                state=read_json(campaign/'status.json');pid=state.get('pid')
                if not pid or not psutil.pid_exists(pid):raise RuntimeError('existing campaign process is absent; inspect partial cases before resuming')
                process=psutil.Process(pid)
                if process.status()==psutil.STATUS_ZOMBIE or abs(process.create_time()-state['process_create_time'])>.01:
                    raise RuntimeError('recorded campaign process is no longer active')
                command=process.cmdline()
                if '--output' not in command or local_path(command[command.index('--output')+1])!=campaign:
                    raise RuntimeError('campaign PID belongs to another command')
                write_json(progress,dict(status='waiting_for_existing_campaign',pid=os.getpid(),campaign_pid=pid,done=state['done'],total=state['total']))
                time.sleep(30)
            if read_json(campaign/'status.json')['status']=='failed':raise RuntimeError('existing campaign failed; inspect before launching another one')
        if not (campaign/'status.json').exists() or read_json(campaign/'status.json')['status']!='complete':
            write_json(progress,dict(status='closed_loop_campaign',pid=os.getpid(),test_evaluated=False))
            command=['--output',args.campaign,'--model-dir',args.models,'--workers',str(args.workers),'--filter-state','--gentle-terminal']
            if args.terrain_control:command+=['--normal-only','--terrain-only']
            run('navigation_campaign.py',command,logs/'campaign.log')
        campaign_audit=audit_root/'campaign'
        if not (campaign_audit/'summary.json').exists():
            write_json(progress,dict(status='auditing_each_episode',pid=os.getpid()))
            run('audit_navigation_campaign.py',['--campaign',args.campaign,'--models',args.models,'--output',str(campaign_audit),'--workers','4'],logs/'campaign_audit.log')
        if not read_json(campaign_audit/'summary.json')['all_case_audits_passed']:raise RuntimeError('one or more episode audits failed')
        summary=read_json(campaign/'summary.json')
        if not args.terrain_control and summary['normal']['success_rate']<.9:
            write_json(progress,dict(status='normal_acceptance_requires_analysis',summary=summary,test_evaluated=False));return
        # The separate final test is opened only after engineering acceptance.
        models=local_path(args.models)
        if not (models/'test_metrics.json').exists():
            write_json(progress,dict(status='frozen_holdout_evaluation',pid=os.getpid()))
            run('evaluate_navigation.py',['--data',args.features,'--model-dir',args.models,'--threads','8'],logs/'holdout.log')
        evaluation=read_json(models/'test_metrics.json')
        if evaluation['evaluation_counts']['total_states']!=1024:raise RuntimeError('incomplete final test')
        if read_json(models/'evaluation_protocol.json')['model_sha256']!=model_fingerprint(models):
            raise RuntimeError('cached test results belong to different model/calibration files')
        inference_audit=audit_root/'inference'
        if not (inference_audit/'audit.json').exists():
            run('audit_navigation_inference.py',['--data',args.features,'--models',args.models,'--output',str(inference_audit)],logs/'inference_audit.log')
        if read_json(inference_audit/'audit.json')['model_sha256']!=model_fingerprint(models):raise RuntimeError('inference audit belongs to different model')
        if not args.terrain_control:
            for script,arguments,name in [
                ('audit_navigation.py',['--episode',args.demo,'--output',str(audit_root/'demo_trajectory')],'trajectory'),
                ('audit_navigation_plant.py',['--run',args.demo,'--output',str(audit_root/'demo_plant.json')],'plant')]:
                run(script,arguments,logs/f'demo_{name}_audit.log')
            control_audit=audit_root/'demo_control'
            if not (control_audit/'audit.json').exists():
                run('audit_navigation_control_replay.py',['--run',args.demo,'--output',str(control_audit)],logs/'demo_control_audit.log')
        write_json(progress,dict(status='complete',campaign=summary,final_test_states=1024,models=args.models,audits=args.audits,
            scope='One condition finished; combined report, reproduction bundle and full goal completion audit still required'))
    except BaseException as exc:
        write_json(progress,dict(status='needs_analysis',error=str(exc)));raise


if __name__=='__main__':main()
