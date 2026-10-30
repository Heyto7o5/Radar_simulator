#!/usr/bin/env python3
"""Wait for both audited conditions, then build the final report and reproduction bundle.

This produces reviewable evidence only. It does not mark the chat goal complete;
the agent must inspect the resulting artifacts against the approved requirements.
"""
import argparse
import json
import os
import time
from pathlib import Path

from advance_navigation import run
from navigation_job import inspect_job
from radar_platform.navigation.io import ROOT,local_path,read_json,write_json
from radar_platform.navigation.episode import model_fingerprint


def wait_conditions(root,progress):
    pairs=[('finish_main_job.json','finish_main_status.json'),('finish_terrain_job.json','finish_terrain_status.json')]
    while True:
        states={};ready=True
        for record_name,status_name in pairs:
            state=read_json(root/status_name);states[status_name]=state['status']
            if state['status']=='complete':continue
            ready=False
            if state['status']=='needs_analysis' or state['status'].endswith('requires_analysis'):
                raise RuntimeError(f'{status_name}: {state}')
            job=inspect_job(root/record_name)
            if not job['alive']:raise RuntimeError(f'{record_name} stopped without completing its condition')
        if ready:return
        write_json(progress,dict(status='waiting_for_audited_conditions',pid=os.getpid(),conditions=states))
        time.sleep(30)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.runtime/navigation_20261030')
    args=p.parse_args();root=local_path(args.root);progress=root/'finalization_status.json';logs=root/'finalization_logs';logs.mkdir(exist_ok=True)
    try:
        wait_conditions(root,progress)
        models=root/'models_final';terrain=root/'terrain_models_final';demo=root/'learned_demo_final'
        for directory in [models,terrain]:
            if read_json(directory/'training_protocol.json').get('development_only',False):raise ValueError('development model cannot substitute for final training')
            if read_json(directory/'test_metrics.json')['evaluation_counts']['total_states']!=1024:raise ValueError('final test is incomplete')
        report=root/'final_report'
        if (report/'artifact_manifest.json').exists():
            old=read_json(report/'artifact_manifest.json')
            if old['main_model_sha256']!=model_fingerprint(models) or old['terrain_model_sha256']!=model_fingerprint(terrain):
                raise ValueError('existing report uses different frozen models; preserve it and choose a versioned output')
        write_json(progress,dict(status='building_report',pid=os.getpid()))
        run('report_navigation_experiment.py',[
            '--campaign',str(root/'campaign_final'),'--terrain-campaign',str(root/'terrain_campaign_final'),
            '--models',str(models),'--terrain-models',str(terrain),'--demo',str(demo),
            '--no-control',str(root/'no_control_v1'),'--oracle-control',str(root/'oracle_campaign_gentle'),
            '--angle-comparison',str(root/'angle_comparison_v1'),'--inference-audit',str(root/'audits_final/inference'),
            '--output',str(report)],logs/'report.log')
        import subprocess,sys
        tests=sorted(str(path.relative_to(ROOT)) for path in (ROOT/'tests').glob('test_navigation*.py')
                     if 'navigation_multidem' not in path.stem)
        tests+=['tests/test_dem_buildings.py','tests/test_unified.py','tests/test_simulation_session.py','tests/test_unified_api.py',
                'tests/test_platform_architecture.py','tests/test_real_dem_geometry.py','tests/test_visibility.py','tests/test_sampled_trajectory.py']
        write_json(progress,dict(status='final_regression',pid=os.getpid()))
        with (logs/'regression.log').open('w') as stream:
            completed=subprocess.run([sys.executable,'-m','pytest',*tests,
                '--basetemp='+str(root/'pytest_release'), '--junitxml='+str(report/'regression.xml')],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if completed.returncode:raise RuntimeError('final regression failed; inspect log')
        write_json(progress,dict(status='packaging_reproduction',pid=os.getpid()))
        package=root/'navigation_results.zip'
        if not package.exists():
            run('package_navigation_result.py',[
                '--report',str(report),'--demo',str(demo),'--models',str(models),'--terrain-models',str(terrain),
                '--campaign',str(root/'campaign_final'),'--terrain-campaign',str(root/'terrain_campaign_final'),
                '--run-root',str(root),'--output',str(package)],logs/'package.log')
        write_json(progress,dict(status='ready_for_agent_completion_audit',report=str(report/'report.html'),package=str(package),
            acceptance=read_json(report/'summary.json')['acceptance'],
            scope='Full artifacts prepared. Inspect actual evidence and media before declaring the user objective complete.'))
    except BaseException as exc:
        write_json(progress,dict(status='needs_analysis',error=str(exc)));raise


if __name__=='__main__':main()
