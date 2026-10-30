#!/usr/bin/env python3
"""Package code, weights, exact configurations, metrics and actual flight media."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from radar_platform.navigation.io import ROOT,local_path,read_json,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',required=True);p.add_argument('--demo',required=True)
    p.add_argument('--models',required=True);p.add_argument('--terrain-models',required=True)
    p.add_argument('--campaign',required=True);p.add_argument('--terrain-campaign',required=True)
    p.add_argument('--run-root',default='.runtime/navigation_20261030');p.add_argument('--output',required=True)
    args=p.parse_args();destination=local_path(args.output)
    if destination.exists():raise FileExistsError('existing reproducibility package preserved')
    report=local_path(args.report);run_root=local_path(args.run_root);demo=local_path(args.demo)
    summary=read_json(report/'summary.json');manifest=read_json(report/'artifact_manifest.json')
    if read_json(demo/'render_manifest.json').get('development_only'):raise ValueError('final package must not substitute a development-pool model')
    selected=set()
    def add(path):
        path=local_path(path)
        # This release covers the validated single-DEM experiment. Independent
        # experiments can be added to the shared workspace while long jobs run.
        relative=path.relative_to(ROOT)
        if 'navigation_multidem' in relative.parts or 'navigation_multidem' in path.stem:return
        if not path.is_file():raise FileNotFoundError(path)
        selected.add(path)
    def tree(path,patterns):
        path=local_path(path)
        for pattern in patterns:
            for file in path.rglob(pattern):
                if file.is_file():add(file)
    tree(ROOT/'backend/radar_platform',['*.py'])
    for file in (ROOT/'scripts').glob('*navigation*.py'):add(file)
    for name in ['run_scene.py','check_scene.py']:add(ROOT/'scripts'/name)
    tree(ROOT/'tests',['test_navigation*.py'])
    for name in ['pyproject.toml','docs/NAVIGATION_IMPLEMENTATION.md','scripts/navigation_env.sh']:
        add(ROOT/name)
    add(run_root/'requirements.lock.txt')
    add(ROOT/'.runtime/stage2_20261023/configs_v1/foothill_hamming_0_s23.json')
    tree(report,['*.json','*.csv','*.html','*.png','*.xml'])
    for models in [args.models,args.terrain_models]:tree(models,['*.json','*.pt','*.npz','*.zip'])
    for models in [args.models,args.terrain_models]:
        data=local_path(read_json(local_path(models)/'training_protocol.json')['data'])
        for filename in ['protocol.json','preprocessing.json','base_config.json','scene_provenance.json','status.json']:
            if (data/filename).is_file():add(data/filename)
        if (local_path(models)/'evaluation_protocol.json').is_file():
            evaluation_data=local_path(read_json(local_path(models)/'evaluation_protocol.json')['data_directory'])
            for filename in ['protocol.json','preprocessing.json','base_config.json','scene_provenance.json','status.json']:
                if (evaluation_data/filename).is_file():add(evaluation_data/filename)
    for campaign in [args.campaign,args.terrain_campaign]:
        path=local_path(campaign)
        if read_json(path/'status.json')['status']!='complete':raise ValueError('campaign incomplete')
        for name in ['protocol.json','status.json','summary.json','results.json']:add(path/name)
        for record in read_json(path/'protocol.json')['records']:
            case=path/record['id']
            for name in ['protocol.json','complete.json','truth_trajectory.json','controls.jsonl','acquisition.jsonl']:add(case/name)
            write_json(case/'replay_spec.json',read_json(case/'protocol.json')['spec']);add(case/'replay_spec.json')
    for name in ['protocol.json','scene.json','scene_provenance.json','complete.json','truth_trajectory.json',
                 'controls.jsonl','acquisition.jsonl','render_manifest.json','render_scope.json','media_audit.json',
                 'visual_inspection.json','report.html','flight.gif','flight.mp4']:
        add(demo/name)
    tree(demo/'rendered_frames',['*.png'])
    observations=sorted((demo/'observations').glob('*.npz'))
    if not observations:raise ValueError('representative flight has no radar observations')
    for i in sorted({0,len(observations)//2,len(observations)-1}):add(observations[i])
    first_adc=demo/'full_adc/frame_0000'
    tree(first_adc,['*.json','*.npz','*.bin'])
    for name in ['building_audit','oracle_audit_v1','angle_comparison_v1']:
        path=run_root/name
        for filename in ['audit.json','comparison.json','building_scatterers.npz']:
            if (path/filename).is_file():add(path/filename)
    source_paths=manifest.get('source_paths',{})
    for media_key in ['stress_demo','height_demo']:
        if not source_paths.get(media_key):continue
        stress_demo=local_path(source_paths[media_key])
        for name in ['scene.json','scene_provenance.json','render_manifest.json','render_scope.json',
                     'media_audit.json','visual_inspection.json','report.html','flight.gif','flight.mp4']:
            add(stress_demo/name)
        tree(stress_demo/'rendered_frames',['*.png'])
        stress_observations=sorted((stress_demo/'observations').glob('*.npz'))
        if not stress_observations:raise ValueError('supplementary movie has no numeric observations')
        for i in sorted({0,len(stress_observations)//2,len(stress_observations)-1}):add(stress_observations[i])
        tree(stress_demo/'full_adc/frame_0000',['*.json','*.npz','*.bin'])
    control_roots=[local_path(source_paths.get('no_control',run_root/'no_control_v1')),
                   local_path(source_paths.get('oracle_control',run_root/'oracle_campaign_v1'))]
    for control_root in control_roots:
        # Retain the actual control trajectories as well as aggregate scores.
        tree(control_root,['*.json','*.jsonl'])
    if source_paths.get('inference_audit'):tree(source_paths['inference_audit'],['*.json'])
    for name in ['audits_final','terrain_audits_final']:
        audit_root=run_root/name
        add(audit_root/'campaign/summary.json')
        add(audit_root/'inference/audit.json')
        tree(audit_root,['*.json'])
    add(run_root/'audits_final/demo_control/audit.json')
    add(run_root/'audits_final/demo_plant.json')
    add(run_root/'audits_final/demo_trajectory/audit.json')
    tree(run_root/'motion_bounds',['*.json'])
    for name in ['hover_split_audit.json','formal_train_val_split_audit.json',
                 'formal_train_val_audit/audit.json','terrain_train_val_audit/audit.json',
                 'paired_train_val_audit.json','main_raw_completion.json','terrain_raw_completion.json',
                 'full_split_metadata_audit.json',
                 'formal_validation_inference_audit/audit.json',
                 'campaign_concurrency_update_001/change.json',
                 'release_source_scope.json',
                 'noise_stress_failure_analysis.json',
                 'applicability_metadata_audit.json']:
        add(run_root/name)
    for filename in ['geometry_audit.json','compiled_bvh_audit.json','motion_check.json']:
        add(run_root/'preparation'/filename)
    file_manifest={}
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_suffix(destination.suffix+'.tmp')
    with zipfile.ZipFile(temporary,'x',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for file in sorted(selected):
            name=str(file.relative_to(ROOT))
            with file.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
            file_manifest[name]=dict(bytes=file.stat().st_size,sha256=sha)
            archive.write(file,name)
        payload=dict(files=file_manifest,acceptance=summary['acceptance'],
            full_training_observations_included=False,
            explanation='Full raw datasets remain in the original project run directories. Package contains regenerating code/configuration, frozen models, all flight metrics/traces, movie/frames and selected full numeric observations/ADC.',
            report=str(report.relative_to(ROOT)/'report.html'),representative=str(demo.relative_to(ROOT)))
        archive.writestr('navigation_reproducibility_manifest.json',json.dumps(payload,ensure_ascii=False,indent=2))
        replay=read_json(demo/'protocol.json')
        command=f'.venv_navigation/bin/python scripts/run_navigation.py --model-dir {local_path(args.models).relative_to(ROOT)} --output .runtime/navigation_replay'
        if replay.get('state_filter'):command+=' --filter-state'
        if replay.get('gentle_terminal'):command+=' --gentle-terminal'
        archive.writestr('REPRODUCE_NAVIGATION.txt',
            'Run from the radar platform project root with the locked dependencies.\n'
            'The supplied environment is .venv_navigation; .venv may point outside the project.\n'
            'If relocating this package, recreate dependencies from pyproject.toml and the lock; replace the editable original-project path with the new root and use the official CPU PyTorch index for +cpu wheels.\n'
            'source scripts/navigation_env.sh\n'+command+'\n'
            'Navigation tests: .venv_navigation/bin/python -m pytest tests/test_navigation*.py --basetemp=.runtime/navigation_test_tmp\n'
            'For a campaign case, add --spec <case/replay_spec.json>. Keep the recorded --filter-state setting.\n'
            'Simulation times are virtual; actual radar synthesis is offline. Never use the truth-control reference as learned guidance evidence.\n')
    temporary.replace(destination)
    with destination.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    write_json(destination.with_suffix('.manifest.json'),dict(path=str(destination),sha256=sha,bytes=destination.stat().st_size,
        files=len(selected),acceptance=summary['acceptance']))
    print(json.dumps(dict(package=str(destination),bytes=destination.stat().st_size,files=len(selected),sha256=sha)))


if __name__=='__main__':main()
