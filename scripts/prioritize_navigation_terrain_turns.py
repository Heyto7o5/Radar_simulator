#!/usr/bin/env python3
"""Use released simulation capacity for already declared terrain training cases.

Completed, configuration-identical results are published atomically as links
into the original producer's cache. Never link partial data, overwrite a case
owned by the original producer, modify its protocol or count extra realizations.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib
import json
import os
import time

from navigation_job import inspect_job
from radar_platform.navigation.io import local_path,read_json,write_json,digest
from radar_platform.navigation.dataset import generate_case


def one(job):
    base,record,extra,original=job;extra=local_path(extra);original=local_path(original)
    if (original/'complete.json').is_file():return dict(id=record['id'],result='already_complete')
    if original.exists() or original.is_symlink():return dict(id=record['id'],result='owned_by_original_producer')
    result=generate_case((base,record,str(extra),False))
    if result['configuration_sha256']!=digest(dict(base=base,record=record)):
        raise ValueError('completed result differs from original case configuration')
    assert (extra/'complete.json').is_file()
    try:
        original.symlink_to(extra,target_is_directory=True)
        disposition='published_complete_cache'
    except FileExistsError:
        disposition='original_producer_started_first_original_preserved'
    return dict(id=record['id'],result=disposition,configuration_sha256=result['configuration_sha256'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wait-record',required=True);p.add_argument('--wait-status',required=True)
    p.add_argument('--terrain-data',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=16)
    args=p.parse_args();root=local_path(args.output);root.mkdir(parents=True,exist_ok=True)
    while read_json(args.wait_status)['status']!='complete':
        job=inspect_job(args.wait_record)
        if not job['alive']:raise RuntimeError('capacity-release prerequisite stopped without completing')
        write_json(root/'status.json',dict(status='waiting_for_released_capacity',pid=os.getpid(),prerequisite_pid=job['pid']))
        time.sleep(30)
    parent=local_path(args.terrain_data);base=read_json(parent/'base_config.json');protocol=read_json(parent/'protocol.json')
    records=[r for r in protocol['records'] if r['split']=='train' and r['id'].startswith('turn_')]
    old=read_json(parent/'source_sha256.json')
    physical_navigation={'__init__.py','io.py','scene.py','dynamics.py','acquisition.py','observation.py','dataset.py'}
    dependencies={}
    for name,sha in old.items():
        if not name.startswith('backend/'):continue
        if '/navigation/' in name and name.rsplit('/',1)[-1] not in physical_navigation:continue
        now=hashlib.sha256(local_path(name).read_bytes()).hexdigest()
        if now!=sha:raise RuntimeError(f'frozen generation dependency changed: {name}')
        dependencies[name]=now
    write_json(root/'protocol.json',dict(records=records,original_protocol_sha256=digest(protocol),base_config_sha256=digest(base),
        physical_source_sha256=dependencies,source=str(parent),
        scope='Reordered computation of existing training cases only; no added samples or test-set change; original producer remains live'))
    write_json(root/'status.json',dict(status='running',pid=os.getpid(),total=len(records)))
    results=[]
    try:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(one,(base,r,str(root/'cases'/r['id']),str(parent/'cases'/r['id']))) for r in records]
            for future in as_completed(futures):
                result=future.result();results.append(result);write_json(root/'results.json',results)
                write_json(root/'status.json',dict(status='running',pid=os.getpid(),done=len(results),total=len(records)))
                print(json.dumps(result),flush=True)
        write_json(root/'status.json',dict(status='complete',cases_considered=len(records),
            published=sum(r['result']=='published_complete_cache' for r in results),new_dataset_samples=0))
    except BaseException as exc:
        write_json(root/'status.json',dict(status='needs_analysis',error=str(exc)));raise


if __name__=='__main__':main()
