#!/usr/bin/env python3
"""Verify every completed learned episode, including all failures, from saved evidence."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import json
import subprocess
import sys

from radar_platform.navigation.io import ROOT,local_path,read_json,write_json
from radar_platform.navigation.episode import model_fingerprint,execution_fingerprint


def inspect_case(job):
    root,output,record,model,execution,filter_enabled,gentle=job
    case=local_path(root)/record['id'];out=local_path(output)/record['id'];out.mkdir(parents=True,exist_ok=True)
    try:
        protocol=read_json(case/'protocol.json');summary=read_json(case/'complete.json')
        if protocol['mode']!='learned' or protocol['spec']!=record:raise ValueError('mode/initial-condition protocol differs')
        if protocol.get('model_sha256')!=model:raise ValueError('different model/calibration used')
        if protocol.get('execution_sha256')!=execution:raise ValueError('different execution source used; inspect compatibility before accepting')
        if bool(protocol.get('state_filter'))!=filter_enabled or protocol.get('gentle_terminal',False)!=gentle:
            raise ValueError('different estimation/guidance policy used')
        if not summary.get('complete_experiment'):raise ValueError('partial experiment cannot count as full acceptance')
        commands=[json.loads(s) for s in (case/'controls.jsonl').read_text().splitlines() if s]
        if summary['termination']=='estimated_arrival':
            if not commands or not commands[-1]['control']['estimated_arrival']:raise ValueError('arrival lacks an online decision')
        for script,arguments in [
            ('audit_navigation.py',['--episode',str(case),'--output',str(out/'trajectory')]),
            ('audit_navigation_plant.py',['--run',str(case),'--output',str(out/'plant.json')])]:
            result=subprocess.run([sys.executable,str(ROOT/'scripts'/script),*arguments],cwd=ROOT,capture_output=True,text=True)
            if result.returncode:raise RuntimeError(result.stderr or result.stdout)
        truth=read_json(out/'trajectory/audit.json')['episode']
        if bool(summary['success'])!=bool(truth['success_recomputed_from_last_3_seconds']):raise ValueError('reported success differs from independent truth check')
        return dict(id=record['id'],audit_passed=True,mission_success=summary['success'],termination=summary['termination'])
    except Exception as exc:
        error=dict(id=record['id'],audit_passed=False,error=str(exc));write_json(out/'failure.json',error);return error


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True);p.add_argument('--models',required=True);p.add_argument('--output',required=True)
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args();root=local_path(args.campaign);output=local_path(args.output);output.mkdir(parents=True,exist_ok=True)
    if read_json(root/'status.json')['status']!='complete':raise ValueError('campaign must be finished first')
    protocol=read_json(root/'protocol.json');records=protocol['records'];results=read_json(root/'results.json')
    expected={r['id'] for r in records};observed={r['episode'] for r in results}
    if len(results)!=len(expected) or observed!=expected:raise ValueError('campaign omits or duplicates declared episodes')
    models=model_fingerprint(args.models);execution=execution_fingerprint();audits=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs=[(str(root),str(output),r,models,execution,protocol.get('state_filter',False),protocol.get('gentle_terminal',False)) for r in records]
        for future in as_completed([pool.submit(inspect_case,j) for j in jobs]):
            audits.append(future.result());write_json(output/'cases.json',audits)
    passed=all(r['audit_passed'] for r in audits)
    write_json(output/'summary.json',dict(status='complete' if passed else 'failed',declared_cases=len(records),audited_cases=len(audits),
        all_case_audits_passed=passed,mission_successes=sum(r.get('mission_success',False) for r in audits),
        model_sha256=models,execution_sha256=execution,
        scope='Every declared case retained; independent motion, acquisition-pose linkage, true arrival, static scene/phase seed, noise indexing and enabled Tx/Rx all-surface visibility checked'))
    print(json.dumps(dict(cases=len(audits),all_audits_passed=passed)))
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
