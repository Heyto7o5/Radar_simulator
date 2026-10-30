#!/usr/bin/env python3
"""Run the frozen parameter campaign through the public CLI; resumable and bounded."""
import argparse,concurrent.futures,hashlib,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=3)
    p.add_argument('--resume',action='store_true');a=p.parse_args()
    if not 1<=a.workers<=3:raise ValueError('one to three independent CPU workers allowed')
    a.output.mkdir(parents=True,exist_ok=a.resume)
    plan=json.loads(a.plan.read_text());files=[*sorted((ROOT/'backend/radar_platform').rglob('*.py')),ROOT/'scripts/run_scene.py']
    sources={str(f.relative_to(ROOT)):sha(f) for f in files}
    record={'plan_sha256':sha(a.plan),'source_sha256':sources,'plan':plan,'pid':os.getpid()}
    frozen=a.output/'campaign_inputs.json'
    if frozen.exists():
        old=json.loads(frozen.read_text())
        if old['plan_sha256']!=record['plan_sha256'] or old['source_sha256']!=sources:raise ValueError('resume requires unchanged plan and source')
    else:frozen.write_text(json.dumps(record,indent=2))
    (a.output/'campaign_process.json').write_text(json.dumps({'pid':os.getpid(),'argv':sys.argv,'workers':a.workers},indent=2))
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'}
    def run(case):
        config=a.plan.parent/case['config'];destination=a.output/case['case']
        if sha(config)!=case['config_sha256']:raise ValueError('prepared configuration changed')
        if any(sha(ROOT/name)!=digest for name,digest in sources.items()):raise ValueError('source changed during campaign')
        if destination.exists():
            manifest=destination/'run.json'
            if not a.resume or not manifest.exists():raise ValueError('existing partial output preserved; choose explicit recovery/version')
            status=json.loads(manifest.read_text())
            if status['status']!='complete':raise ValueError('existing failed/partial output preserved, not silently replaced')
            return {**case,'status':'complete','reused':True,'elapsed_s':status['total_elapsed_s']}
        started=time.perf_counter()
        with (a.output/(case['case']+'.log')).open('w') as log:
            result=subprocess.run([sys.executable,str(ROOT/'scripts/run_scene.py'),'--config',str(config.resolve()),'--output',str(destination.resolve())],
                                  cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        return {**case,'status':'complete' if result.returncode==0 else 'failed','exit_code':result.returncode,'elapsed_s':time.perf_counter()-started}
    rows=[];started=time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as executor:
        futures={executor.submit(run,c):c for c in plan['cases']}
        for future in concurrent.futures.as_completed(futures):
            case=futures[future]
            try:row=future.result()
            except Exception as exc:row={**case,'status':'failed','error':str(exc)}
            rows.append(row)
            status={'status':'running','completed_cases':len(rows),'expected_cases':len(plan['cases']),
                    'elapsed_s':time.perf_counter()-started,'results':rows}
            temporary=a.output/'campaign_status.tmp';temporary.write_text(json.dumps(status,indent=2));temporary.replace(a.output/'campaign_status.json')
            print(json.dumps({k:row[k] for k in ['case','status','elapsed_s'] if k in row}),flush=True)
    status['status']='complete' if all(r['status']=='complete' for r in rows) else 'failed'
    (a.output/'campaign_status.json').write_text(json.dumps(status,indent=2))
    if status['status']!='complete':raise SystemExit(1)


if __name__=='__main__':main()
