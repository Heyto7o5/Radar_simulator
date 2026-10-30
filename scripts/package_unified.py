#!/usr/bin/env python3
"""Audit and package the unified framework's derived demonstration/verification evidence."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import re
import shutil
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
from radar_platform.unified.presets import presets
from radar_platform.unified.storage import write_json


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,default=Path('.runtime/unified_campaign'))
    parser.add_argument('--output',type=Path,default=Path('docs/unified'))
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[1];root=args.campaign.resolve();out=args.output.resolve()
    images,metrics,configs,provenance=[out/name for name in ['images','metrics','configs','provenance']]
    for folder in [images,metrics,configs,provenance]:folder.mkdir(parents=True,exist_ok=True)
    checks={}
    for name in ['analytic_v2','real_transfer_v2']:
        summary=json.loads((root/name/'summary.json').read_text())
        expected=summary.get('cases',summary.get('frames'))
        if summary['passed']!=expected:raise ValueError(f'{name} contains failures')
        for path in (root/name).iterdir():
            if path.suffix in ('.json','.csv','.zip'):
                target=metrics/name/path.name;target.parent.mkdir(exist_ok=True)
                shutil.copy2(path,target)
        checks[name]=summary
    cases=list(csv.DictReader((root/'analytic_v2/cases.csv').open(encoding='utf-8-sig')))
    if len(cases)!=644 or any(c['passed']!='True' for c in cases):raise ValueError('analytic case count/status drift')
    tests=ET.parse(root/'tests_final.xml').getroot().findall('testsuite')
    counts={name:sum(int(t.attrib[name]) for t in tests) for name in ['tests','failures','errors','skipped']}
    if counts['tests']<146 or counts['failures'] or counts['errors']:raise ValueError('regression not complete')
    checks['tests']=counts
    shutil.copy2(root/'tests_final.xml',provenance/'tests.xml')

    for name in ['final_lfm','final_fmcw']:
        run=root/name;manifest=json.loads((run/'run.json').read_text())
        if manifest['status']!='complete' or len(manifest['frames'])!=4:raise ValueError('incomplete composite demonstration')
        # The current core must still match the source that produced the final images.
        for filename,digest in manifest['source_sha256'].items():
            if sha(repo/filename)!=digest:raise ValueError(f'source changed since final demo: {filename}')
        runcheck=[]
        for spec in manifest['frames']:
            frame=run/spec['directory'];meta=json.loads((frame/'metadata.json').read_text())
            for product,item in meta['products'].items():
                path=frame/item['file']
                if path.stat().st_size!=np.prod(item['shape'])*np.dtype(item['dtype']).itemsize:
                    raise ValueError('binary product size mismatch')
                values=np.memmap(path,dtype=item['dtype'],mode='r',shape=tuple(item['shape']))
                if not np.isfinite(values).all():raise ValueError('non-finite saved product')
            if meta['array']['virtual_aperture_rank']!=2:raise ValueError('composite example did not run a planar array')
            if len(meta['scene']['objects'])!=4:raise ValueError('composite example missing a scene component')
            runcheck.append({'frame_index':meta['frame_index'],'elapsed_s':meta['elapsed_s'],
                             'adc_shape':meta['products']['adc']['shape'],'rad_shape':meta['products']['rad']['shape'],
                             'warnings':meta['warnings']})
        target=metrics/name;target.mkdir(exist_ok=True)
        for filename in ['run.json','request.json','scene.json','terrain.json','source.zip']:
            shutil.copy2(run/filename,target/filename)
        checks[name]={'frames':runcheck,'outputs_verified_finite':True,'source_matches_current':True,
                      'synthetic_binary_results_remain_at':str(run)}
    sources={
        'final_lfm/images/scene.png':'scene.png',
        'final_lfm/images/rd_sequence.png':'lfm_rd.png',
        'final_fmcw/images/rd_sequence.png':'fmcw_rd.png',
        'final_lfm/images/array_and_angle.png':'array_and_angle.png',
        'verification_images/analytic_checks.png':'analytic_checks.png',
        'verification_images/mimo_resolution.png':'mimo_resolution.png',
    }
    for source,target in sources.items():shutil.copy2(root/source,images/target)
    resolution=json.loads((root/'verification_images/resolution_metrics.json').read_text())
    by_tx={r['tx']:r for r in resolution['separation_sweep'] if r['separation_deg']==24.}
    if by_tx[1]['resolved_by_declared_rule'] or not by_tx[2]['resolved_by_declared_rule']:
        raise ValueError('two-target MIMO demonstration did not pass its declared rule')
    write_json(metrics/'resolution.json',resolution)
    for name,value in presets().items():write_json(configs/(name+'.json'),value)
    shutil.copy2(repo/'docs/UNIFIED_PLATFORM.md',out/'使用说明.md')
    shutil.copy2(repo/'docs/UNIFIED_IMPLEMENTATION.md',out/'完成条件与证据.md')
    files=sorted([*repo.glob('backend/**/*.py'),*repo.glob('scripts/*.py'),*repo.glob('tests/**/*.py'),repo/'pyproject.toml'])
    with zipfile.ZipFile(provenance/'source_after.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for path in files:archive.write(path,path.relative_to(repo))
    write_json(provenance/'source_sha256.json',{str(p.relative_to(repo)):sha(p) for p in files})
    checks['utc']=datetime.now(timezone.utc).isoformat()
    checks['scope']='unified functional framework and ideal MIMO verified; not final real-world high fidelity, real-time speed or a new frontend'
    write_json(provenance/'completion_audit.json',checks)
    write_json(provenance/'environment.json',{'python':platform.python_version(),
        'packages':{name:version(name) for name in ['numpy','scipy','pydantic','fastapi','pytest','matplotlib']},
        'demonstration_execution':'CPU NumPy, OPENBLAS_NUM_THREADS=1; no GPU or web service launched'})
    for path in out.glob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if '://' not in link and not (path.parent/link).exists():raise ValueError(f'broken local link: {path.name}: {link}')
    manifest={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='package_manifest.json'}
    write_json(out/'package_manifest.json',manifest)
    destination=repo/'docs/统一框架与MIMO_文档图片配置指标.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file():archive.write(path,Path('统一框架与MIMO')/path.relative_to(out))
        if archive.testzip():raise ValueError('archive integrity failure')
    print(json.dumps({'archive':str(destination),'bytes':destination.stat().st_size,'files':len(manifest),
                      'tests':counts,'analytic_cases':644,'real_adc_frames':checks['real_transfer_v2']['frames'],
                      'demo_frames':8,'source_and_binary_audit':'passed'},ensure_ascii=False))


if __name__=='__main__':main()
