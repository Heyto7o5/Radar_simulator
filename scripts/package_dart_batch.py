#!/usr/bin/env python3
"""Bundle report/images/configs/metrics and reproducible code, never raw data."""
import hashlib,json,zipfile
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
REPORT=PROJECT/'docs/validation_diagnostics/2026-10-15/dart/batch_v1'

def main():
    required=['DART面目标与背景批量报告.html','summary.json','per_frame_metrics.csv','report_selection.json']
    for name in required:
        if not (REPORT/name).is_file():raise FileNotFoundError('completed report missing: '+name)
    state=json.loads((REPORT/'state.json').read_text())
    if state['status']!='complete':raise ValueError('batch incomplete; do not package as finished')
    files=[(p,Path('report')/p.relative_to(REPORT)) for p in REPORT.rglob('*') if p.is_file() and p.suffix in {'.html','.json','.csv','.png'}]
    for folder in ['backend','scripts','tests']:
        files.extend((p,Path('source')/p.relative_to(PROJECT)) for p in (PROJECT/folder).rglob('*.py'))
    files.extend((PROJECT/name,Path('source')/name) for name in ['pyproject.toml','README.md'])
    path=PROJECT/'docs/deliveries/DART面目标批量验证_2026-10-15.zip';path.parent.mkdir(exist_ok=True)
    inventory=[]
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for source,target in sorted(files,key=lambda pair:str(pair[1])):
            content=source.read_bytes();z.writestr(str(target),content)
            inventory.append({'file':str(target),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
        z.writestr('inventory.json',json.dumps({'raw_data_included':False,'files':inventory},ensure_ascii=False,indent=2))
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:raise ValueError('package CRC check failed')
    print(json.dumps({'zip':str(path),'bytes':path.stat().st_size,'files':len(inventory),'crc_verified':True,'raw_data_included':False},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
