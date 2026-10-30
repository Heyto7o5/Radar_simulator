#!/usr/bin/env python3
"""Package additional background reports; preserve the original frozen batch ZIP."""
import json,zipfile
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
ROOT=PROJECT/'docs/validation_diagnostics/2026-10-15/dart'

def main():
    path=PROJECT/'docs/deliveries/DART背景诊断与有色高斯对照_2026-10-15.zip'
    required=[ROOT/'background_audit/summary.json',ROOT/'colored_background_followup/summary.json']
    for p in required:
        if not p.is_file():raise FileNotFoundError(p)
    names=[]
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in ['background_audit','colored_background_followup']:
            for p in sorted((ROOT/folder).rglob('*')):
                if p.is_file() and p.suffix in {'.html','.png','.json','.csv'}:
                    name=str(p.relative_to(ROOT));archive.write(p,name);names.append(name)
        archive.writestr('package_info.json',json.dumps({'raw_data_included':False,'original_batch_archive_not_modified':True,
            'scope':'additional reports, figures, empirical parameters and metrics; code remains in project workspace',
            'independent_scenes':1,'independent_holdout_scenes':0,'files':names},indent=2))
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:raise ValueError('ZIP CRC mismatch')
    print(json.dumps({'zip':str(path),'bytes':path.stat().st_size,'crc_verified':True},ensure_ascii=False))

if __name__=='__main__':main()
