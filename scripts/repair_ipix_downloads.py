#!/usr/bin/env python3
"""Recover truncated new downloads; preserve rejected bytes and validate NetCDF."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,urllib.request
from pathlib import Path
from scipy.io import netcdf_file


def validate(path):
    with netcdf_file(path,mmap=False) as f:
        adc=f.variables['adc_data']
        return {'adc_shape':list(adc.shape),'dimensions':list(adc.dimensions),'prf_hz':float(f.variables['PRF'].data)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--root',type=Path,default=Path('/nas/datasets/hyt/datasets/radar_open/ipix_dartmouth'));a=p.parse_args()
    rows=json.loads(a.selection.read_text())['records'];raw=a.root/'raw'
    def recover(row):
        path=raw/row['filename'];url='https://soma.ece.mcmaster.ca/ipix/dartmouth/data/'+row['filename']
        entry=dict(row,url=url)
        if path.exists():
            try:
                shape=validate(path)
                return dict(entry,status='validated_existing',bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**shape)
            except Exception as exc:
                digest=hashlib.sha256(path.read_bytes()).hexdigest()
                quarantine=raw/'incomplete'/f'{path.name}.{digest[:12]}.part'
                quarantine.parent.mkdir(exist_ok=True)
                if not quarantine.exists():path.replace(quarantine)
                else:
                    # Preserve both paths rather than removing any raw data.
                    quarantine=quarantine.with_name(quarantine.name+'.duplicate')
                    if quarantine.exists():raise ValueError('duplicate quarantine target already exists')
                    path.replace(quarantine)
                entry.update(rejected_file=str(quarantine),rejected_reason=str(exc))
        for attempt in range(4):
            part=path.with_suffix('.cdf.part')
            try:
                with urllib.request.urlopen(url,timeout=40) as response,part.open('wb') as out:
                    expected=response.headers.get('Content-Length');count=0
                    while True:
                        block=response.read(1_048_576)
                        if not block:break
                        count+=len(block)
                        if count>20_000_000:raise ValueError('response exceeds per-record size guard')
                        out.write(block)
                if expected is not None and count!=int(expected):raise ValueError(f'truncated HTTP body {count} != {expected}')
                shape=validate(part);part.replace(path)
                print(row['record_id'],'recovered',count,flush=True)
                return dict(entry,status='recovered_and_validated',bytes=count,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),**shape)
            except Exception as exc:
                entry['last_error']=str(exc);print(row['record_id'],'attempt',attempt+1,str(exc),flush=True)
        return dict(entry,status='unavailable_after_retries')
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(recover,rows))
    (a.selection.parent/'recovery_manifest.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'files':results},indent=2)+'\n')
    print('validated',sum(x['status']!='unavailable_after_retries' for x in results),'/',len(results))


if __name__=='__main__':main()
