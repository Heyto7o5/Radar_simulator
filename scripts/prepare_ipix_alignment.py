#!/usr/bin/env python3
"""Acquire every officially listed target/clutter record for prespecified splits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib,json,re,urllib.request
from pathlib import Path
from repair_ipix_downloads import validate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/nas/datasets/hyt/datasets/radar_open/ipix_dartmouth'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    html=(a.root/'raw/datasets.html').read_text()
    pattern=r'href="data/([^"]+\.cdf)">#(\d+)</a>\s*<td[^>]*>(\d+)<td[^>]*>(\d+):(\d+)'
    matches=re.findall(pattern,html)
    if len(matches)!=14:raise ValueError('official target table changed; inspect before selecting')
    plan=[]
    for filename,number,primary,first,last in matches:
        n=int(number);plan.append({'record_id':n,'filename':filename,'excluded_target_gates':list(range(int(first),int(last)+1)),
            'role':'development_record' if n in [17,18,19,25] else 'new_heldout_record',
            'calibration_sweeps':[0,32768],'evaluation_sweeps':[65536,131072]})
    (a.output/'selection.json').write_text(json.dumps({'records':plan,'selection_before_scoring':True,
        'scope':'frozen transfer and record-prefix-conditioned prediction are separate claims; new records are not refit on evaluation tails'},indent=2)+'\n')
    def download(row):
        path=a.root/'raw'/row['filename'];url='https://soma.ece.mcmaster.ca/ipix/dartmouth/data/'+row['filename']
        if not path.exists():
            try:
                part=path.with_suffix('.cdf.part');count=0
                with urllib.request.urlopen(url,timeout=40) as response,part.open('wb') as out:
                    expected=response.headers.get('Content-Length')
                    while True:
                        block=response.read(1_048_576)
                        if not block:break
                        count+=len(block)
                        if count>20_000_000:raise ValueError('oversize raw response')
                        out.write(block)
                if expected is not None and count!=int(expected):raise ValueError('truncated HTTP body')
                validate(part);part.replace(path)
            except Exception as e:return dict(row,status='download_failed',url=url,error=str(e))
        try:validate(path)
        except Exception as e:return dict(row,status='invalid_existing_file',url=url,error=str(e))
        data=path.read_bytes();print(row['record_id'],len(data),flush=True)
        return dict(row,status='obtained',url=url,bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    with ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(download,plan))
    (a.output/'manifest.json').write_text(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'files':files,
        'source':'https://soma.ece.mcmaster.ca/ipix/dartmouth/datasets.html','terms':'research citation and communication conditions; no external messages sent'},indent=2)+'\n')


if __name__=='__main__':main()
