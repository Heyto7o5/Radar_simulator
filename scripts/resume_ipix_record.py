#!/usr/bin/env python3
"""Download a selected public CDF in verified HTTP byte ranges, avoiding truncation."""
import argparse,json,urllib.request,hashlib,re
from pathlib import Path
from repair_ipix_downloads import validate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--selection',type=Path,required=True);p.add_argument('--record',type=int,required=True)
    p.add_argument('--root',type=Path,default=Path('/nas/datasets/hyt/datasets/radar_open/ipix_dartmouth'));a=p.parse_args()
    row=next(r for r in json.loads(a.selection.read_text())['records'] if r['record_id']==a.record)
    path=a.root/'raw'/row['filename'];url='https://soma.ece.mcmaster.ca/ipix/dartmouth/data/'+row['filename']
    with urllib.request.urlopen(urllib.request.Request(url,headers={'Range':'bytes=0-0'}),timeout=30) as response:
        cr=response.headers.get('Content-Range','');match=re.fullmatch(r'bytes 0-0/(\d+)',cr)
        if response.status!=206 or match is None:raise ValueError('server does not support verified range downloads')
        size=int(match.group(1))
    if size>20_000_000:raise ValueError('unexpected CDF size')
    part=path.with_suffix('.cdf.ranged.part');chunks=[]
    with part.open('wb') as output:
        for start in range(0,size,1_048_576):
            end=min(size-1,start+1_048_575);last_error=None
            for attempt in range(4):
                try:
                    request=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                    with urllib.request.urlopen(request,timeout=30) as response:
                        if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{size}':raise ValueError('range mismatch')
                        data=response.read() # incomplete declared body raises, unlike read(max_bytes)
                    if len(data)!=end-start+1:raise ValueError('short range response')
                    output.write(data);chunks.append({'start':start,'end':end,'sha256':hashlib.sha256(data).hexdigest()});last_error=None;break
                except Exception as exc:last_error=exc
            if last_error:raise last_error
            print('verified range',start,end,flush=True)
    shape=validate(part)
    if path.exists():raise ValueError('refuse to replace an existing final record')
    part.replace(path)
    result={'record':a.record,'file':str(path),'bytes':size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'url':url,'chunks':chunks,**shape}
    (a.selection.parent/f'record_{a.record}_range_recovery.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
