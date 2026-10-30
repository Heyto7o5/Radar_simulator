#!/usr/bin/env python3
"""Download the public classification tile covering the predeclared DEM sites."""
from datetime import datetime,timezone
import hashlib,json,os
from pathlib import Path
import httpx

ROOT=Path(__file__).resolve().parents[1]
BASE='https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/'
FILES=['map/ESA_WorldCover_10m_2021_v200_N45E006_Map.tif','docs/WorldCover_PUM_V2.0.pdf']


def main():
    root=Path(os.environ.get('RADAR_DATA_ROOT','/nas/datasets/hyt/datasets/radar_open')).resolve()/'worldcover_2021'
    if root.is_relative_to(ROOT):raise ValueError('public raw data must stay outside the source tree')
    raw=root/'raw';raw.mkdir(parents=True,exist_ok=True);entries=[]
    with httpx.Client(timeout=120,follow_redirects=True) as client:
        for key in FILES:
            url=BASE+key;head=client.head(url);head.raise_for_status()
            size=int(head.headers['content-length']);etag=head.headers.get('etag','').strip('"');target=raw/Path(key).name
            if not target.exists():
                partial=target.with_suffix(target.suffix+'.part');offset=partial.stat().st_size if partial.exists() else 0
                with client.stream('GET',url,headers={'Range':f'bytes={offset}-'} if offset else {}) as response:
                    response.raise_for_status()
                    if offset and response.status_code!=206:raise ValueError('resume not honored; partial file retained')
                    with partial.open('ab' if offset else 'xb') as handle:
                        for block in response.iter_bytes(1024*1024):handle.write(block)
                if partial.stat().st_size!=size:raise ValueError('download size mismatch')
                partial.rename(target)
            payload=target.read_bytes()
            if len(payload)!=size:raise ValueError('existing source size mismatch; not overwritten')
            entries.append(dict(url=url,file=str(target.relative_to(root)),bytes=size,etag=etag,
                                sha256=hashlib.sha256(payload).hexdigest(),
                                verification='HTTPS transfer length and local SHA256; multipart ETag is not a whole-file MD5'))
            print(json.dumps({'file':target.name,'bytes':size}),flush=True)
    manifest={'dataset':'ESA WorldCover 2021 v200','source':'https://esa-worldcover.org/en/data-access',
        'doi':'https://doi.org/10.5281/zenodo.7254221','license':'CC-BY-4.0',
        'attribution':'© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium',
        'downloaded_utc':datetime.now(timezone.utc).isoformat(),'files':entries,
        'limits':['Classification product, not field-verified material truth','Different acquisition epoch from Copernicus DSM','Does not supply radar scattering parameters']}
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
