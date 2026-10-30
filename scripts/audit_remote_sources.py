#!/usr/bin/env python3
"""Bounded official catalog inspection; no account acceptance or bulk archives."""
import argparse
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile


def fetch(url,limit=4_000_000):
    with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'radar-platform-research/0.1'}),timeout=25) as r:
        data=r.read(limit+1)
    if len(data)>limit:raise ValueError('response exceeds bounded catalog size')
    return data


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    result={'time_utc':datetime.now(timezone.utc).isoformat(),'bulk_data_downloaded':False,'sources':{}}
    checks={
        'maroon_zenodo':'https://zenodo.org/api/records/18254440',
        'maroon_readme':'https://raw.githubusercontent.com/vwirth/maroon/master/README.md',
        'sim2radar_api':'https://huggingface.co/api/datasets/AnonymousAuthors0018/sim2radar_artifacts',
        'sim2radar_card':'https://huggingface.co/datasets/AnonymousAuthors0018/sim2radar_artifacts/raw/main/README.md',
        'coloradar':'https://arpg.colorado.edu/coloradar/',
        'ipix':'https://soma.ece.mcmaster.ca/ipix/dartmouth/datasets.html',
    }
    for name,url in checks.items():
        try:
            data=fetch(url);(a.output/(name+'.txt')).write_bytes(data)
            item={'status':'obtained','url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            if name=='maroon_zenodo':
                metadata=json.loads(data)
                item['files']=[{'key':f['key'],'size':f['size'],'url':f['links']['self']} for f in metadata['files']]
                # Only the small official metadata ZIP, not the radar or MVS archives.
                matches=[f for f in item['files'] if f['key']=='0_maroon_v2_meta.zip']
                if matches and matches[0]['size']<2_000_000:
                    payload=fetch(matches[0]['url'],2_000_000)
                    (a.output/'maroon_meta.zip').write_bytes(payload)
                    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                        names=archive.namelist()
                        item['metadata_members']=names
                        records={}
                        for n in names:
                            if n.endswith('metadata.json') and archive.getinfo(n).file_size<200_000:
                                records[n]=json.loads(archive.read(n))
                        (a.output/'maroon_capture_catalog.json').write_text(json.dumps(records,indent=2))
                        item['capture_metadata_count']=len(records)
            elif name=='sim2radar_api':
                meta=json.loads(data);item['gated']=meta.get('gated');item['visible_filenames']=[s['rfilename'] for s in meta.get('siblings',[])][:50]
            result['sources'][name]=item
        except Exception as e:
            result['sources'][name]={'status':'blocked_or_unavailable','url':url,'error':str(e),'no_bypass_attempted':True}
        print(name,result['sources'][name]['status'],flush=True)
    (a.output/'catalog_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
