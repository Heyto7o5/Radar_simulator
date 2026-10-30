#!/usr/bin/env python3
"""Acquire two original IPIX Dartmouth records plus source processing notes."""
import argparse,hashlib,json,urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path

BASE='https://soma.ece.mcmaster.ca/ipix/dartmouth/'
FILES=['datasets.html','cdfhowto.html','mfiles/ipixload.m','data/19931107_135603_starea.cdf','data/19931107_141630_starea.cdf']

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/nas/datasets/hyt/datasets/radar_open/ipix_dartmouth'))
    args=p.parse_args();root=args.root.resolve();project=Path(__file__).resolve().parents[1]
    if root==project or project in root.parents:raise SystemExit('raw data must stay outside source project')
    (root/'raw').mkdir(parents=True,exist_ok=True)
    def download(relative):
        path=root/'raw'/Path(relative).name;url=BASE+relative
        if not path.exists():
            data=urllib.request.urlopen(url,timeout=60).read()
            part=path.with_suffix(path.suffix+'.part');part.write_bytes(data);part.replace(path)
        data=path.read_bytes()
        if relative.endswith('.cdf') and not data.startswith(b'CDF'):raise ValueError('not a NetCDF classic file')
        print(path.name,len(data),flush=True)
        return {'file':path.name,'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    with ThreadPoolExecutor(max_workers=4) as ex:files=list(ex.map(download,FILES))
    manifest={'dataset':'McMaster IPIX Dartmouth 1993','downloaded_utc':datetime.now(timezone.utc).isoformat(),
        'source':BASE+'datasets.html','terms':'Research use: cite source; source requests keeping maintainers informed of results. No external messages sent.',
        'records':{'19931107_135603_starea.cdf':{'official_id':17,'target_primary_one_based':9,'target_secondary_one_based':[8,9,10,11]},
                   '19931107_141630_starea.cdf':{'official_id':18,'target_primary_one_based':9,'target_secondary_one_based':[8,9,10,11]}},
        'target':'1 m diameter wire-mesh-covered styrofoam sphere, with sea clutter; exact motion unknown',
        'files':files}
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
