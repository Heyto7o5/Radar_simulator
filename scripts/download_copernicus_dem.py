#!/usr/bin/env python3
"""Acquire a public GLO-30 tile, preserving source checksums and license."""
from datetime import datetime,timezone
import argparse,hashlib,json,os
from pathlib import Path
import xml.etree.ElementTree as ET
import httpx

ROOT=Path(__file__).resolve().parents[1]
HOST='https://copernicus-dem-30m.s3.eu-central-1.amazonaws.com/'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=ROOT/'configs/real_dem_validation_plan.json')
    args=parser.parse_args();plan=json.loads(args.plan.read_text());tile=plan['tile']
    root=Path(os.environ.get('RADAR_DATA_ROOT','/nas/datasets/hyt/datasets/radar_open')).resolve()/'copernicus_glo30'
    if root.is_relative_to(ROOT):raise ValueError('raw data must not be placed inside the source tree')
    raw=root/'raw'/tile;raw.mkdir(parents=True,exist_ok=True)
    with httpx.Client(follow_redirects=True,timeout=120) as client:
        listing=client.get(HOST,params={'list-type':'2','prefix':tile+'/'});listing.raise_for_status()
        namespace={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
        entries=[]
        for item in ET.fromstring(listing.content).findall('s:Contents',namespace):
            key=item.findtext('s:Key',namespaces=namespace)
            if not (key.endswith('_DEM.tif') and '/PREVIEW/' not in key or key.endswith('.xml') or key.endswith('/INFO/eula_F.pdf')):continue
            entry=dict(key=key,url=HOST+key,size=int(item.findtext('s:Size',namespaces=namespace)),
                       etag=item.findtext('s:ETag',namespaces=namespace).strip('"'))
            target=raw/Path(key).name
            if not target.exists():
                partial=target.with_suffix(target.suffix+'.part');offset=partial.stat().st_size if partial.exists() else 0
                headers={'Range':f'bytes={offset}-'} if offset else {}
                with client.stream('GET',entry['url'],headers=headers) as response:
                    response.raise_for_status()
                    if offset and response.status_code!=206:raise ValueError('server did not honor resume; retained partial file')
                    with partial.open('ab' if offset else 'xb') as handle:
                        for chunk in response.iter_bytes(1024*1024):handle.write(chunk)
                if partial.stat().st_size!=entry['size']:raise ValueError('download byte count mismatch')
                digest=hashlib.md5(partial.read_bytes()).hexdigest()
                if '-' not in entry['etag'] and digest!=entry['etag']:raise ValueError('object ETag/MD5 mismatch')
                partial.rename(target)
            payload=target.read_bytes()
            if len(payload)!=entry['size']:raise ValueError('existing file size mismatch; refusing overwrite')
            md5=hashlib.md5(payload).hexdigest()
            if '-' not in entry['etag'] and md5!=entry['etag']:raise ValueError('existing file ETag mismatch')
            entries.append({**entry,'file':str(target.relative_to(root)), 'sha256':hashlib.sha256(payload).hexdigest(),
                            'md5':md5,'etag_md5_verified':'-' not in entry['etag']})
            print(json.dumps({'file':target.name,'bytes':len(payload),'verified':True}),flush=True)
    manifest={'dataset':plan['dataset'],'source_registry':plan['source_registry'],'tile':tile,
        'downloaded_utc':datetime.now(timezone.utc).isoformat(),'files':entries,
        'surface_type':plan['surface_type'],'horizontal_crs':plan['horizontal_crs'],'vertical_datum':plan['vertical_datum'],
        'license':'Copernicus DEM GLO-30 free and open license, retained in raw INFO/eula_F.pdf source object',
        'derived_attribution':'produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved',
        'selection_plan_sha256':hashlib.sha256(args.plan.read_bytes()).hexdigest()}
    (root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(root/'manifest.json',flush=True)


if __name__=='__main__':main()
