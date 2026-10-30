#!/usr/bin/env python3
"""Inspect ZIP directories over HTTP Range; never fetch full radar/MVS archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import zipfile
from remote_zip import HTTPRangeFile

OBJECTS=('24_corner_reflector','28_v1_metal_plate','29_v2_metal_plate','34_metal_angle')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    source=json.loads(a.catalog.read_text())['sources']['maroon_zenodo']
    files=[f for f in source['files'] if f['key'] in ('1_maroon_v2_radar_raw.zip','5_maroon_v2_mvs.zip')]
    def inspect(item):
        result={'archive':item['key'],'bytes':item['size'],'url':item['url']}
        try:
            with zipfile.ZipFile(HTTPRangeFile(item['url'],item['size'])) as archive:
                all_names=[i.filename for i in archive.infolist()]
                selected=[{'name':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size} for i in archive.infolist()
                          if any('/'+obj+'/' in i.filename for obj in OBJECTS) and
                          (i.filename.endswith('.obj') or '/calibrated_data/' in i.filename)]
            result.update(status='range_catalog_obtained',selected=selected,total_members=len(all_names))
        except Exception as exc:result.update(status='blocked',error=str(exc),bulk_download_refused=True)
        print(item['key'],result['status'],len(result.get('selected',[])),flush=True)
        return result
    with ThreadPoolExecutor(max_workers=2) as ex:results=list(ex.map(inspect,files))
    (a.output/'selected_catalog.json').write_text(json.dumps({'objects_selected_before_results':OBJECTS,'archives':results},indent=2)+'\n')


if __name__=='__main__':main()
