#!/usr/bin/env python3
"""Fetch all repeated filtered measurements for preselected exact-mesh scenes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import zipfile
from remote_zip import HTTPRangeFile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--catalog',type=Path,required=True);p.add_argument('--metadata-zip',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    project=Path(__file__).resolve().parents[1]
    if a.output.resolve()==project or project in a.output.resolve().parents:raise ValueError('public raw data must stay outside project')
    a.output.mkdir(parents=True,exist_ok=True)
    cat=json.loads(a.catalog.read_text())
    mvs=next(x for x in cat['archives'] if x['archive']=='5_maroon_v2_mvs.zip')
    radar=next(x for x in cat['archives'] if x['archive']=='1_maroon_v2_radar_raw.zip')
    meshes=[x['name'] for x in mvs['selected'] if x['name'].endswith('/mesh_synthetic.obj')]
    scenes=sorted(n.rsplit('/photogrammetry/',1)[0] for n in meshes)
    selected={mvs['archive']:meshes,radar['archive']:[x['name'] for x in radar['selected']
              if any(x['name'].startswith(s+'/') for s in scenes) and x['name'].endswith('_emptyfiltered.npy')]}
    # The selection depends ONLY on paired geometry availability, before scores.
    selection={'scenes':scenes,'reason':'all exact synthetic meshes in four preselected simple metal objects; every repeated filtered capture',
               'members':selected,'expected_repeated_captures':len(selected[radar['archive']])}
    (a.output/'selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    entries=[]
    with zipfile.ZipFile(a.metadata_zip) as z:
        for name in z.namelist():
            if any(name.startswith(s+'/') for s in scenes) and name.endswith(('alignment.json','metadata.json')):
                path=a.output/name;path.parent.mkdir(parents=True,exist_ok=True)
                data=z.read(name);path.write_bytes(data)
                entries.append({'name':name,'bytes':len(data),'source':'official metadata ZIP','sha256':hashlib.sha256(data).hexdigest()})
    def download_member(task):
        item,name=task;path=a.output/name
        if not path.exists():
            error=None
            for attempt in range(3):
                try:
                    with zipfile.ZipFile(HTTPRangeFile(item['url'],item['bytes'])) as z:
                        info=z.getinfo(name)
                        if info.file_size>25_000_000:raise ValueError('unexpectedly large selected member')
                        data=z.read(name) # validates member CRC
                    path.parent.mkdir(parents=True,exist_ok=True)
                    part=path.with_suffix(path.suffix+'.part');part.write_bytes(data);part.replace(path)
                    error=None;break
                except Exception as exc:error=str(exc)
            if error:return {'name':name,'status':'failed','error':error}
        data=path.read_bytes()
        print(name,len(data),flush=True)
        return {'name':name,'status':'obtained','bytes':len(data),'source_archive':item['url'],'sha256':hashlib.sha256(data).hexdigest()}
    tasks=[(item,name) for item in (mvs,radar) for name in selected[item['archive']]]
    with ThreadPoolExecutor(max_workers=4) as ex:entries.extend(ex.map(download_member,tasks))
    manifest={'retrieved_utc':datetime.now(timezone.utc).isoformat(),'source':'https://zenodo.org/records/18254440',
              'selection':selection,'files':entries,'raw_archive_downloaded':False,
              'license':'see license in Zenodo metadata; retained alongside campaign source catalog'}
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
