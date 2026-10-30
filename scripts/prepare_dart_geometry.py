#!/usr/bin/env python3
"""Acquire only independent geometry/pose and metadata from an official DART scene."""
import hashlib,json,zipfile
from pathlib import Path
import h5py
import numpy as np
from remote_zip import HTTPRangeFile

ROOT=Path('/nas/datasets/hyt/datasets/radar_open/dart_lab')
AUDIT=Path('/nas/datasets/hyt/datasets/radar_open/surface_rd_source_audit/20261015/audit.json')

def main():
    for p in [ROOT/'raw',ROOT/'extracted',ROOT/'processed']:p.mkdir(parents=True,exist_ok=True)
    meta=json.loads(AUDIT.read_text())['dart']
    entry=next(x for x in meta['files'] if x['name']=='lab-1.zip')
    selected=['lab-1/metadata.json','lab-1/sensor.json','lab-1/trajectory.h5','lab-1/map.npz']
    manifest={'source':'https://zenodo.org/records/10938617','license':meta['license'],'archive':entry,
        'selected_members':selected,'full_archive_checksum_verified':False,
        'geometry_provenance':'official DART README describes map.npz as obtained from LiDAR',
        'map_values_are_not_independent_radar_RCS':True,'radar_signal_acquired':False,'files':[]}
    with zipfile.ZipFile(HTTPRangeFile(entry['url'],entry['bytes'])) as z:
        for name in selected:
            info=z.getinfo(name)
            if info.file_size>100_000_000 or info.compress_size>10_000_000:raise ValueError('member exceeds bounded extraction budget')
            target=ROOT/'extracted'/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():data=target.read_bytes()
            else:data=z.read(name);target.write_bytes(data)
            if len(data)!=info.file_size:raise ValueError('member size mismatch')
            manifest['files'].append({'file':str(target.relative_to(ROOT)),'source_member':name,'bytes':len(data),
                'zip_crc32':f'{info.CRC:08x}','sha256':hashlib.sha256(data).hexdigest()})
            print('obtained',name,len(data),flush=True)
    fields={}
    with np.load(ROOT/'extracted/lab-1/map.npz',allow_pickle=False) as z:
        for key in z.files:
            a=z[key];fields[key]={'shape':list(a.shape),'dtype':str(a.dtype)}
            if a.size<30:fields[key]['values']=a.tolist()
            elif a.dtype.kind in 'fiu':fields[key]['minmax']=[float(np.nanmin(a)),float(np.nanmax(a))]
    manifest['map_fields']=fields
    with h5py.File(ROOT/'extracted/lab-1/trajectory.h5','r') as f:
        fields={}
        def visit(name,obj):
            if isinstance(obj,h5py.Dataset):fields[name]={'shape':list(obj.shape),'dtype':str(obj.dtype)}
        f.visititems(visit);manifest['trajectory_fields']=fields
    (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(json.dumps(manifest,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
