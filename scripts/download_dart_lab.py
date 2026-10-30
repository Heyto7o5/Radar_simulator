#!/usr/bin/env python3
"""Download one official lab archive resumably; verify MD5 before extraction."""
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path
import h5py
import numpy as np

ROOT = Path('/nas/datasets/hyt/datasets/radar_open/dart_lab')

def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''): h.update(block)
    return h.hexdigest()

def main():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    entry = manifest['archive']
    path = ROOT/'raw/lab-1.zip'
    partial = path.with_suffix('.zip.part')
    if not path.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > entry['bytes']: raise ValueError('oversized partial archive')
        if offset < entry['bytes']:
            request = urllib.request.Request(entry['url'], headers={'Range':f'bytes={offset}-'})
            with urllib.request.urlopen(request, timeout=60) as response:
                expected = f"bytes {offset}-{entry['bytes']-1}/{entry['bytes']}"
                if response.status != 206 or response.headers.get('Content-Range') != expected:
                    raise IOError('unexpected range response; will not append')
                with partial.open('ab') as target:
                    while True:
                        chunk = response.read(8*1024**2)
                        if not chunk: break
                        if offset+len(chunk)>entry['bytes']:raise IOError('response exceeds archive length')
                        target.write(chunk);offset+=len(chunk)
                        print(f'downloaded {offset}/{entry["bytes"]}',flush=True)
        if partial.stat().st_size != entry['bytes']: raise IOError('incomplete archive')
        if 'md5:'+digest(partial,'md5') != entry['checksum']: raise IOError('archive MD5 mismatch')
        partial.rename(path)
    if path.stat().st_size != entry['bytes'] or 'md5:'+digest(path,'md5') != entry['checksum']:
        raise IOError('archive validation failed')
    output = ROOT/'extracted/lab-1/radar.h5'
    with zipfile.ZipFile(path) as archive:
        if not output.exists():
            temp=output.with_suffix('.h5.part')
            with archive.open('lab-1/radar.h5') as src,temp.open('wb') as dst:
                shutil.copyfileobj(src,dst,length=8*1024**2)
            temp.rename(output)
        if output.stat().st_size != archive.getinfo('lab-1/radar.h5').file_size:
            raise IOError('radar size mismatch')
    with h5py.File(ROOT/'extracted/lab-1/trajectory.h5') as trajectory,h5py.File(output) as radar:
        mask,valid=trajectory['mask'][:],trajectory['valid'][:]
        if int(mask.sum())!=len(valid) or len(mask)!=len(radar['rad']):raise ValueError('alignment mismatch')
        aligned=np.flatnonzero(valid)[:3];raw=np.flatnonzero(mask)[aligned]
        values=radar['rad'][raw]
    subset=ROOT/'extracted/lab-1/radar_first_valid_subset.npz'
    if subset.exists(): raise FileExistsError('refusing to overwrite subset')
    np.savez_compressed(subset,rad=values,raw_indices=raw,aligned_indices=aligned)
    manifest.update(full_archive_checksum_verified=True,radar_signal_acquired=True)
    manifest['files'].append({'file':str(output.relative_to(ROOT)),'bytes':output.stat().st_size,
                              'sha256':digest(output,'sha256')})
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('verified archive and extracted radar plus preview subset',flush=True)

if __name__=='__main__':main()
