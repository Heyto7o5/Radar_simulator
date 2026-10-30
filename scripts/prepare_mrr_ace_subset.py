#!/usr/bin/env python3
"""Bounded CC-BY ACE precipitation spectra acquisition; never download full 1.1 GB ZIP."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile
from remote_zip import HTTPRangeFile

ROOT=Path('/nas/datasets/hyt/datasets/radar_open/mrr_ace')
SOURCE='https://zenodo.org/records/3929289'
# Verified against Zenodo record API on 2026-10-15; immutable version 1.2.
ARCHIVE_URL=SOURCE+'/files/RawSpectra.zip?download=1'
ARCHIVE_SIZE=1118457403
SMALL={
    'README.txt':('37a44225bb7cc9e01ae6e09d8a9d64da',9017),
    'RawSpectra_data_file_header.txt':('facd1706218e0906495c1179080ddab2',1011),
    'PrecipitationEvents.txt':('11a8abe82b8be740c56f04a9426bff46',1036),
}

def main():
    for p in [ROOT/'raw',ROOT/'extracted',ROOT/'processed']:p.mkdir(parents=True,exist_ok=True)
    manifest={'source':SOURCE,'doi':'10.5281/zenodo.3929289','license':'CC BY 4.0',
        'instrument':'Metek MRR-2, 24 GHz vertically pointing FMCW precipitation radar',
        'data_kind':'measured Doppler spectral profiles, NOT raw complex ADC or a droplet digital twin',
        'archive':ARCHIVE_URL,'archive_size':ARCHIVE_SIZE,'full_archive_md5':'0a9f143953cefa84db31a99a9b108fa6',
        'full_archive_checksum_verified':False,'partial_acquisition':True,'files':[],'errors':[]}
    def save(): (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    for name,(md5,size) in SMALL.items():
        path=ROOT/'raw'/name;url=SOURCE+'/files/'+name+'?download=1'
        try:
            if path.exists():data=path.read_bytes()
            else:
                with urllib.request.urlopen(url,timeout=25) as r:data=r.read(size+1)
            if len(data)!=size or hashlib.md5(data).hexdigest()!=md5:raise ValueError('official metadata size/MD5 mismatch')
            path.write_bytes(data)
            manifest['files'].append({'file':str(path.relative_to(ROOT)),'source':url,'bytes':size,'md5':md5,'sha256':hashlib.sha256(data).hexdigest()})
            print('metadata',name,flush=True)
        except Exception as exc:manifest['errors'].append({'source':url,'error':str(exc)});print('metadata failed',name,str(exc),flush=True)
        save()
    try:
        with zipfile.ZipFile(HTTPRangeFile(ARCHIVE_URL,ARCHIVE_SIZE)) as z:
            entries=[x for x in z.infolist() if not x.is_dir() and x.filename.lower().endswith('.nc')]
            (ROOT/'archive_index.json').write_text(json.dumps([{'name':x.filename,'bytes':x.file_size,'compressed_bytes':x.compress_size} for x in entries],indent=2))
            eligible=sorted([x for x in entries if 0<x.file_size<=32_000_000],key=lambda x:(x.file_size,x.filename))
            chosen=eligible[:3]
            if not chosen or sum(x.compress_size for x in chosen)>64_000_000:raise ValueError('no bounded sample selection')
            manifest['selection_rule']='three smallest nonempty NetCDF members <=32 MB each, chosen from archive metadata before inspecting spectra'
            manifest['selected_members']=[x.filename for x in chosen];save()
            for info in chosen:
                relative=Path(info.filename)
                if relative.is_absolute() or '..' in relative.parts:raise ValueError('unsafe archive path')
                path=ROOT/'extracted'/relative;path.parent.mkdir(parents=True,exist_ok=True)
                data=z.read(info) # zipfile validates the member CRC.
                if len(data)!=info.file_size:raise ValueError('member size mismatch')
                path.write_bytes(data)
                manifest['files'].append({'file':str(path.relative_to(ROOT)),'source_member':info.filename,
                    'bytes':len(data),'zip_crc32':f'{info.CRC:08x}','sha256':hashlib.sha256(data).hexdigest()})
                save();print('spectra',info.filename,len(data),flush=True)
    except Exception as exc:
        manifest['errors'].append({'source':ARCHIVE_URL,'error':str(exc)});print('bounded spectra retrieval failed',str(exc),flush=True)
    manifest['spectra_files_obtained']=sum('source_member' in x for x in manifest['files']);save()
    print(json.dumps({'root':str(ROOT),'spectra_files':manifest['spectra_files_obtained'],'errors':manifest['errors']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
