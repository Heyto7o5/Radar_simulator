#!/usr/bin/env python3
"""Public-source access/format audit; gated resources are never downloaded."""
import hashlib,json,urllib.request,zipfile
from pathlib import Path
from remote_zip import HTTPRangeFile

ROOT=Path('/nas/datasets/hyt/datasets/radar_open/surface_rd_source_audit/20261015')

def fetch(url,limit=2_000_000):
    with urllib.request.urlopen(url,timeout=25) as r:data=r.read(limit+1)
    if len(data)>limit:raise ValueError('metadata budget exceeded')
    return data

def main():
    ROOT.mkdir(parents=True,exist_ok=True);results={}
    url='https://huggingface.co/api/datasets/AnonymousAuthors0018/sim2radar_artifacts?expand%5B%5D=gated&expand%5B%5D=sha'
    try:
        payload=fetch(url);meta=json.loads(payload)
        (ROOT/'sim2radar_public_metadata.json').write_bytes(payload)
        results['sim2radar']={'url':url,'gated':meta.get('gated'),'sha':meta.get('sha'),'data_download_attempted':False,
            'reason':'requires approved access if gated; only public metadata inspected'}
    except Exception as exc:results['sim2radar']={'url':url,'error':str(exc)}
    print('Sim2Radar',results['sim2radar'],flush=True)
    previous=Path('/nas/datasets/hyt/datasets/radar_open/radical_sample/manifest.json')
    if previous.exists():
        m=json.loads(previous.read_text());results['radical']={'source':m['source'],'status':m['status'],'error':m.get('error'),
            'bytes_downloaded':m['downloaded_source_bytes'],'scope':'official public sample link; HEAD reachable but actual ranged GET was denied; no bypass attempted'}
    try:
        url='https://zenodo.org/api/records/10938617';payload=fetch(url);meta=json.loads(payload)
        (ROOT/'dart_zenodo_metadata.json').write_bytes(payload)
        files=[{'name':x['key'],'bytes':x['size'],'url':x['links']['self'],'checksum':x['checksum']} for x in meta['files']]
        item={'source':url,'license':meta['metadata'].get('license'),'files':files,'bulk_archive_downloaded':False}
        selected=next(x for x in files if x['name']=='lab-1.zip')
        with zipfile.ZipFile(HTTPRangeFile(selected['url'],selected['bytes'])) as archive:
            index=[{'name':x.filename,'bytes':x.file_size,'compressed_bytes':x.compress_size,'compression':x.compress_type} for x in archive.infolist()]
            item['lab1_members']=index;item['metadata_files']=[]
            for info in archive.infolist():
                path=Path(info.filename)
                if path.is_absolute() or '..' in path.parts:raise ValueError('unsafe ZIP member')
                if info.is_dir() or info.file_size>1_000_000 or path.suffix.lower() not in ['.json','.yaml','.yml','.txt','.cfg']:continue
                data=archive.read(info);dest=ROOT/'dart_lab1_metadata'/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
                item['metadata_files'].append({'source_member':info.filename,'file':str(dest),'sha256':hashlib.sha256(data).hexdigest()})
        results['dart']=item
    except Exception as exc:
        results['dart']={'error':str(exc),'bulk_archive_downloaded':False}
    (ROOT/'audit.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
    print(json.dumps(results,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
