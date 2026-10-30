#!/usr/bin/env python3
"""Extract a bounded Maqu calibration/angular subset without fetching 21 GB."""
import argparse,hashlib,json,urllib.request,zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
from remote_zip import HTTPRangeFile

URL='https://phys-techsciences.datastations.nl/api/access/datafile/259802'
SIZE=9665656345

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/nas/datasets/hyt/datasets/radar_open/maqu_scatterometer'))
    args=p.parse_args();root=args.root.resolve();project=Path(__file__).resolve().parents[1]
    if root==project or project in root.parents:raise SystemExit('raw data must stay outside source project')
    (root/'raw').mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(HTTPRangeFile(URL,SIZE)) as z:
        members=[i for i in z.infolist() if not i.is_dir() and
          ('Scatterometer_Calibration_Sky/' in i.filename or '/Scatterometer_Angular_Vars/sigma0_' in i.filename or i.filename=='original/Readme.pdf')]
        index=[{'name':i.filename,'bytes':i.file_size} for i in z.infolist()]
        # Preserve central metadata once so each worker can read just its member.
    (root/'archive_index.json').write_text(json.dumps(index,indent=2))
    if sum(i.file_size for i in members)>60_000_000:raise ValueError('subset unexpectedly large')
    def extract(info):
        relative=Path(info.filename)
        if '..' in relative.parts or relative.is_absolute():raise ValueError('unsafe member path')
        path=root/'raw'/relative;path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            reader=HTTPRangeFile(URL,SIZE)
            # Using an independent ZipFile keeps concurrent seeks independent.
            with zipfile.ZipFile(reader) as archive: payload=archive.read(info.filename)
            temporary=path.with_suffix(path.suffix+'.part');temporary.write_bytes(payload);temporary.replace(path)
        data=path.read_bytes()
        if len(data)!=info.file_size:raise ValueError('member length mismatch')
        print(relative.name,len(data),flush=True)
        return {'file':str(relative),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'source_archive':URL,'source_member':info.filename,'zip_crc32':f'{info.CRC:08x}'}
    with ThreadPoolExecutor(max_workers=4) as ex:files=list(ex.map(extract,members))
    for name,url in [('processing_code.rar','https://phys-techsciences.datastations.nl/api/access/datafile/4398'),
                     ('reference_targets.xlsx','https://essd.copernicus.org/articles/13/2819/2021/essd-13-2819-2021-t05.xlsx')]:
        path=root/'raw'/name
        if not path.exists():path.write_bytes(urllib.request.urlopen(url,timeout=60).read())
        data=path.read_bytes();files.append({'file':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'url':url})
    (root/'manifest.json').write_text(json.dumps({'dataset':'Maqu broadband scatterometer',
        'doi':'10.17026/DANS-ZFB-QEGY','code_doi':'10.17026/dans-xyf-fmkk','license':'CC BY 4.0 data; Apache-2.0 processing code',
        'downloaded_utc':datetime.now(timezone.utc).isoformat(),'source':'https://essd.copernicus.org/articles/13/2819/2021/index.html','files':files},ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':main()
