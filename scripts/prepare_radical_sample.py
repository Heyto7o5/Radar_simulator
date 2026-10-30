#!/usr/bin/env python3
"""Read a bounded subset of the public RaDICaL SDK sample via verified HTTP ranges.

No account/auth bypass, full 293 MB download, Python pickle loading, or implied
surface geometry/material truth. Source sample license: CC BY 4.0 per SDK README.
"""
import argparse,hashlib,io,json,urllib.request
from pathlib import Path
import h5py
import numpy as np
from remote_zip import fetch_range

SOURCE='https://fireball.teckyianlim.me/file/flaming-cake/indoor_sample_50.h5'
SIZE=292664496
ROOT=Path('/nas/datasets/hyt/datasets/radar_open/radical_sample')

class BoundedHTTPFile(io.RawIOBase):
    def __init__(self,url,size,budget=64*1024**2,block=1024**2):
        self.url=url;self.size=size;self.budget=budget;self.block=block;self.pos=0;self.cache={};self.downloaded=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,n,whence=0):
        pos=n if whence==0 else self.pos+n if whence==1 else self.size+n
        if pos<0:raise ValueError('negative seek')
        self.pos=pos;return pos
    def read(self,n=-1):
        n=min(n if n>=0 else self.size-self.pos,self.size-self.pos)
        if n<=0:return b''
        if n>self.budget:raise ValueError('single HDF5 read exceeds subset budget')
        result=[]
        while n:
            start=self.pos//self.block*self.block
            if start not in self.cache:
                size=min(self.block,self.size-start)
                if self.downloaded+size>self.budget:raise ValueError('remote subset byte budget exceeded')
                self.cache[start]=fetch_range(self.url,self.size,start,size);self.downloaded+=size
                print('remote bytes',self.downloaded,flush=True)
            offset=self.pos-start;take=min(n,len(self.cache[start])-offset)
            result.append(self.cache[start][offset:offset+take]);self.pos+=take;n-=take
        return b''.join(result)
    def readinto(self,b):
        view=memoryview(b).cast('B');data=self.read(len(view));view[:len(data)]=data;return len(data)

def safe(value):
    if isinstance(value,bytes):return value.decode('utf8','replace')
    if isinstance(value,np.ndarray):return [safe(x) for x in value.tolist()] if value.ndim else safe(value.item())
    if isinstance(value,np.generic):return safe(value.item())
    if isinstance(value,(list,tuple)):return [safe(x) for x in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return str(value)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inspect-only',action='store_true');a=parser.parse_args()
    for p in [ROOT/'raw',ROOT/'extracted',ROOT/'processed']:p.mkdir(parents=True,exist_ok=True)
    manifest={'source':SOURCE,'source_size':SIZE,'expected_full_md5':'b195ff422cc4c979eeb81623899050cb',
        'full_source_checksum_verified':False,'license':'CC BY 4.0 according to official SDK README',
        'source_documentation':'https://github.com/moodoki/radical_sdk','frame_selection':[0,25,49],
        'surface_target_geometry_verified':False,'material_scattering_parameters_available':False,'files':[]}
    remote=BoundedHTTPFile(SOURCE,SIZE)
    try:
        with h5py.File(remote,'r') as f:
            info=[]
            def visit(name,obj):
                if isinstance(obj,h5py.Dataset):info.append({'name':name,'shape':list(obj.shape),'dtype':str(obj.dtype),'attrs':{k:safe(v) for k,v in obj.attrs.items()}})
            f.visititems(visit)
            manifest['datasets']=info;manifest['root_attributes']={k:safe(v) for k,v in f.attrs.items()}
            print(json.dumps(info,ensure_ascii=False),flush=True)
            if not a.inspect_only:
                path=ROOT/'extracted/frames_000_025_049.h5'
                if path.exists():raise ValueError('subset exists; refusing overwrite')
                temp=path.with_suffix('.h5.part')
                with h5py.File(temp,'w') as target:
                    target.attrs['source']=SOURCE;target.attrs['source_frames']=[0,25,49]
                    extracted=[]
                    for item in info:
                        ds=f[item['name']]
                        if ds.dtype.kind=='O':continue # no opaque/object payload interpretation
                        if ds.ndim and ds.shape[0]==50:
                            values=ds[[0,25,49]]
                            target.create_dataset(item['name'],data=values)
                            extracted.append(item['name'])
                        elif ds.size<=4096:
                            values=ds[()];target.create_dataset(item['name'],data=values)
                            extracted.append(item['name'])
                    target.attrs['metadata_semantics']='only leading-50 arrays subset; other small metadata preserved; no alignment inferred'
                temp.replace(path)
                manifest['extracted_datasets']=extracted
                manifest['files'].append({'file':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
        cfg='https://raw.githubusercontent.com/moodoki/radical_sdk/master/samples/indoor_human_rcs.cfg'
        with urllib.request.urlopen(cfg,timeout=20) as r:data=r.read(16001)
        if len(data)>16000:raise ValueError('configuration text too large')
        path=ROOT/'raw/indoor_human_rcs.cfg';path.write_bytes(data)
        manifest['files'].append({'file':str(path.relative_to(ROOT)),'source':cfg,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        manifest['status']='metadata_inspected' if a.inspect_only else 'bounded_subset_acquired'
    except Exception as exc:
        manifest['status']='incomplete';manifest['error']=str(exc);print(type(exc).__name__,str(exc),flush=True)
    finally:
        manifest['downloaded_source_bytes']=remote.downloaded
        manifest['fetched_block_sha256']={str(k):hashlib.sha256(v).hexdigest() for k,v in remote.cache.items()}
        (ROOT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
        remote.close()
    print(json.dumps({'status':manifest['status'],'downloaded_source_bytes':remote.downloaded,'root':str(ROOT)},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
