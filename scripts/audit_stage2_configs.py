#!/usr/bin/env python3
"""Prove that each declared control changes only its permitted physical inputs."""
import argparse,hashlib,json
from pathlib import Path


def changed(a,b,path=()):
    if type(a)!=type(b):return [path]
    if isinstance(a,dict):
        if set(a)!=set(b):return [path]
        return sum((changed(a[k],b[k],path+(k,)) for k in a),[])
    if isinstance(a,list):
        if len(a)!=len(b):return [path]
        return sum((changed(x,y,path+(i,)) for i,(x,y) in enumerate(zip(a,b))),[])
    return [] if a==b else [path]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('audit already exists')
    plan=json.loads(a.plan.read_text());rows=[]
    for row in plan['cases']:
        path=a.plan.parent/row['config']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['config_sha256']:raise ValueError('configuration hash changed')
        value=json.loads(path.read_text());base=json.loads((a.plan.parent/f'{row["site"]}_hamming_0_s{row["seed"]}.json').read_text())
        differences=changed(base,value);bad=[]
        for key in differences:
            if key==('name',):continue
            if row['variant']=='uniform_0' and key[:3]==('radiation','transmit_aperture','amplitudes'):continue
            if row['variant']=='hamming_8' and key==('radiation','transmit_aperture','steering_azimuth_deg'):continue
            if row['variant'] in ('tree_plus6','water_minus6') and key[:4]==('scene','terrain','land_scattering','tables') and len(key)==8 and key[5]=='sigma0_db':
                label='tree_cover' if row['variant']=='tree_plus6' else 'permanent_water'
                if base['scene']['terrain']['land_scattering']['tables'][key[4]]['land_cover']==label:continue
            bad.append(key)
        if bad:raise ValueError(f'undeclared input changes in {row["case"]}: {bad}')
        rows.append({'case':row['case'],'changes':['/'.join(map(str,k)) for k in differences],'passed':True})
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'cases':rows,'all_declared_only':True},indent=2))
    print(json.dumps({'checked_cases':len(rows),'all_declared_only':True}))


if __name__=='__main__':main()
