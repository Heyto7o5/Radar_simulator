#!/usr/bin/env python3
"""Execute a configuration-only acceptance matrix using the same public CLI.

This verifies framework integration, NOT measured-scene fidelity. Every case
gets its complete input and native output; no synthetic case is called real.
"""
import argparse
import copy
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def cases():
    read=lambda path:json.loads((ROOT/'configs'/path).read_text())
    high=read('unified/high_altitude_1ghz.json');high['sequence']['frame_count']=1
    result={'high_altitude_1ghz':high,'moving_plate_po':read('lab_moving_plate_po.json')}
    for label,nt,nr in [('siso',1,1),('simo',1,4),('mimo',2,4)]:
        r=read('unified/lab_fmcw_mimo.json')
        r['array']={'layout':'siso' if label=='siso' else 'ula','num_tx':nt,'num_rx':nr}
        result['fmcw_'+label]=r
    for terrain in ['flat','hills','dem']:
        r=copy.deepcopy(high);r['scene']['objects']=[]
        t={'id':'ground','kind':terrain,'relative_reflectivity':.1}
        if terrain=='dem':
            t.update(north_m=[750,875,1000],east_m=[-100,0,100],elevation_up_m=[[0,1,2],[1,2,3],[0,1,2]])
        else:
            t.update(north_bounds_m=[750,1000],east_bounds_m=[-100,100],spacing_m=125)
            if terrain=='hills':t['hills']=[dict(center_ne_m=[875,0],height_m=10,sigma_ne_m=[100,100])]
        r['scene']['terrain']=t;result['terrain_'+terrain]=r
    for method in ['image_paths','physical_optics']:
        r=read('surface_plane.json');r['method']=method;result['sfcw_'+method]=r
    for family in ['white_gaussian','correlated_gaussian','compound_gamma']:
        r=read('unified/statistical_background.json');r['model']['family']=family;result[family]=r
    result['historical_block_texture']=read('unified/block_texture_background.json')
    result['historical_separable_if']=read('unified/legacy_separable_if.json')
    r=read('unified/lab_fmcw_mimo.json');r['receiver_noise']={'variance':.001,'seed':42}
    result['fmcw_with_adc_noise']=r
    # Optional high-level features must also reach the SAME public runner.
    r=read('unified/high_altitude_ridge_visibility.json')
    result['terrain_with_occlusion']=r
    r=read('unified/phased_scan_10ghz.json')
    result['transmit_aperture_scan']=r
    r=copy.deepcopy(result['fmcw_simo'])
    r['array']={'layout':'ura','num_tx':2,'num_rx':2}
    r['processing']['elevation_deg']=[-20,0,20]
    result['planar_array']=r
    r=copy.deepcopy(result['fmcw_siso'])
    r['sequence']['frame_count']=1
    r['platform']['trajectory']={'time_s':[0,.1],
        'position_ned_m':[[0,0,0],[.1,0,0]],'velocity_ned_mps':[[1,0,0],[1,0,0]],
        'quaternion_xyzw':[[0,0,0,1],[0,0,0,1]],'provenance':'Analytic constant-motion integration case, not measured pose'}
    result['sampled_platform_motion']=r
    r=copy.deepcopy(result['terrain_dem'])
    r['scene']['terrain'].update(relative_reflectivity=1.,land_cover_map=[['A','B'],['B','A']],
        land_scattering={'land_cover':'A','polarization':'vv','tables':[
            {'land_cover':label,'polarization':'vv','frequencies_hz':[.9e9,1.1e9],
             'incidence_deg':[0,90],'sigma0_db':[[db,db],[db,db]],'evidence':'assumed',
             'source':'Explicit integration-test constants, not measured material data'}
            for label,db in [('A',-20),('B',-10)]]})
    result['classified_land_scattering']=r
    return result


def verify_output(directory):
    manifest=json.loads((directory/'run.json').read_text())
    if manifest['status']!='complete':raise ValueError('incomplete run')
    product_names=set();frame_count=0
    for item in manifest['frames']:
        folder=directory/item['directory'];meta=json.loads((folder/'metadata.json').read_text())
        frame_count+=1
        for name,spec in meta['products'].items():
            file=folder/spec['file']
            if file.stat().st_size!=spec['bytes']:raise ValueError('binary size mismatch')
            array=np.fromfile(file,dtype=spec['dtype']).reshape(spec['shape'])
            if not np.isfinite(array).all():raise ValueError('nonfinite product')
            if name in ('adc','spectrum','iq','if') and not np.any(abs(array)>0):
                raise ValueError('nonempty acceptance scene produced no signal')
            product_names.add(name)
    return {'frames':frame_count,'products':sorted(product_names),'elapsed_s':manifest['total_elapsed_s'],
            'request_sha256':manifest['request_sha256'],'source_sha256':manifest['source_sha256']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--dart-config',type=Path)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    configs=cases()
    if args.dart_config:configs['dart_prepared_scene']=json.loads(args.dart_config.read_text())
    rows=[]
    for name,config in configs.items():
        source=args.output/(name+'.json');source.write_text(json.dumps(config,ensure_ascii=False,indent=2))
        destination=args.output/name
        command=[sys.executable,str(ROOT/'scripts/run_scene.py'),'--config',str(source.resolve()),'--output',str(destination.resolve())]
        print('Running '+name,flush=True)
        with (args.output/(name+'.log')).open('w') as log:
            process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=1800)
        row={'case':name,'input':source.name,'output':name,'passed':False}
        try:
            if process.returncode:raise ValueError(f'CLI exit {process.returncode}')
            row.update(verify_output(destination));row['passed']=True
        except Exception as exc:row['error']=str(exc)
        rows.append(row)
        (args.output/'matrix.json').write_text(json.dumps({'scope':'configuration integration, not measured fidelity','cases':rows},indent=2))
    table=''.join('<tr><td>'+html.escape(r['case'])+'</td><td>'+('通过' if r['passed'] else '失败')+'</td><td>'+str(r.get('frames','—'))+'</td><td>'+html.escape(', '.join(r.get('products',[])))+'</td><td>'+f"{r.get('elapsed_s',0):.3f}"+'</td></tr>' for r in rows)
    page='<html lang="zh"><meta charset="UTF-8"><title>统一框架配置验收</title><style>body{font:16px sans-serif;max-width:1200px;margin:40px auto;color:#233}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd;padding:10px;text-align:left}th{background:#eef}</style><h1>统一框架：换配置运行验收</h1><p>全部使用 scripts/run_scene.py --config；仅替换输入配置。这里验证入口、参数与输出完整性，不证明实测保真。地形案例为明确标注的理想几何。</p><table><tr><th>配置</th><th>结果</th><th>帧/批次数</th><th>原生输出</th><th>运行秒数</th></tr>'+table+'</table><p>静态 SFCW 没有 Doppler 轴；统计背景为距离门 IQ，不是 ADC。历史近似模型不等于标准物理 FMCW。</p></html>'
    (args.output/'report.html').write_text(page)
    if not all(r['passed'] for r in rows):raise SystemExit(1)
    print(json.dumps({'cases':len(rows),'passed':len(rows),'report':str(args.output/'report.html')}))


if __name__=='__main__':main()
