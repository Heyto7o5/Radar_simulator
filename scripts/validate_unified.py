#!/usr/bin/env python3
"""Prespecified batch checks of the generic scene/array framework, not measured fidelity."""
import argparse
import csv
import hashlib
from itertools import product
import json
from pathlib import Path
import zipfile

import numpy as np

from radar_platform.models import C0
from radar_platform.unified.schema import Simulation, Platform
from radar_platform.unified.geometry import sensor_basis, scan_directions, build_layout
from radar_platform.unified.engine import simulate_frame
from radar_platform.unified.processing import array_factor
from radar_platform.unified.storage import write_json


LAYOUTS={'SISO':('siso',1,1),'SIMO':('ula',1,4),'MIMO_ULA':('ula',2,4),'MIMO_URA':('ura',4,4)}
POSES={'forward':(1.,0.,0.),'nadir':(0.,0.,1.),'oblique':(.25,.05,1.)}


def case_request(case):
    kind, radius=case['waveform'],case['range_m']
    if kind=='pulsed_lfm':
        w={'kind':kind,'carrier_hz':10e9,'bandwidth_hz':5e6,'sample_rate_hz':10e6,'pulse_width_s':4e-6,
           'slot_period_s':100e-6,'adc_start_s':2*radius/C0-3e-6,'samples':128,'loops':32}
    else:
        start=2*radius/C0+5e-6
        w={'kind':kind,'start_hz':9.995e9,'slope_hz_s':1e11,'sample_rate_hz':4e6,'ramp_s':start+64e-6+5e-6,
           'slot_period_s':150e-6,'adc_start_s':start,'samples':256,'loops':32}
    platform=Platform(boresight_ned=POSES[case['pose']],beam_half_angle_deg=60)
    basis=sensor_basis(platform)
    direction=scan_directions([case['azimuth_deg']],[case['elevation_deg']])[0,0]@basis.T
    speed=case['speed_mps'];mode=case['motion']
    pv=np.zeros(3);tv=np.zeros(3)
    if mode=='target_moves':tv=-speed*direction
    elif mode=='radar_moves':pv=speed*direction
    elif mode=='both_move':pv=.4*speed*direction;tv=-.6*speed*direction
    elif mode=='same_velocity':pv=speed*direction;tv=pv.copy()
    elif mode=='transverse':tv=speed*basis[:,1]
    platform.velocity_ned_mps=tuple(pv)
    position=np.array(platform.position_ned_m)+radius*direction
    layout,ntx,nrx=LAYOUTS[case['array']]
    return Simulation.model_validate({'name':case['id'],'waveform':w,'platform':platform.model_dump(),
        'array':{'layout':layout,'num_tx':ntx,'num_rx':nrx},
        'scene':{'objects':[{'kind':'point','id':'target','position_ned_m':position.tolist(),'velocity_ned_mps':tv.tolist()}]},
        'processing':{'range_limits_m':[radius-100,radius+100],
                      'azimuth_deg':list(np.arange(-32.,33.)) if nrx>1 else [0.],
                      'elevation_deg':list(np.arange(-16.,17.)) if layout=='ura' else [0.]}})


def run_case(case):
    request=case_request(case)
    frame=simulate_frame(request)
    p=frame.products
    r,d,e,a=np.unravel_index(np.argmax(abs(p.rad)),p.rad.shape)
    # Independent evaluator: no simulator truth/peak-search helper is called.
    t=frame.metadata['truth_epoch_s']
    target=request.scene.objects[0]
    delta=np.array(target.position_ned_m)-np.array(request.platform.position_ned_m)
    relative=np.array(target.velocity_ned_mps)-np.array(request.platform.velocity_ned_mps)
    delta+=relative*t
    truth_range=np.linalg.norm(delta)
    closing=-np.dot(relative,delta/truth_range)
    local=delta@sensor_basis(request.platform)
    true_az=np.rad2deg(np.arctan2(local[1],local[0]))
    true_el=np.rad2deg(np.arctan2(-local[2],np.hypot(local[0],local[1])))
    range_error=abs(p.range_m[r]-truth_range)
    velocity_error=abs(p.velocity_mps[d]-closing)
    az_error=abs(p.azimuth_deg[a]-true_az) if request.array.num_rx>1 else None
    el_error=abs(p.elevation_deg[e]-true_el) if request.array.layout=='ura' else None
    rstep=float(np.diff(p.range_m)[0]);vstep=p.metadata['native_velocity_cell_mps']
    passed=(range_error<=rstep and velocity_error<=vstep and (az_error is None or az_error<=1.) and (el_error is None or el_error<=1.))
    return {**case,'true_range_m':float(truth_range),'true_closing_mps':float(closing),
            'range_error_m':float(range_error),'velocity_error_mps':float(velocity_error),
            'range_error_bins':float(range_error/rstep),'velocity_error_bins':float(velocity_error/vstep),
            'azimuth_error_deg':az_error,'elevation_error_deg':el_error,'elapsed_s':frame.metadata['elapsed_s'],'passed':bool(passed),'error':''}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    cases=[]
    for waveform,pose,radius,speed,array in product(['pulsed_lfm','fmcw'],POSES,[1500.,5000.],[-8.,0.,8.],LAYOUTS):
        for az,el in product([0.] if array=='SISO' else [-25.,0.,25.],[-12.,0.,12.] if array=='MIMO_URA' else [0.]):
            cases.append({'id':f'case_{len(cases):04d}','waveform':waveform,'pose':pose,'range_m':radius,
                          'speed_mps':speed,'array':array,'azimuth_deg':az,'elevation_deg':el,'motion':'target_moves'})
    for waveform,mode,array in product(['pulsed_lfm','fmcw'],['radar_moves','both_move','same_velocity','transverse'],LAYOUTS):
        cases.append({'id':f'case_{len(cases):04d}','waveform':waveform,'pose':'oblique','range_m':5000.,
                      'speed_mps':8.,'array':array,'azimuth_deg':0. if array=='SISO' else 15.,
                      'elevation_deg':10. if array=='MIMO_URA' else 0.,'motion':mode})
    # Off-grid cases prevent a falsely precise result from grid-aligned angles.
    for waveform,pose,az,el in product(['pulsed_lfm','fmcw'],['nadir','oblique'],[-24.6,.37,25.3],[-11.7,.45,11.2]):
        cases.append({'id':f'case_{len(cases):04d}','waveform':waveform,'pose':pose,'range_m':5000.37,
                      'speed_mps':3.7,'array':'MIMO_URA','azimuth_deg':az,'elevation_deg':el,'motion':'target_moves'})
    write_json(args.output/'selection.json',{'cases':cases,'is_real':False,'selection_before_scoring':True,
        'acceptance':'one native range/velocity cell, one degree angular grid; low range migration, unambiguous Doppler',
        'not_independent_real_scenes':True})
    project=Path(__file__).resolve().parents[1]
    files=[*project.glob('backend/radar_platform/unified/*.py'),Path(__file__).resolve()]
    with zipfile.ZipFile(args.output/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for f in files:archive.write(f,f.relative_to(project))
    write_json(args.output/'source_sha256.json',{str(f.relative_to(project)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    rows=[]
    for index,case in enumerate(cases):
        try:row=run_case(case)
        except Exception as exc:row={**case,'passed':False,'error':str(exc)}
        rows.append(row)
        if index%50==0:print('analytic cases',index,'/',len(cases),'failures',sum(not r['passed'] for r in rows),flush=True)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (args.output/'cases.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    summary={'cases':len(rows),'passed':sum(r['passed'] for r in rows),'failed_cases':[r for r in rows if not r['passed']],
             'by_waveform':{w:{'cases':sum(r['waveform']==w for r in rows),'passed':sum(r['passed'] for r in rows if r['waveform']==w)} for w in ['pulsed_lfm','fmcw']},
             'scope':'ideal analytic checks; no measured scattering fidelity claim',
             'max_error':{key:max(r[key] for r in rows if r.get(key) is not None) for key in ['range_error_bins','velocity_error_bins','azimuth_error_deg','elevation_error_deg']}}
    write_json(args.output/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    if summary['passed']!=len(rows):raise SystemExit(1)


if __name__=='__main__':main()
