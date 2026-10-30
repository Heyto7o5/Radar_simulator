#!/usr/bin/env python3
"""Score frozen surface/noise baselines. Never refit on temporal-check frames."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import binary_dilation
from radar_platform.models import C0
from radar_platform.dart_dataset import load_dart_frame
from radar_platform.dart_processing import rover_magnitude,published_to_closing_axis
from radar_platform.unified.schema import Simulation
from radar_platform.unified.scene import compile_scene
from radar_platform.unified.trajectory import TrajectoryEvaluator
from dart_batch_metrics import fit_power_gain,db_power,mae_db,complex_white_noise

PROJECT=Path(__file__).resolve().parents[1]
DATA=Path('/nas/datasets/hyt/datasets/radar_open/dart_lab')
OUT=PROJECT/'docs/validation_diagnostics/2026-10-15/dart/batch_v1'
SEEDS=[11,23,37]

def load(index):
    prefix=f'surface_sampled_motion_smooth_beam_batch_v1_f{index:04d}'
    request=Simulation.model_validate_json((OUT/f'{prefix}_request.json').read_text())
    status=json.loads((OUT/f'{prefix}_status.json').read_text())
    with np.load(status['file'],allow_pickle=False) as f:adc=f['adc']
    frame=load_dart_frame(DATA/'extracted/lab-1',index)
    velocity,real=published_to_closing_axis(frame.published_velocity_mps,frame.rd_power)
    return request,adc,frame.range_m,velocity,real

def align(adc,request,ranges,velocity):
    magnitude=rover_magnitude(adc);power=np.mean(magnitude**2,axis=2)
    w=request.waveform
    native_r=np.arange(w.samples)*w.sample_rate_hz/w.samples*C0/(2*w.slope_hz_s)
    raw_v=np.fft.fftshift(np.fft.fftfreq(w.loops,request.array.loop_period_s))*w.wavelength_m/2
    native_v,power=published_to_closing_axis(raw_v,power)
    rr,vv=np.meshgrid(ranges,velocity,indexing='ij')
    return RegularGridInterpolator((native_r,native_v),power,bounds_error=False,fill_value=np.nan)(np.stack([rr,vv],axis=-1))

def masks(request,ranges,velocity,sim):
    scene=compile_scene(request.scene)
    times=np.linspace(0,(request.waveform.loops-1)*request.array.loop_period_s,17)
    positions,speeds,bases,_=TrajectoryEvaluator(request.platform.trajectory).at(times)
    delta=scene.positions_ned_m[:,None,:]-positions[None,:,:]
    radius=np.linalg.norm(delta,axis=-1)
    unit=delta/radius[:,:,None];closing=np.sum(unit*speeds[None,:,:],axis=-1)
    lit=np.sum(unit*bases[None,:,:,0],axis=-1)>=np.cos(np.deg2rad(request.platform.beam_half_angle_deg))
    ri=np.rint((radius-ranges[0])/(ranges[1]-ranges[0])).astype(int)
    vi=np.rint((closing-velocity[0])/(velocity[1]-velocity[0])).astype(int)
    inside=lit&(ri>=0)&(ri<len(ranges))&(vi>=0)&(vi<len(velocity))
    geometry=np.zeros(sim.shape,bool);geometry[ri[inside],vi[inside]]=True
    geometry=binary_dilation(geometry,structure=np.ones((5,7),bool))
    valid=np.isfinite(sim)
    primary=valid&(abs(velocity)[None,:]>1.01*abs(velocity[1]-velocity[0]))
    speed_limit=float(np.linalg.norm(speeds,axis=1).max())+.05
    off=primary&(abs(velocity)[None,:]>speed_limit)
    return {'full':valid,'primary':primary,'geometry':primary&geometry,'off_kinematic':off},speed_limit

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args()
    manifest=json.loads((OUT/'manifest.json').read_text())
    protocol={'version':1,'calibration_valid_indices':[r['valid_frame_index'] for r in manifest['records'] if r['role']=='calibration'],
              'temporal_check_valid_indices':[r['valid_frame_index'] for r in manifest['records'] if r['role']!='calibration'],
              'gain':'one ratio of summed real/simulated linear power pooled across calibration frames, excluding native zero +/-1 Doppler bins',
              'reference':'99.9th percentile of calibration real power on primary score support, fixed across all frames/models',
              'floor_db':-60,'noise_floor':'median measured calibration power outside stationary-scene kinematic support (max sampled platform speed +0.05m/s)',
              'noise_variance':'floor/(frozen_gain * 8 * sum(range_Hann^2) * sum(Doppler_Hann^2))',
              'noise_seed_list':SEEDS,'noise_realization':'SeedSequence([seed,valid_frame_index]); complex circular white noise added before FFT',
              'geometry_region':'frozen surface centers projected over 17 trajectory samples, dilated +/-2 range and +/-3 Doppler bins; NOT independent target labels',
              'primary_score':'dB MAE on shared physical grid excluding three center Doppler bins; also report full/geometry/off-kinematic MAE',
              'models':['surface','noise_only','surface_plus_noise'],
              'independent_scenes':1,'independent_holdout_scenes':0,
              'limitations':['partial geometry','upstream center-bin subtraction not invertible','amplitude scale is relative nuisance calibration, not RCS truth','off-kinematic floor may include sidelobes or interference; not physical noise-figure calibration','multiple seeds and frames are not independent scenes']}
    protocol_path=OUT/'evaluation_protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise ValueError('evaluation protocol is frozen')
    protocol_path.write_text(json.dumps(protocol,indent=2))
    if args.freeze_only:print('evaluation protocol frozen before scoring',flush=True);return
    state=json.loads((OUT/'state.json').read_text())
    if state['status']!='complete':raise RuntimeError('batch not complete; no partial score or fit produced')
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1024**2),b''):h.update(block)
        return h.hexdigest()
    for name,expected in manifest['hashes'].items():
        if digest(PROJECT/name)!=expected:raise ValueError('frozen simulation code changed: '+name)
    for name,expected in manifest['input_sha256'].items():
        if digest(DATA/name)!=expected:raise ValueError('frozen geometry/trajectory input changed: '+name)
    cal=[]
    for index in protocol['calibration_valid_indices']:
        request,adc,r,v,real=load(index);sim=align(adc,request,r,v);region,limit=masks(request,r,v,sim)
        cal.append((real,sim,region))
    gain=fit_power_gain(np.stack([x[0] for x in cal]),np.stack([x[1] for x in cal]),np.stack([x[2]['primary'] for x in cal]))
    reference=float(np.quantile(np.concatenate([x[0][x[2]['primary']] for x in cal]),.999))
    outside=np.concatenate([x[0][x[2]['off_kinematic']] for x in cal])
    if len(outside)==0:raise ValueError('no calibration noise-floor support')
    floor=float(np.median(outside))
    variance=floor/(gain*8*np.sum(np.hanning(512)**2)*np.sum(np.hanning(256)**2))
    calibration={'gain':gain,'shared_power_reference':reference,'measured_floor_proxy':floor,'adc_noise_variance':variance,
                 'calibration_frames':protocol['calibration_valid_indices'],'numpy_version':np.__version__,'noise_bit_generator':'PCG64',
                 'no_testing_data_used_for_fit':True}
    (OUT/'frozen_calibration.json').write_text(json.dumps(calibration,indent=2))
    cache=DATA/'processed/batch_v1/evaluation';cache.mkdir(parents=True,exist_ok=True)
    rows=[]
    for index in protocol['temporal_check_valid_indices']:
        request,adc,r,v,real=load(index);surface=align(adc,request,r,v);region,limit=masks(request,r,v,surface)
        def score(name,seed,power):
            row={'valid_frame_index':index,'model':name,'seed':seed,'unsupported_bins':int((~region['full']).sum()),'kinematic_limit_mps':limit}
            for label,mask in region.items():
                row[f'{label}_bins']=int(mask.sum())
                row[f'{label}_mae_db']=mae_db(real,power,reference,mask) if mask.any() else None
            rows.append(row)
        score('surface',-1,gain*surface)
        products={'real_power':real,'surface_power':gain*surface,'range_m':r,'closing_velocity_mps':v,
                  'primary_mask':region['primary'],'geometry_mask':region['geometry']}
        for seed in SEEDS:
            noise=complex_white_noise(adc.shape,variance,np.random.SeedSequence([seed,index]))
            for name,data in [('noise_only',noise),('surface_plus_noise',adc+noise)]:
                power=gain*align(data,request,r,v);score(name,seed,power)
                products[f'{name}_seed_{seed}']=power
        np.savez_compressed(cache/f'f{index:04d}_power_products.npz',**products)
        print('scored frame',index,flush=True)
    with (OUT/'per_frame_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'calibration':calibration,'scene_count':1,'temporal_check_frames':len(protocol['temporal_check_valid_indices']),
             'seed_repeats_are_not_additional_scenes':True,'independent_holdout_scenes':0,'models':{}}
    for model in protocol['models']:
        selected=[row for row in rows if row['model']==model];metrics={}
        for region_name in ['full','primary','geometry','off_kinematic']:
            values=[]
            for index in protocol['temporal_check_valid_indices']:
                numbers=[row[f'{region_name}_mae_db'] for row in selected if row['valid_frame_index']==index and row[f'{region_name}_mae_db'] is not None]
                if numbers:values.append(float(np.mean(numbers)))
            metrics[region_name]={'frames':len(values),'mean_mae_db':float(np.mean(values)) if values else None,
                                  'median_mae_db':float(np.median(values)) if values else None,'p90_mae_db':float(np.quantile(values,.9)) if values else None}
        summary['models'][model]=metrics
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
