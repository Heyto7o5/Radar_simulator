#!/usr/bin/env python3
"""Compare expanded world geometry on fixed v1 grids, masks and noise reference."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from radar_platform.dart_dataset import load_dart_frame
from radar_platform.dart_processing import published_to_closing_axis
from radar_platform.unified.schema import Simulation
from radar_platform.colored_adc_background import colored_adc_noise
from evaluate_dart_frozen_batch import align,DATA,PROJECT,OUT as BASE
from dart_batch_metrics import fit_power_gain,mae_db

OUT=BASE.parent/'batch_v2_world_geometry'
TAG='batch_v2_world_geometry'

def load(index):
    prefix=f'surface_sampled_motion_smooth_beam_{TAG}_f{index:04d}'
    request=Simulation.model_validate_json((OUT/f'{prefix}_request.json').read_text())
    status=json.loads((OUT/f'{prefix}_status.json').read_text())
    with np.load(status['file'],allow_pickle=False) as f:adc=f['adc']
    real=load_dart_frame(DATA/'extracted/lab-1',index)
    v,p=published_to_closing_axis(real.published_velocity_mps,real.rd_power)
    return request,adc,real.range_m,v,p

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--freeze-only',action='store_true');args=parser.parse_args()
    manifest=json.loads((OUT/'manifest.json').read_text())
    protocol={'version':1,'baseline_batch':'batch_v1','calibration_indices':[0,8,16],
              'evaluation_indices':[32,40,48,56,64,72,80,88],'seeds':[11,23,37],
              'amplitude_cases':['unchanged v1 gain','single refitted gain using only the same three calibration frames'],
              'reference_and_floor':'unchanged v1 reference; -60 dB floor',
              'noise':'same frozen colored PSD in MEASURED power units; ADC PSD scaled inversely with each gain to preserve noise power',
              'region_masks':'same saved v1 primary and geometry masks; no enlarged target region substituted in scoring',
              'fitting':'pooled calibration linear power ratio excluding same three central Doppler bins',
              'counterfactual_limits':['geometry fit domain, plane support and diffuse phase realization changed; not a pure one-parameter scattering ablation',
                                       'previously inspected frames: development comparison, not independent held-out validation'],
              'scene_count':1,'independent_holdout_scenes':0}
    p=OUT/'evaluation_protocol.json'
    if p.exists() and json.loads(p.read_text())!=protocol:raise ValueError('protocol already frozen')
    p.write_text(json.dumps(protocol,indent=2))
    if args.freeze_only:print('world-geometry evaluation protocol frozen',flush=True);return
    if json.loads((OUT/'state.json').read_text())['status']!='complete':raise RuntimeError('world batch incomplete; no partial score')
    for name,expected in manifest['hashes'].items():
        if hashlib.sha256((PROJECT/name).read_bytes()).hexdigest()!=expected:raise ValueError('simulation code changed during batch: '+name)
    for name,expected in manifest['input_sha256'].items():
        if hashlib.sha256((DATA/name).read_bytes()).hexdigest()!=expected:raise ValueError('geometry input changed during batch: '+name)
    calibration=json.loads((BASE/'frozen_calibration.json').read_text());old_gain=calibration['gain'];reference=calibration['shared_power_reference']
    colored=json.loads((BASE.parent/'colored_background_followup/frozen_colored_model.json').read_text())
    q=np.array(colored['spectral_power']);cal_real=[];cal_sim=[];cal_mask=[]
    for index in protocol['calibration_indices']:
        request,adc,r,v,real=load(index);sim=align(adc,request,r,v)
        mask=np.isfinite(sim)&(abs(v)[None,:]>1.01*abs(v[1]-v[0]))
        cal_real.append(real);cal_sim.append(sim);cal_mask.append(mask)
    refitted=fit_power_gain(np.stack(cal_real),np.stack(cal_sim),np.stack(cal_mask))
    fit={'old_gain':old_gain,'refitted_gain':refitted,'calibration_indices':protocol['calibration_indices'],
         'reference':reference,'noise_psd_in_observed_units_unchanged':True}
    (OUT/'frozen_calibration.json').write_text(json.dumps(fit,indent=2))
    cache=DATA/'processed'/TAG/'evaluation';cache.mkdir(parents=True,exist_ok=True);rows=[]
    for index in protocol['evaluation_indices']:
        request,adc,r,v,real=load(index)
        with np.load(DATA/'processed/batch_v1/evaluation'/f'f{index:04d}_power_products.npz',allow_pickle=False) as previous:
            if not np.allclose(previous['range_m'],r) or not np.allclose(previous['closing_velocity_mps'],v):raise ValueError('physical axes changed')
            regions={'primary':previous['primary_mask'],'geometry':previous['geometry_mask']}
        surface=align(adc,request,r,v);regions['full']=np.isfinite(surface)
        def score(name,seed,power):
            row={'valid_frame_index':index,'model':name,'seed':seed}
            for region,mask in regions.items():row[region+'_mae_db']=mae_db(real,power,reference,mask) if mask.any() else None
            rows.append(row)
        products={'real_power':real,'range_m':r,'velocity_mps':v,'primary_mask':regions['primary'],'geometry_mask':regions['geometry']}
        for case,gain in [('fixed',old_gain),('refitted',refitted)]:
            score('expanded_clean_'+case,-1,gain*surface)
            products['expanded_clean_'+case]=gain*surface
            scaled_q=q*old_gain/gain
            for seed in protocol['seeds']:
                noise=colored_adc_noise(adc.shape,scaled_q,np.random.SeedSequence([seed,index]))
                power=gain*align(adc+noise,request,r,v)
                score('expanded_colored_'+case,seed,power)
                if seed==protocol['seeds'][0]:products['expanded_colored_'+case]=power
        np.savez_compressed(cache/f'f{index:04d}.npz',**products)
        print('world geometry scored',index,flush=True)
    with (OUT/'per_frame_metrics.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary={'scene_count':1,'evaluation_frames':8,'independent_holdout_scenes':0,'calibration':fit,'models':{}}
    for name in sorted({row['model'] for row in rows}):
        summary['models'][name]={}
        for region in ['full','primary','geometry']:
            values=[np.mean([row[region+'_mae_db'] for row in rows if row['model']==name and row['valid_frame_index']==i]) for i in protocol['evaluation_indices']]
            summary['models'][name][region]={'mean_mae_db':float(np.mean(values)),'median_mae_db':float(np.median(values)),'p90_mae_db':float(np.quantile(values,.9))}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
