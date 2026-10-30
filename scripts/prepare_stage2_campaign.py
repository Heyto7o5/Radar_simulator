#!/usr/bin/env python3
"""Freeze controlled real-geometry experiments before evaluating their RD.

Real inputs: DSM and public classes. NRCS constants below are explicitly
counterfactual controls, not an inferred material model or measured truth.
"""
import argparse,copy,hashlib,json,os
from pathlib import Path
import numpy as np
from radar_platform.unified.schema import Simulation
from radar_platform.unified.preflight import preflight
from radar_platform.worldcover import CLASSES

ROOT=Path(__file__).resolve().parents[1]
VARIANTS=['uniform_0','hamming_0','hamming_8','tree_plus6','water_minus6']


def configuration(site,variant,seed,root):
    original=ROOT/f'.runtime/real_dem_20261022/{site}_30m_10ghz.json'
    raw=json.loads(original.read_text())
    classes=root/f'worldcover_2021/processed/fixed_dem_cells_20261022/{site}_30m_cells.npz'
    with np.load(classes,allow_pickle=False) as f:
        codes=f['dominant_code'];metadata=json.loads(str(f['metadata_json']))
    labels=np.array([[CLASSES[int(code)] for code in row] for row in codes])
    tables=[]
    for code in np.unique(codes):
        label=CLASSES[int(code)];db=-20.
        if variant=='tree_plus6' and code==10:db+=6
        if variant=='water_minus6' and code==80:db-=6
        tables.append(dict(land_cover=label,polarization='vv',frequencies_hz=[9e9,11e9],incidence_deg=[0,30,60,90],
            sigma0_db=[[db]*4,[db]*4],evidence='assumed',
            source='Counterfactual sensitivity control: common -20 dB baseline, optional +/-6 dB for one observed class; not measured NRCS.'))
    terrain=raw['scene']['terrain'];terrain['relative_reflectivity']=1.
    terrain['land_cover_map']=labels.tolist()
    terrain['land_scattering']=dict(tables=tables,polarization='vv',land_cover=tables[0]['land_cover'])
    raw['scene']['seed']=seed
    wavelength=299792458/raw['waveform']['carrier_hz'];positions=np.zeros((16,3))
    positions[:,1]=(np.arange(16)-7.5)*wavelength/2
    raw['radiation']=dict(tx_element=dict(kind='cosine_power',power_exponent=1),
        rx_element=dict(kind='cosine_power',power_exponent=1),replace_legacy_cone=True,
        transmit_aperture=dict(positions_local_m=positions.tolist(),
            amplitudes=(np.ones(16) if variant=='uniform_0' else np.hamming(16)).tolist(),
            steering_azimuth_deg=8. if variant=='hamming_8' else 0.))
    raw['visibility'].update(accelerator='bvh',batch_bvh=True,max_ray_triangle_tests=500_000_000)
    raw['name']=f'{site}/{variant}/seed{seed}: real inputs, assumed NRCS sensitivity'
    raw['input_provenance']=[x for x in raw['input_provenance'] if not x.startswith('Uniform relative scatter')]
    raw['input_provenance'] += [
        'WorldCover 2021 classification is remotely inferred, not locally verified material truth; epochs differ from DSM.',
        f'Class grid SHA256={hashlib.sha256(classes.read_bytes()).hexdigest()}; mixed cells below 80% purity={metadata["mixed_cells_below_80_percent"]}/{metadata["total_cells"]}.',
        'Every class starts at the same assumed -20 dB NRCS; tree/water changes are controlled sensitivity probes, not fitted material values.',
        'Across variants all geometry, radar sampling, trajectory and seed remain fixed; only declared radiation or NRCS factors change.',
        '16 coherent Tx elements, one logical Tx and one Rx. This experiment does not claim receive-angle observability.',
        'All mathematical sidelobes retained inside the finite 5.76 km scene; clutter beyond the scene is absent.'
    ]
    return Simulation.model_validate(raw)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--benchmark-only',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(os.environ.get('RADAR_DATA_ROOT','/nas/datasets/hyt/datasets/radar_open'))
    rows=[]
    for site in (['alpine'] if args.benchmark_only else ['alpine','valley','foothill']):
        for seed in ([11] if args.benchmark_only else [11,23,37]):
            for variant in (['uniform_0'] if args.benchmark_only else VARIANTS):
                config=configuration(site,variant,seed,root);_,estimate=preflight(config)
                key=f'{site}_{variant}_s{seed}';path=args.output/(key+'.json')
                path.write_text(config.model_dump_json(indent=2)+'\n')
                rows.append(dict(case=key,site=site,seed=seed,variant=variant,config=path.name,
                                 config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),work=estimate['sample_scatter_channel_work_per_frame']))
    plan={'purpose':'controlled functional comparison, not measured-data fitting or validation',
        'independent_real_sites':len({r['site'] for r in rows}),'seeds_are_not_independent_scenes':True,
        'parameters_changed':'only aperture taper/pointing or one class NRCS; paired seed and all other acquisition parameters fixed',
        'cases':rows,'selection':'all fixed sites and cases retained; no outcome-based selection'}
    (args.output/'plan.json').write_text(json.dumps(plan,indent=2))
    print(json.dumps({'cases':len(rows),'plan':str(args.output/'plan.json')}),flush=True)


if __name__=='__main__':main()
