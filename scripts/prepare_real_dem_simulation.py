#!/usr/bin/env python3
"""Real height-grid input -> ordinary platform config; no radar synthesis here."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from radar_platform.unified.schema import Simulation
from radar_platform.unified.preflight import preflight


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--patch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--off-nadir-deg',type=float,default=0)
    args=p.parse_args()
    if not 0<=args.off_nadir_deg<=20:raise ValueError('current declared geometry benchmark uses 0..20 degrees')
    if args.output.exists():raise FileExistsError('existing configuration preserved')
    with np.load(args.patch,allow_pickle=False) as f:
        n,e,h=f['north_m'],f['east_m'],f['elevation_up_m'];meta=json.loads(str(f['metadata_json']))
    offset=5000*np.tan(np.deg2rad(args.off_nadir_deg))
    config=Simulation.model_validate(dict(name='Real Copernicus DSM, 10 GHz ideal relative scattering',
        input_provenance=[f'Source: {meta["source_file"]}; SHA256={meta["source_sha256"]}',
            f'Origin lat/lon={meta["origin_latitude_deg"]},{meta["origin_longitude_deg"]}; EGM2008 height={meta["origin_orthometric_height_m"]} m',
            f'Local grid {meta["spacing_m"]} m, width {meta["width_m"]} m; no height scaling; DSM not bare-earth DTM.',
            'Source orthometric relief in local projected flat coordinates; curvature/geoid-gradient phase corrections not included.',
            'Uniform relative scatter coefficient and seeded phase are assumptions, not measured land-cover or RCS.',
            'Platform is 5000 m above patch-center elevation, not necessarily 5000 m AGL at its own horizontal location.',
            '1200 m/s platform motion may cause Doppler ambiguity at 10 GHz; single PRF, no de-aliasing.',
            f'Prepared patch SHA256={hashlib.sha256(args.patch.read_bytes()).hexdigest()}'],
        waveform=dict(kind='pulsed_lfm',carrier_hz=10e9,bandwidth_hz=20e6,sample_rate_hz=40e6,
                      pulse_width_s=5e-6,slot_period_s=100e-6,adc_start_s=28e-6,samples=1024,loops=32),
        platform=dict(position_ned_m=[-offset,0,-5000],velocity_ned_mps=[208.3778132,0,1181.7693036],
                      boresight_ned=[offset,0,5000],beam_half_angle_deg=4),
        array=dict(layout='siso',num_tx=1,num_rx=1),
        scene=dict(seed=42,max_scatterers=131072,amplitude_reference_range_m=5000,
                   terrain=dict(kind='dem',id='real_dsm',max_grid_vertices=65536,north_m=n.tolist(),east_m=e.tolist(),
                                elevation_up_m=h.tolist(),relative_reflectivity=.002)),
        visibility=dict(occlusion_enabled=True,report_coverage=True,occluders='terrain',accelerator='bvh'),
        processing=dict(range_limits_m=[4200,8000],azimuth_deg=[0.],elevation_deg=[0.]),
        sequence=dict(frame_count=1)))
    _,estimate=preflight(config)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:f.write(config.model_dump_json(indent=2)+'\n')
    print(json.dumps({'configuration':str(args.output),'scatterers':estimate['scatterers'],
                      'applicability':estimate['applicability']['status'],'simulation_performed':False}),flush=True)


if __name__=='__main__':main()
