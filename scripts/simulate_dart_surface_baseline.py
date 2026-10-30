#!/usr/bin/env python3
"""First surface-chain baseline. Not a calibrated DART digital twin or final score."""
import json
import argparse
import time
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from radar_platform.dart_dataset import FLU_TO_NED
from radar_platform.unified.schema import Simulation
from radar_platform.unified.engine import simulate_frame

ROOT=Path('/nas/datasets/hyt/datasets/radar_open/dart_lab')
OUT=Path(__file__).resolve().parents[1]/'docs/validation_diagnostics/2026-10-15/dart'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--sampled-trajectory',action='store_true');parser.add_argument('--smooth-beam',action='store_true')
    parser.add_argument('--valid-frame-index',type=int,default=0);parser.add_argument('--batch-tag',default=None)
    parser.add_argument('--geometry',choices=['initial','trajectory_bounds'],default='initial');args=parser.parse_args()
    if args.geometry!='initial' and args.batch_tag is None:raise ValueError('expanded geometry requires a distinct batch tag to preserve baseline results')
    output=OUT
    if args.batch_tag is not None:
        if not args.batch_tag.replace('_','').isalnum():raise ValueError('batch tag must be alphanumeric or underscore')
        output=OUT/args.batch_tag;output.mkdir(parents=True,exist_ok=True)
    audit=json.loads((OUT/'radar_configuration_audit.json').read_text())
    geometry_file='lab-1_supported_planes_cell_0.25'+('' if args.geometry=='initial' else '_maxplanes_32_trajectory_bounds')+'.npz'
    with np.load(ROOT/'processed'/geometry_file,allow_pickle=False) as f:
        vertices=f['vertices_flu_m']@FLU_TO_NED.T;faces=f['faces']
    with h5py.File(ROOT/'extracted/lab-1/trajectory.h5') as f:
        valid_indices=np.flatnonzero(f['valid'][:])
        if not 0<=args.valid_frame_index<len(valid_indices):raise ValueError('valid frame index out of bounds')
        index=int(valid_indices[args.valid_frame_index]);raw_index=int(np.flatnonzero(f['mask'][:])[index])
        pos=FLU_TO_NED@f['pos'][index];vel=FLU_TO_NED@f['vel'][index];rot=FLU_TO_NED@f['rot'][index]
    objects=[]
    for begin in range(0,len(faces),2048):
        part=faces[begin:begin+2048];used,inv=np.unique(part,return_inverse=True)
        objects.append(dict(kind='mesh',id=f'lidar_surface_{begin}',vertices_local_m=vertices[used].tolist(),
                            faces=inv.reshape(-1,3).tolist(),scattering='diffuse',two_sided=True,
                            relative_reflectivity=1.))
    wave=audit['waveform'];cpi_mid=(wave['loops']-1)*.001/2
    request=Simulation.model_validate(dict(name='DART geometry-only surface baseline; uncalibrated',
        waveform=wave,array=audit['array_timing_candidate'],
        platform=dict(position_ned_m=(pos-vel*cpi_mid).tolist(),velocity_ned_mps=vel.tolist(),
                      boresight_ned=rot[:,0].tolist(),right_hint_ned=(-rot[:,1]).tolist(),beam_half_angle_deg=65),
        scene=dict(objects=objects,seed=42,max_scatterers=8192 if args.geometry=='initial' else 32768,amplitude_reference_range_m=1.),
        processing=dict(range_limits_m=[0,5.5],azimuth_deg=[0.],elevation_deg=[0.],tdm_compensation=False),
        sequence=dict(frame_count=1)))
    prefix='surface_sampled_motion' if args.sampled_trajectory else 'surface_baseline'
    if args.smooth_beam:prefix+='_smooth_beam'
    if args.batch_tag is not None:prefix+=f'_{args.batch_tag}_f{args.valid_frame_index:04d}'
    elif args.valid_frame_index!=0:prefix+=f'_f{args.valid_frame_index:04d}'
    if args.sampled_trajectory:
        with h5py.File(ROOT/'extracted/lab-1/trajectory.h5') as f:
            source_t=f['t'][:];center=source_t[index]
            indices=np.flatnonzero(abs(source_t-center)<.3)
            positions=f['pos'][indices]@FLU_TO_NED.T
            velocities=f['vel'][indices]@FLU_TO_NED.T
            bases=np.einsum('ij,njk,kl->nil',FLU_TO_NED,f['rot'][indices],FLU_TO_NED)
        trajectory=dict(time_s=(source_t[indices]-center+cpi_mid).tolist(),position_ned_m=positions.tolist(),
                        velocity_ned_mps=velocities.tolist(),quaternion_xyzw=Rotation.from_matrix(bases).as_quat().tolist(),
                        provenance='DART lab-1 published smoothed trajectory, FLU to NED/FRD; Hermite translation and SLERP attitude; not independent motion-capture truth')
        data=request.model_dump();data['platform']['trajectory']=trajectory
        request=Simulation.model_validate(data)
    if args.smooth_beam:
        data=request.model_dump();data['platform'].update(beam_model='cosine_edge',beam_taper_deg=5.)
        request=Simulation.model_validate(data)
    (output/f'{prefix}_request.json').write_text(request.model_dump_json(indent=2))
    print('synthesizing',len(faces),'surface cells with 2Tx4Rx',flush=True)
    start=time.perf_counter();frame=simulate_frame(request)
    target=ROOT/'processed'/f'lab-1_{prefix}.npz'
    np.savez_compressed(target,adc=frame.adc,rd_channels=frame.products.rd_channels,
                        range_m=frame.products.range_m,closing_velocity_mps=frame.products.velocity_mps)
    report={'status':'simulated_not_validated','elapsed_s':time.perf_counter()-start,'surface_triangles':len(faces),
            'trajectory_index':index,'raw_frame_index':raw_index,'valid_frame_index':args.valid_frame_index,'source_scene_count':1,'simulation_frames':1,
            'geometry_source':'independent LiDAR, fitted on geometry training cells',
            'geometry_file':geometry_file,'geometry_crop_mode':args.geometry,
            'scattering':'relative diffuse area cells, random seeded phase, two-sided',
            'material_calibration':False,'absolute_amplitude_calibration':False,
            'model_limitations':['partial scene geometry','25cm surface discretization, convergence not established',
                                 'no occlusion or environmental multipath','ideal ULA not verified physical element placement',
                                 'constant velocity and orientation within CPI','hard cone not measured antenna pattern',
                                 'real ADC phase convention not yet confirmed','real zero-Doppler correction not yet matched'],
            'file':str(target),'not_final_real_data_mae':True,'sampled_platform_trajectory':args.sampled_trajectory,
            'engine_motion':frame.metadata['platform_motion'],'engine_warnings':frame.metadata['warnings']}
    if args.sampled_trajectory:
        report['model_limitations'].remove('constant velocity and orientation within CPI')
        report['model_limitations'].append('published smoothed trajectory interpolated; target motion unchanged; no angular motion compensation')
    if args.smooth_beam:
        report['model_limitations'].remove('hard cone not measured antenna pattern')
        report['model_limitations'].append('5 degree cosine beam edge: controlled numerical model, not measured antenna pattern')
    report['beam_response']=frame.metadata['beam_response']
    (output/f'{prefix}_status.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
