"""DART geometry/pose/configuration -> common scene. No measured RD or synthesis."""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from .dart_dataset import FLU_TO_NED
from .unified.schema import Simulation


def prepare_scene(geometry_path, trajectory_path, radar_configuration_path, *,
                  valid_frame_index=0, sampled_trajectory=True, smooth_beam=True):
    geometry_path,trajectory_path,radar_configuration_path=map(Path,
        [geometry_path,trajectory_path,radar_configuration_path])
    audit=json.loads(radar_configuration_path.read_text())
    with np.load(geometry_path,allow_pickle=False) as data:
        vertices=data['vertices_flu_m']@FLU_TO_NED.T;faces=data['faces']
    wave=audit['waveform'];array=audit['array_timing_candidate']
    period=array.get('loop_period_s') or array['num_tx']*wave['slot_period_s']
    midpoint=(wave['loops']-1)*period/2
    with h5py.File(trajectory_path) as data:
        valid=np.flatnonzero(data['valid'][:])
        if not 0<=valid_frame_index<len(valid):raise ValueError('valid frame index out of bounds')
        index=int(valid[valid_frame_index])
        raw_index=int(np.flatnonzero(data['mask'][:])[index])
        pos=FLU_TO_NED@data['pos'][index];vel=FLU_TO_NED@data['vel'][index]
        rot=FLU_TO_NED@data['rot'][index]
        trajectory=None
        if sampled_trajectory:
            times=data['t'][:];center=times[index]
            select=np.flatnonzero(abs(times-center)<max(.3,midpoint*2))
            bases=np.einsum('ij,njk,kl->nil',FLU_TO_NED,data['rot'][select],FLU_TO_NED)
            trajectory=dict(time_s=(times[select]-center+midpoint).tolist(),
                position_ned_m=(data['pos'][select]@FLU_TO_NED.T).tolist(),
                velocity_ned_mps=(data['vel'][select]@FLU_TO_NED.T).tolist(),
                quaternion_xyzw=Rotation.from_matrix(bases).as_quat().tolist(),
                provenance='DART lab-1 published smoothed trajectory, FLU to NED/FRD; Hermite translation and SLERP attitude; not independent motion-capture truth')
    objects=[]
    for begin in range(0,len(faces),2048):
        part=faces[begin:begin+2048];used,inverse=np.unique(part,return_inverse=True)
        objects.append(dict(kind='mesh',id=f'lidar_surface_{begin}',vertices_local_m=vertices[used].tolist(),
            faces=inverse.reshape(-1,3).tolist(),scattering='diffuse',two_sided=True,relative_reflectivity=1.))
    platform=dict(position_ned_m=(pos-vel*midpoint).tolist(),velocity_ned_mps=vel.tolist(),
        boresight_ned=rot[:,0].tolist(),right_hint_ned=(-rot[:,1]).tolist(),beam_half_angle_deg=65)
    if trajectory is not None:platform['trajectory']=trajectory
    if smooth_beam:platform.update(beam_model='cosine_edge',beam_taper_deg=5.)
    notes=[
        f'DART valid frame {valid_frame_index}; trajectory index {index}; raw radar frame index {raw_index}.',
        'Inputs: fitted independent LiDAR surfaces and published smoothed poses; no measured RD used to generate scene.',
        'All surface reflectivities are relative one, two-sided diffuse cells; materials and absolute power uncalibrated.',
        'Radar timing and array are the supplied audited candidate, not a verified per-record instrument log.',
        'No environmental occlusion/multipath; geometry discretization convergence not established.'
    ]
    for source in [geometry_path,trajectory_path,radar_configuration_path]:
        notes.append(f'{source.name} SHA256: {hashlib.sha256(source.read_bytes()).hexdigest()}')
    return Simulation.model_validate(dict(name='DART prepared common scene; uncalibrated',input_provenance=notes,
        waveform=wave,array=array,platform=platform,
        scene=dict(objects=objects,seed=42,max_scatterers=8192 if len(faces)<=8192 else 32768,amplitude_reference_range_m=1.),
        processing=dict(range_limits_m=[0,5.5],azimuth_deg=[0.],elevation_deg=[0.],tdm_compensation=False),
        sequence=dict(frame_count=1)))
