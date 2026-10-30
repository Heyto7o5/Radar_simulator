"""Explicit binary product contracts; full complex channels are never overwritten by display scaling."""
import json
from pathlib import Path
import numpy as np


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def save_scene(root, request, scene):
    root.mkdir(parents=True, exist_ok=True)
    write_json(root/'request.json',request.model_dump(mode='json'))
    write_json(root/'scene.json',{**scene.metadata(),
        'environment_occlusion':request.visibility.occlusion_enabled,
        'visibility_model':request.visibility.model_dump(mode='json')})
    if scene.terrain is not None:
        write_json(root/'terrain.json',scene.terrain)
    np.savez_compressed(root/'scatterers.npz',position_ned_m=scene.positions_ned_m,
        velocity_ned_mps=scene.velocities_ned_mps,amplitude=scene.amplitudes,phase_rad=scene.phases_rad,
        normals_ned=scene.normals_ned,object_index=scene.object_index,scalar_po=scene.scalar_po,two_sided=scene.two_sided,
        land_cover_labels=np.asarray(scene.land_cover_labels if scene.land_cover_labels is not None else [],dtype=str))


def save_frame(root, frame, save_adc=True):
    directory=root/f'frame_{frame.metadata["frame_index"]:04d}'
    directory.mkdir(parents=True,exist_ok=False)
    p=frame.products
    data={'rd_channels':p.rd_channels,'rad':p.rad,'rd_power':p.rd_power,'ra_power':p.ra_power}
    if frame.components:data.update({k:v for k,v in frame.components.items() if save_adc or not k.endswith('_adc')})
    if save_adc:
        data['adc']=frame.adc
    manifest={}
    for name,array in data.items():
        dtype='<c8' if np.iscomplexobj(array) else '<f4'
        path=directory/(name+'.bin')
        np.asarray(array,dtype=dtype,order='C').tofile(path)
        manifest[name]={'file':path.name,'shape':list(array.shape),'dtype':dtype,'order':'C','bytes':path.stat().st_size}
    meta={**frame.metadata,'axes':{'range_m':p.range_m.tolist(),'closing_doppler_hz':p.closing_doppler_hz.tolist(),
                                 'velocity_mps':p.velocity_mps.tolist(),'azimuth_deg':p.azimuth_deg.tolist(),
                                 'elevation_deg':p.elevation_deg.tolist()},'products':manifest}
    write_json(directory/'metadata.json',meta)
    coverage=frame.metadata.get('visibility',{}).get('coverage')
    if coverage is not None:write_json(directory/'coverage.json',coverage)
    return meta
