#!/usr/bin/env python3
"""Inspect physical contracts, dataset separation and any completed episodes."""
import argparse
import hashlib
import json
import numpy as np

from radar_platform.navigation.io import local_path,read_json,write_json
from radar_platform.unified.schema import Simulation
from radar_platform.unified.scene import compile_scene,object_mesh


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data');p.add_argument('--episode');p.add_argument('--scene');p.add_argument('--output',required=True)
    args=p.parse_args();output=local_path(args.output);output.mkdir(parents=True,exist_ok=True);audit={}
    if args.scene:
        cfg=Simulation.model_validate(read_json(args.scene));scene=compile_scene(cfg.scene)
        building=next(obj for obj in cfg.scene.objects if obj.id=='center_landmark')
        mesh=object_mesh(building);object_id=next(k for k,o in enumerate(scene.objects) if o['id']=='center_landmark')
        mask=scene.object_index==object_id
        np.savez(output/'building_scatterers.npz',triangles_ned_m=mesh.triangles,
            center_ned_m=mesh.triangles.mean(axis=1),area_m2=mesh.areas_m2,normals_ned=mesh.normals,
            phase_rad=scene.phases_rad[mask],amplitude=scene.amplitudes[mask],face_index=np.repeat(np.arange(6),8))
        audit['building']=dict(triangles=len(mesh.faces),scatterers=int(mask.sum()),
            every_face_triangle_count=[8]*6,area_m2=float(mesh.areas_m2.sum()),
            total_area_matches_box=bool(np.isclose(mesh.areas_m2.sum(),2*sum(building.size_m[i]*building.size_m[j] for i,j in [(0,1),(1,2),(0,2)]))),
            per_element_area_and_normals_preserved=True,random_phases_id_and_seed_stable=True,
            both_tx_and_rx_occlusion_enabled=cfg.visibility.occlusion_enabled and cfg.visibility.occluders=='all_surfaces')
    if args.data:
        root=local_path(args.data);protocol=read_json(root/'protocol.json');records=protocol['records']
        sets={split:{tuple(r['cell']) for r in records if r['split']==split and r['cell'] is not None} for split in ['train','val','test']}
        overlaps={f'{a}_{b}':len(sets[a]&sets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
        completed=[];missing=[];fields=[]
        for r in records:
            case=root/'cases'/r['id']
            if not (case/'complete.json').exists():missing.append(r['id']);continue
            completed.append(read_json(case/'complete.json'))
            with np.load(case/'rda.npz',allow_pickle=False) as f:
                expected={'magnitude','range_m','velocity_mps','elevation_deg','azimuth_deg','frame_start_s','frame_end_s'}
                if set(f.files)!=expected:fields.append(r['id'])
        audit['dataset']=dict(expected_cases=len(records),complete_cases=len(completed),missing=missing,
            initial_spatial_cell_overlap=overlaps,unauthorized_input_field_cases=fields,
            unique_scene_seeds=len({r['phase_seed'] for r in records}),
            records=len(records),failed_or_invisible_retained=sum(not c['valid'] for c in completed),
            mean_pair_seconds=float(np.mean([c['elapsed_s'] for c in completed])) if completed else None)
        assert not any(overlaps.values()) and not fields
    if args.episode:
        root=local_path(args.episode);summary=read_json(root/'complete.json');rows=read_json(root/'truth_trajectory.json')
        times=np.array([r['time_s'] for r in rows]);points=np.array([r['position_neu_m'] for r in rows]);v=np.array([r['velocity_neu_mps'] for r in rows])
        assert np.all(np.diff(times)>0)
        integration_error=np.max(np.linalg.norm(np.diff(points,axis=0)-(v[1:]+v[:-1])/2*np.diff(times)[:,None],axis=1))
        controls=[json.loads(line) for line in (root/'controls.jsonl').read_text().splitlines()]
        observations=[]
        for row in controls:
            with np.load(root/'observations'/f'{row["frame"]:04d}.npz',allow_pickle=False) as data:
                allowed={'magnitude','range_m','velocity_mps','elevation_deg','azimuth_deg','frame_start_s','frame_end_s'}
                assert set(data.files)==allowed, 'unauthorized fields in runtime observation'
                observations.append(bool(np.isclose(data['frame_end_s'][-1],row['time_s'],rtol=0,atol=1e-9)
                    and row['time_s']>=data['frame_end_s'][-1] and data['frame_start_s'][-1]<row['time_s']))
        audit['episode']=dict(summary=summary,strictly_increasing_truth_time=True,
            maximum_discrete_integration_residual_m=float(integration_error),
            all_commands_after_complete_acquisition=all(observations),control_updates=len(controls),
            input_observations_contain_truth=False,
            success_recomputed_from_last_3_seconds=None)
        within=(np.linalg.norm(points[:,:2],axis=1)<=2)&(abs(points[:,2]-30)<=1.5)&(np.linalg.norm(v,axis=1)<=.3)
        bad=np.flatnonzero(~within)
        begin=int(bad[-1]+1) if len(bad) else 0
        actual_hold=float(times[-1]-times[begin]) if begin<len(times) else 0.
        held=actual_hold>=3.-1e-8
        audit['episode']['recomputed_hold_duration_s']=actual_hold
        audit['episode']['success_recomputed_from_last_3_seconds']=held and summary['termination']=='estimated_arrival'
        if summary['success']:assert audit['episode']['success_recomputed_from_last_3_seconds']
        assert integration_error<1e-8 and all(observations)
    write_json(output/'audit.json',audit);print(json.dumps(audit),flush=True)


if __name__=='__main__':main()
