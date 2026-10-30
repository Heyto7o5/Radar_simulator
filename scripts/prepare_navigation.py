#!/usr/bin/env python3
"""Prepare, inspect and benchmark the authorized real-DSM navigation scene."""
import argparse
import json
import time
import numpy as np

from radar_platform.navigation.scene import navigation_scene
from radar_platform.navigation.io import local_path, write_json
from radar_platform.navigation.dynamics import State, Command, Limits, propagate
from radar_platform.navigation.acquisition import RadarWorld
from radar_platform.unified.geometry import build_layout
from radar_platform.unified.scene import object_mesh
from radar_platform.unified.motion_check import motion_error_report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='.runtime/navigation_20261030/preparation')
    p.add_argument('--benchmark',action='store_true')
    args=p.parse_args(); root=local_path(args.output); root.mkdir(parents=True,exist_ok=True)
    cfg, provenance=navigation_scene()
    write_json(root/'scene.json',cfg.model_dump(mode='json'));write_json(root/'provenance.json',provenance)
    world=RadarWorld(cfg); layout=build_layout(cfg.array,cfg.waveform,cfg.platform)
    mesh=object_mesh(cfg.scene.objects[0]); centers=mesh.triangles.mean(axis=1)
    center=np.asarray(cfg.scene.objects[0].position_ned_m)
    audit=dict(triangles=len(mesh.faces),area_m2=float(mesh.areas_m2.sum()),
        outward_normals=bool(np.all(np.sum((centers-center)*mesh.normals,axis=1)>0)),
        virtual_array=layout.metadata(),preflight=world.estimate,
        scene_scatterers=len(world.scene.amplitudes),context=provenance['context_bounds_ne_m'],
        scattering_phases_sha256=__import__('hashlib').sha256(world.scene.phases_rad.tobytes()).hexdigest())
    write_json(root/'geometry_audit.json',audit)
    print(json.dumps(dict(triangles=audit['triangles'],scatterers=audit['scene_scatterers'],
                         virtual_positions=audit['virtual_array']['unique_virtual_positions'],preflight=world.estimate)),flush=True)
    if args.benchmark:
        yaw=np.arctan2(18,24)+np.deg2rad(60)
        state=State(0.,np.array([-24.,-18.,30.]),3*np.array([np.cos(yaw),np.sin(yaw),0.]),yaw)
        _, states=propagate(state,Command(np.array([-.3,.2,.1]),np.deg2rad(30)),world.duration_s)
        started=time.perf_counter()
        frame,diagnostic,request=world.acquire(states,0,save_directory=root/'benchmark',assessment=True)
        write_json(root/'benchmark.json',dict(elapsed_s=time.perf_counter()-started,
                   rda_shape=list(frame.magnitude.shape),rda_bytes=frame.magnitude.nbytes,diagnostic=diagnostic))
        write_json(root/'motion_check.json',motion_error_report(request,max_scatterers=128,max_slots=8))
        print(json.dumps(dict(elapsed_s=diagnostic['elapsed_s'],rda_shape=list(frame.magnitude.shape))),flush=True)


if __name__=='__main__':main()
