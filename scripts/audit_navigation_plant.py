#!/usr/bin/env python3
"""Independently check commanded motion and the poses actually sent to the radar."""
import argparse
import bisect
import json
import math
import numpy as np
from scipy.spatial.transform import Rotation

from radar_platform.navigation.io import local_path,read_json,write_json,digest


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=local_path(args.run)
    if not (root/'complete.json').exists():raise ValueError('completed episode required')
    protocol=read_json(root/'protocol.json');spec=protocol['spec'];limits=spec['limits']
    trace=read_json(root/'truth_trajectory.json');rows=[json.loads(x) for x in (root/'controls.jsonl').read_text().splitlines() if x]
    times=np.array([r['time_s'] for r in trace]);positions=np.array([r['position_neu_m'] for r in trace])
    velocities=np.array([r['velocity_neu_mps'] for r in trace]);yaw=np.array([r['yaw_rad'] for r in trace])
    np.testing.assert_allclose(positions[0],spec['position_neu_m'],atol=1e-12,rtol=0)
    np.testing.assert_allclose(velocities[0],spec['velocity_neu_mps'],atol=1e-12,rtol=0)
    assert abs(yaw[0]-spec['yaw_rad'])<1e-12 and np.all(np.diff(times)>0)
    command_times=[r['time_s'] for r in rows]
    errors={'position_m':0.,'velocity_mps':0.,'yaw_rad':0.}
    for k in range(1,len(trace)):
        dt=times[k]-times[k-1];index=bisect.bisect_right(command_times,times[k-1]+1e-12)-1
        command=dict(acceleration_body_mps2=[0.,0.,0.],yaw_rate_rad_s=0.) if index<0 else rows[index]['command']
        a=np.array(command['acceleration_body_mps2'],float);magnitude=np.linalg.norm(a[:2])
        if magnitude>limits['horizontal_acceleration_mps2']:a[:2]*=limits['horizontal_acceleration_mps2']/magnitude
        a[2]=np.clip(a[2],-limits['vertical_acceleration_mps2'],limits['vertical_acceleration_mps2'])
        omega=float(np.clip(command['yaw_rate_rad_s'],-math.radians(limits['yaw_rate_deg_s']),math.radians(limits['yaw_rate_deg_s'])))
        midpoint=yaw[k-1]+omega*dt/2;c,s=math.cos(midpoint),math.sin(midpoint)
        acceleration=np.array([c*a[0]-s*a[1],s*a[0]+c*a[1],a[2]])
        next_velocity=velocities[k-1]+acceleration*dt
        cap=max(limits['horizontal_speed_mps'],np.linalg.norm(velocities[k-1,:2]));speed=np.linalg.norm(next_velocity[:2])
        if speed>cap:next_velocity[:2]*=cap/speed
        zcap=max(limits['vertical_speed_mps'],abs(velocities[k-1,2]))
        next_velocity[2]=np.clip(next_velocity[2],-zcap,zcap)
        next_position=positions[k-1]+(velocities[k-1]+next_velocity)*dt/2
        errors['position_m']=max(errors['position_m'],float(np.max(abs(next_position-positions[k]))))
        errors['velocity_mps']=max(errors['velocity_mps'],float(np.max(abs(next_velocity-velocities[k]))))
        errors['yaw_rad']=max(errors['yaw_rad'],abs(yaw[k-1]+omega*dt-yaw[k]))
    assert max(errors.values())<1e-9,'recorded trajectory does not follow issued commands'
    dt=np.diff(times);acc=np.diff(velocities,axis=0)/dt[:,None];omega=np.diff(yaw)/dt
    assert np.max(np.linalg.norm(acc[:,:2],axis=1))<=limits['horizontal_acceleration_mps2']+1e-8
    assert np.max(abs(acc[:,2]))<=limits['vertical_acceleration_mps2']+1e-8
    assert np.max(abs(omega))<=math.radians(limits['yaw_rate_deg_s'])+1e-8
    radar_samples=0;max_pose_error=0.;scene_hashes=set();noise_seeds=[];physical_frames=0
    for path in sorted((root/'truth').glob('*.json')):
        saved=read_json(path);trajectory=saved['request']['platform']['trajectory']
        scene_hashes.add(digest(saved['request']['scene']));noise_seeds.append(saved['request']['receiver_noise']['seed'])
        visibility=saved['request']['visibility']
        assert visibility['occlusion_enabled'] and visibility['occluders']=='all_surfaces'
        assert saved['diagnostic']['visibility']['occlusion_evaluated']
        physical_frames+=1
        for t,pn,vn,q in zip(trajectory['time_s'],trajectory['position_ned_m'],trajectory['velocity_ned_mps'],trajectory['quaternion_xyzw']):
            k=int(np.argmin(abs(times-t)))
            assert abs(times[k]-t)<1e-10
            pe=float(np.max(abs(np.asarray(pn)*[1,1,-1]-positions[k])))
            ve=float(np.max(abs(np.asarray(vn)*[1,1,-1]-velocities[k])))
            c,s=math.cos(yaw[k]),math.sin(yaw[k]);expected=np.array([[0,-s,-c],[0,c,-s],[1,0,0]])
            re=float(np.max(abs(Rotation.from_quat(q).as_matrix()-expected)))
            max_pose_error=max(max_pose_error,pe,ve,re);radar_samples+=1
    assert max_pose_error<1e-10,'radar acquisition used a different platform trajectory'
    assert len(scene_hashes)==1,'scene or material phase seed changed during flight'
    assert len(set(noise_seeds))==len(noise_seeds),'receiver noise seed was reused'
    assert len(list((root/'frames').glob('*.npz')))==physical_frames
    result=dict(integration_samples=len(trace),control_updates=len(rows),
        independent_command_to_motion_residual=errors,radar_trajectory_samples_checked=radar_samples,
        radar_pose_link_max_difference=max_pose_error,all_acceleration_and_turn_bounds_satisfied=True,
        physical_radar_frames=physical_frames,scene_configuration_constant=True,independent_noise_seeds=len(set(noise_seeds)),
        all_surface_occlusion_enabled_every_frame=True,
        scope='Independent equations use declared initial state and recorded commands; no dynamics.propagate call. Radar request samples are checked against the actual trajectory.')
    write_json(args.output,result);print(json.dumps(result))


if __name__=='__main__':main()
