import numpy as np
import pytest

from radar_platform.navigation.dynamics import (State,Command,Limits,propagate,body_to_neu,
                                                radar_to_ned,sampled_trajectory)
from radar_platform.navigation.controller import Controller,Estimate
from radar_platform.navigation.observation import Observation,encode,FEATURE_SHAPE
from radar_platform.navigation.scene import navigation_scene, SOURCE
from radar_platform.navigation.io import local_path
from radar_platform.unified.geometry import build_layout
from radar_platform.unified.scene import compile_scene,object_mesh
from radar_platform.unified.trajectory import TrajectoryEvaluator


@pytest.mark.local_data
@pytest.mark.skipif(not local_path(SOURCE).is_file(), reason='requires explicitly prepared real DSM; no synthetic substitution')
def test_center_building_real_context_array_and_phase_persistence():
    cfg,meta=navigation_scene()
    mesh=object_mesh(cfg.scene.objects[0]);scene=compile_scene(cfg.scene)
    assert len(mesh.faces)==48
    assert np.all(mesh.areas_m2>0)
    center=np.array(cfg.scene.objects[0].position_ned_m)
    assert np.all(np.sum((mesh.triangles.mean(axis=1)-center)*mesh.normals,axis=1)>0)
    sizes=cfg.scene.objects[0].size_m
    assert mesh.areas_m2.sum()==pytest.approx(2*(sizes[0]*sizes[1]+sizes[1]*sizes[2]+sizes[0]*sizes[2]))
    assert meta['cover_counts']['cropland']<sum(meta['cover_counts'].values())
    assert meta['context_bounds_ne_m']==[[-210,-210],[210,210]]
    assert len(scene.positions_ned_m)==3576
    layout=build_layout(cfg.array,cfg.waveform,cfg.platform)
    assert layout.metadata()['unique_virtual_positions']==64
    assert layout.metadata()['virtual_aperture_rank']==2
    again=compile_scene(cfg.scene)
    np.testing.assert_array_equal(scene.phases_rad,again.phases_rad)
    assert cfg.visibility.occlusion_enabled and cfg.visibility.occluders=='all_surfaces'


def test_body_coordinates_at_four_headings_and_radar_basis():
    for yaw in np.arange(4)*np.pi/2:
        r=body_to_neu(yaw);b=radar_to_ned(yaw)
        np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-14)
        np.testing.assert_allclose(b.T@b,np.eye(3),atol=1e-14)
        assert np.linalg.det(b)==pytest.approx(1)
        np.testing.assert_allclose(b[:,0],[0,0,1])
        state=State(0,np.array([10.,0,30]),np.array([2.,0,0]),yaw)
        np.testing.assert_allclose(r@state.label()[:3],[-10,0,-30],atol=1e-12)
        np.testing.assert_allclose(r@state.label()[3:],[2,0,0],atol=1e-12)


def test_continuous_bounded_motion_and_sampled_cpi():
    state=State(0,np.array([-24.,-18,30]),np.array([3.,0,0]),0.)
    end,samples=propagate(state,Command(np.array([100.,100.,100.]),100.),.5)
    assert end.time_s==pytest.approx(.5)
    assert end.yaw_rad==pytest.approx(np.pi/8)
    velocities=np.array([s.velocity_neu_mps for s in samples])
    dt=np.diff([s.time_s for s in samples])
    a=np.diff(velocities,axis=0)/dt[:,None]
    assert np.max(np.linalg.norm(a[:,:2],axis=1))<=2.+1e-9
    assert np.max(abs(a[:,2]))<=1.+1e-9
    ev=TrajectoryEvaluator(sampled_trajectory(samples))
    position,velocity,_,_=ev.at(.5)
    np.testing.assert_allclose(position,end.position_neu_m*[1,1,-1])
    np.testing.assert_allclose(velocity,end.velocity_neu_mps*[1,1,-1])


def test_guidance_works_without_pose_or_heading_argument():
    controller=Controller()
    command,info=controller.step(Estimate(.3,np.array([-10.,8.,-35.]),np.zeros(3)))
    assert command.acceleration_body_mps2[0]<0<command.acceleration_body_mps2[1]
    assert command.acceleration_body_mps2[2]<0
    assert not info['estimated_arrival']
    for t in np.arange(0,3.1,.5):
        _,info=controller.step(Estimate(float(t),np.array([0.,0.,-30.]),np.zeros(3)))
    assert info['estimated_arrival']
    _,info=controller.step(Estimate(3.5,np.zeros(3),np.zeros(3),valid=False))
    assert not info['estimated_arrival'] and info['mode']=='reacquire'


def test_observation_contract_preprocessing_and_no_truth_fields(tmp_path):
    a=np.random.default_rng(4).uniform(.1,2,(2,10,8,5,5)).astype('float32')
    o=Observation(a,np.arange(10),np.arange(8),np.arange(5),np.arange(5),np.array([0.,.25]),np.array([.026,.276]))
    assert encode(o).shape==FEATURE_SHAPE
    changed=Observation(a*2,o.range_m,o.velocity_mps,o.elevation_deg,o.azimuth_deg,o.frame_start_s,o.frame_end_s)
    np.testing.assert_allclose(encode(o),encode(changed),rtol=1e-5)
    path=tmp_path/'bad.npz';np.savez(path,**o.arrays(),truth_xyz=[0,0,0])
    with pytest.raises(ValueError,match='unauthorized'):Observation.load(path)
    with pytest.raises(ValueError,match='outside authorized'):local_path('/home/heyiting/projects/radar_navigation')
