import numpy as np
import pytest
from radar_platform.unified.visibility import blocked_segments
from radar_platform.unified.schema import Simulation
from radar_platform.unified.engine import simulate_frame
from radar_platform.unified.preflight import preflight


def wall(x=5):
    return np.array([[[x,-2,-2],[x,2,-2],[x,2,2]],[[x,-2,-2],[x,2,2],[x,-2,2]]],float)


def config(enabled=False):
    return Simulation.model_validate(dict(
        waveform=dict(kind='fmcw',start_hz=10e9,slope_hz_s=1e12,ramp_s=20e-6,
                      slot_period_s=50e-6,adc_start_s=2e-6,sample_rate_hz=4e6,samples=64,loops=8),
        platform=dict(position_ned_m=[0,0,0],boresight_ned=[1,0,0],beam_half_angle_deg=70),
        array=dict(layout='siso',num_tx=1,num_rx=1),
        scene=dict(amplitude_reference_range_m=10,objects=[
            dict(kind='point',id='target',position_ned_m=[10,0,0]),
            dict(kind='plate',id='wall',position_ned_m=[5,0,0],normal_ned=[-1,0,0],
                 size_m=[5,5],divisions=1,relative_reflectivity=1e-10,scattering='coherent')]),
        visibility=dict(occlusion_enabled=enabled,report_coverage=True),
        processing=dict(range_limits_m=[0,100],azimuth_deg=[0.],elevation_deg=[0.])))


def target_counts(frame):
    c=frame.metadata['visibility']['coverage']
    obj=c['objects'].index('target');i=c['object_index'].index(obj)
    return {k:c[k][i] for k in ['illuminated_tx_slots','unblocked_channel_slots','sampled_echo_channel_slots']}


def test_segment_triangle_interior_edges_endpoints_and_backside():
    endpoints=np.array([[10,0,0],[4,0,0],[5,0,0],[10,10,0]])
    assert blocked_segments([0,0,0],endpoints,wall()).tolist()==[True,False,False,False]
    assert blocked_segments([10,0,0],[[0,0,0]],wall())[0]
    # A ray starting on its reflector is not self-blocked.
    assert not blocked_segments([5,0,0],[[10,0,0]],wall())[0]


def test_endpoint_tolerance_is_physical_not_fraction_of_range():
    assert blocked_segments([0,0,0],[[10000,0,0]],wall(9999.99),endpoint_epsilon_m=1e-4)[0]
    assert not blocked_segments([0,0,0],[[10000,0,0]],wall(10000),endpoint_epsilon_m=1e-4)[0]


def test_occlusion_changes_echo_not_only_plot_and_wall_does_not_block_itself():
    off=simulate_frame(config(False));on=simulate_frame(config(True))
    assert target_counts(off)['unblocked_channel_slots']==8
    assert target_counts(on)==dict(illuminated_tx_slots=8,unblocked_channel_slots=0,sampled_echo_channel_slots=0)
    assert np.max(abs(on.adc))<np.max(abs(off.adc))*1e-8
    raw=config(True).model_dump();raw['scene']['objects']=raw['scene']['objects'][1:]
    wall_only=simulate_frame(Simulation.model_validate(raw))
    np.testing.assert_array_equal(on.adc,wall_only.adc)
    assert np.any(on.adc)


def test_disabled_switch_and_unobstructed_enabled_are_identical():
    r=config(False);raw=r.model_dump();raw['visibility']['report_coverage']=False
    np.testing.assert_array_equal(simulate_frame(r).adc,simulate_frame(Simulation.model_validate(raw)).adc)
    raw=r.model_dump();raw['scene']['objects'][1]['position_ned_m']=[15,0,0]
    off=simulate_frame(Simulation.model_validate(raw));raw['visibility']['occlusion_enabled']=True
    on=simulate_frame(Simulation.model_validate(raw))
    np.testing.assert_array_equal(off.adc,on.adc)


def test_receiver_path_can_be_blocked_even_when_transmitter_path_is_clear():
    raw=config(True).model_dump()
    raw['array']=dict(layout='custom',num_tx=1,num_rx=1,tx_local_m=[[0,0,0]],rx_local_m=[[0,4,0]])
    raw['scene']['objects'][1].update(position_ned_m=[5,2,0],size_m=[1,4])
    frame=simulate_frame(Simulation.model_validate(raw))
    assert target_counts(frame)['unblocked_channel_slots']==0
    raw['array']['rx_local_m']=[[0,0,0]]
    assert target_counts(simulate_frame(Simulation.model_validate(raw)))['unblocked_channel_slots']==8


def test_moving_wall_and_terrain_only_scope_and_budget():
    raw=config(True).model_dump();raw['scene']['objects'][1]['velocity_ned_mps']=[0,1000,0]
    raw['sequence']=dict(frame_count=2,frame_interval_s=.01)
    r=Simulation.model_validate(raw)
    assert target_counts(simulate_frame(r,0))['unblocked_channel_slots']==0
    assert target_counts(simulate_frame(r,1))['unblocked_channel_slots']==8
    raw=config(True).model_dump();raw['visibility']['occluders']='terrain'
    assert target_counts(simulate_frame(Simulation.model_validate(raw)))['unblocked_channel_slots']==8
    raw=config(True).model_dump();raw['visibility']['max_ray_triangle_tests']=1
    with pytest.raises(ValueError,match='ray-triangle budget'):preflight(Simulation.model_validate(raw))


def test_dem_ridge_blocks_a_low_ray_but_not_high_ray():
    from radar_platform.unified.scene import terrain_mesh
    from radar_platform.unified.schema import Terrain
    mesh,_=terrain_mesh(Terrain(kind='dem',north_m=[0,5,10],east_m=[-2,2],
                               elevation_up_m=[[0,3,0],[0,3,0]]))
    assert blocked_segments([0,0,-1],[[10,0,-1]],mesh.triangles)[0]
    assert not blocked_segments([0,0,-5],[[10,0,-5]],mesh.triangles)[0]


def test_shared_api_exposes_occlusion_coverage_and_persists_enabled_state(tmp_path,monkeypatch):
    import json
    from fastapi.testclient import TestClient
    from radar_platform.api import app
    from radar_platform.unified.jobs import manager
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    with TestClient(app) as client:
        response=client.post('/api/v1/scene-simulations',json=config(True).model_dump(mode='json'))
        assert response.status_code==202,response.text
        created=response.json();job=created['job_id']
        assert created['estimate']['visibility']['enabled']
        with client.websocket_connect(f'/api/v1/scene-simulations/{job}/events') as ws:
            while True:
                event=ws.receive_json()
                if event['type']=='job_completed':break
                if event['type']=='snapshot' and event['job']['status']=='complete':break
                assert event['type']!='job_failed',event
        status=client.get(f'/api/v1/scene-simulations/{job}').json()
        coverage=status['frames'][0]['visibility']['coverage']
        assert coverage['occlusion_evaluated']
        root=manager.get(job).root
        assert json.loads((root/'scene.json').read_text())['environment_occlusion']
        assert json.loads((root/'frame_0000/coverage.json').read_text())==coverage


def test_bvh_and_brute_force_produce_same_echo_and_channel_coverage():
    raw=config(True).model_dump()
    brute=simulate_frame(Simulation.model_validate(raw))
    raw['visibility']['accelerator']='bvh'
    fast=simulate_frame(Simulation.model_validate(raw))
    np.testing.assert_array_equal(brute.adc,fast.adc)
    assert brute.metadata['visibility']['coverage']==fast.metadata['visibility']['coverage']
    assert fast.metadata['visibility']['bvh_work']['triangle_tests']>0
    raw['visibility']['max_ray_triangle_tests']=1
    with pytest.raises(ValueError,match='runtime work budget'):simulate_frame(Simulation.model_validate(raw))


@pytest.mark.parametrize('radiation',[None,{}])
def test_batched_bvh_preserves_echo_and_coverage(radiation):
    raw=config(True).model_dump()
    raw['scene']['objects']=[]
    raw['scene']['terrain']=dict(kind='hills',north_bounds_m=[-6,6],east_bounds_m=[-6,6],spacing_m=1,
        hills=[dict(center_ne_m=[0,0],height_m=3,sigma_ne_m=[1,2])])
    raw['platform'].update(position_ned_m=[0,0,-10],velocity_ned_mps=[1,2,3],boresight_ned=[0,0,1])
    raw['visibility'].update(accelerator='bvh',batch_bvh=False)
    raw['radiation']=radiation
    a=simulate_frame(Simulation.model_validate(raw))
    raw['visibility']['batch_bvh']=True
    b=simulate_frame(Simulation.model_validate(raw))
    np.testing.assert_array_equal(a.adc,b.adc)
    assert a.metadata['visibility']['coverage']==b.metadata['visibility']['coverage']
