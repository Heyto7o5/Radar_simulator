from dataclasses import asdict,replace
import numpy as np
import pytest
from pydantic import ValidationError
from radar_platform.ground_scattering import Sigma0Observation
from radar_platform.unified.schema import Simulation
from radar_platform.unified.scene import compile_scene
from radar_platform.unified.geometry import build_layout
from radar_platform.unified.engine import simulate_frame,synthesize_adc
from radar_platform.unified.land_response import land_amplitude_factor

def observations(sigma=-10):
    return [asdict(Sigma0Observation('analytic-test-not-real-data',9e9,9.875e9,'vv',a,a-3,a+3,sigma,1,-1)) for a in [30.,70.]]

def request(kind='fmcw',speed=1.):
    w=({'kind':kind,'start_hz':9.4e9,'slope_hz_s':4e12,'ramp_s':40e-6,'slot_period_s':1e-3,
        'adc_start_s':2e-6,'sample_rate_hz':4e6,'samples':128,'loops':64} if kind=='fmcw' else
       {'kind':kind,'carrier_hz':9.4e9,'bandwidth_hz':80e6,'sample_rate_hz':200e6,'pulse_width_s':.5e-6,
        'slot_period_s':1e-3,'adc_start_s':0.,'samples':256,'loops':64})
    return Simulation.model_validate({'waveform':w,'platform':{'position_ned_m':[0,0,-6],
        'velocity_ned_mps':[speed,0,0],'boresight_ned':[1,0,1],'right_hint_ned':[0,1,0],'beam_half_angle_deg':30.},
        'array':{'layout':'siso','num_tx':1,'num_rx':1},
        'scene':{'terrain':{'kind':'flat','north_bounds_m':[5,11],'east_bounds_m':[-2,2],'spacing_m':1.,
            'relative_reflectivity':1.,'land_scattering':{'observations':observations(),'polarization':'vv'}},
            'amplitude_reference_range_m':10.,'seed':42},
        'processing':{'range_limits_m':[3,20],'azimuth_deg':[0.],'elevation_deg':[0.]},
        'sequence':{'frame_count':3,'frame_interval_s':.2}})

@pytest.mark.parametrize('kind',['pulsed_lfm','fmcw'])
def test_sigma0_changes_power_not_geometry_or_phase(kind):
    r=request(kind);a=simulate_frame(r)
    d=r.model_dump(mode='json');d['scene']['terrain']['land_scattering']['observations']=observations(0)
    b=simulate_frame(Simulation.model_validate(d))
    np.testing.assert_allclose(b.adc,a.adc*np.sqrt(10),rtol=5e-6,atol=1e-7)
    assert b.metadata['scene']['empirical_ground_objects']==1
    assert any('footprint averages' in s for s in b.metadata['warnings'])

def test_area_strength_conserved_and_no_double_cosine_factor():
    for spacing in [2.,1.,.5]:
        r=request();r.scene.terrain.spacing_m=spacing;s=compile_scene(r.scene)
        assert np.sum(s.amplitudes**2)==pytest.approx(24.)
        delta=s.positions_ned_m-np.array(r.platform.position_ned_m);distance=np.linalg.norm(delta,axis=-1)
        ct=-np.sum(s.normals_ned*delta,axis=-1)/distance
        factor=land_amplitude_factor(s,slice(None),ct[:,None],ct[:,None],delta[:,None,:],delta[:,None,:],
            distance[:,None],distance[:,None],np.ones((len(ct),1),bool),r.waveform.carrier_hz,ct[:,None])[:,0]
        np.testing.assert_allclose(factor,np.sqrt(.1),atol=1e-14)
        assert np.sum((s.amplitudes*factor)**2)==pytest.approx(24*.1)

@pytest.mark.parametrize('fault',['band','edge','polarization','land_cover','coherent'])
def test_land_model_refuses_unsupported_inputs(fault):
    d=request('pulsed_lfm').model_dump(mode='json')
    spec=d['scene']['terrain']['land_scattering']
    if fault=='band':d['waveform']['carrier_hz']=1e9
    elif fault=='edge':d['waveform']['carrier_hz']=9.86e9
    elif fault=='polarization':spec['polarization']='hh'
    elif fault=='land_cover':spec['land_cover']='urban'
    else:d['scene']['terrain']['scattering']='coherent'
    with pytest.raises((ValueError,ValidationError)):Simulation.model_validate(d)

def test_nadir_extrapolation_rejected_at_run_time_not_clipped_or_omitted():
    r=request();r.scene.terrain.north_bounds_m=(0.,1.);r.scene.terrain.east_bounds_m=(0.,1.)
    r.platform.boresight_ned=(0.,0.,1.)
    with pytest.raises(ValueError,match='outside measured coefficient coverage'):simulate_frame(r)

def test_large_bistatic_angle_is_not_treated_as_monostatic():
    d=request().model_dump(mode='json')
    d['array']={'layout':'custom','num_tx':1,'num_rx':1,'tx_local_m':[[0,-2,0]],'rx_local_m':[[0,2,0]]}
    with pytest.raises(ValueError,match='bistatic geometry'):simulate_frame(Simulation.model_validate(d))

@pytest.mark.parametrize('kind',['pulsed_lfm','fmcw'])
def test_static_ground_is_world_fixed_and_sequence_keeps_phase(kind):
    r=request(kind,speed=0);s=compile_scene(r.scene)
    np.testing.assert_array_equal(s.at(10),s.positions_ned_m)
    a=simulate_frame(r,0);b=simulate_frame(r,1)
    np.testing.assert_array_equal(a.adc,b.adc)
    r=request(kind,speed=1);whole=simulate_frame(r).adc
    short=r.model_copy(deep=True);short.waveform.loops=32;short.sequence.frame_interval_s=.032
    together=np.concatenate([simulate_frame(short,0).adc,simulate_frame(short,1).adc],axis=1)
    np.testing.assert_allclose(whole,together,rtol=1e-6,atol=1e-7)

def test_random_phase_ensemble_power_matches_sum_of_cell_powers():
    r=request('pulsed_lfm',speed=0);r.waveform.bandwidth_hz=1.
    r.waveform.samples=8;r.waveform.loops=4;r.waveform.adc_start_s=.12e-6
    layout=build_layout(r.array,r.waveform,r.platform);s=compile_scene(r.scene)
    ranges=np.linalg.norm(s.positions_ned_m-np.array(r.platform.position_ned_m),axis=1)
    expected=float(np.sum(s.amplitudes**2*.1*s.reference_range_m**4/ranges**4))
    powers=[]
    for seed in range(256):
        r.scene.seed=seed
        adc,_=synthesize_adc(r,compile_scene(r.scene),layout,0.)
        powers.append(abs(adc[0,0,0,0])**2)
    assert np.mean(powers)==pytest.approx(expected,rel=.2)


def test_land_scene_api_reports_model_and_out_of_coverage_failure(monkeypatch,tmp_path):
    from fastapi.testclient import TestClient
    from radar_platform.api import app
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    with TestClient(app) as client:
        for bad in [False,True]:
            r=request();r.sequence.frame_count=1
            if bad:
                r.scene.terrain.north_bounds_m=(0.,1.);r.scene.terrain.east_bounds_m=(0.,1.)
                r.platform.boresight_ned=(0.,0.,1.)
            result=client.post('/api/v1/scene-simulations',json=r.model_dump(mode='json'))
            assert result.status_code==202,result.text
            job=result.json()['job_id']
            with client.websocket_connect(f'/api/v1/scene-simulations/{job}/events') as ws:
                while True:
                    event=ws.receive_json()
                    if event['type'] in ['job_completed','job_failed']:break
            state=client.get(f'/api/v1/scene-simulations/{job}').json()
            if bad:assert state['status']=='failed' and 'outside measured' in state['error']
            else:assert state['status']=='complete' and state['frames'][0]['scene']['empirical_ground_objects']==1
