import numpy as np
import pytest
from radar_platform.unified.schema import Simulation
from radar_platform.unified.motion_check import motion_error_report


def config(velocity=(0,1200,0),kind='fmcw'):
    wave=dict(kind='fmcw',start_hz=10e9,slope_hz_s=1e10,ramp_s=.0004,slot_period_s=.0005,
              samples=128,sample_rate_hz=4e5,adc_start_s=1e-6,loops=4)
    if kind=='pulsed_lfm':
        wave=dict(kind=kind,carrier_hz=10e9,bandwidth_hz=2e5,pulse_width_s=20e-6,
                  slot_period_s=.0005,samples=128,sample_rate_hz=4e5,adc_start_s=0,loops=4)
    return Simulation.model_validate(dict(waveform=wave,
        platform=dict(position_ned_m=[0,0,0],velocity_ned_mps=velocity,boresight_ned=[1,0,0]),
        array=dict(layout='siso',num_tx=1,num_rx=1),
        scene=dict(objects=[dict(kind='point',id='p',position_ned_m=[10,0,0])]),
        processing=dict(range_limits_m=[0,100],azimuth_deg=[0.],elevation_deg=[0.])))


@pytest.mark.parametrize('kind',['fmcw','pulsed_lfm'])
def test_axial_translation_has_negligible_path_linearization_error(kind):
    result=motion_error_report(config((1200,0,0),kind))
    assert result['metrics']['max_two_way_path_error_m']<1e-12
    assert result['metrics']['max_unwrapped_phase_error_rad']<1e-8


def test_transverse_error_matches_analytic_curved_range():
    r=config();result=motion_error_report(r,max_slots=1,fast_samples=128)
    fast=r.waveform.adc_start_s+(r.waveform.samples-1)/r.waveform.sample_rate_hz
    expected=2*(np.sqrt(10**2+(1200*fast)**2)-10)
    assert result['metrics']['max_two_way_path_error_m']==pytest.approx(expected,abs=1e-12)
    assert result['status']=='risk'
    assert result['coverage']['all_fast_samples_checked']
    assert result['coverage']['slots_checked']==1


def test_radar_and_target_exchange_preserves_relative_geometry_error():
    radar=config();raw=radar.model_dump()
    raw['platform']['velocity_ned_mps']=[0,0,0]
    raw['scene']['objects'][0]['velocity_ned_mps']=[0,-1200,0]
    target=Simulation.model_validate(raw)
    a=motion_error_report(radar);b=motion_error_report(target)
    assert a['metrics']==b['metrics']


def test_common_translation_cancels_and_grid_refinement_not_smaller_maximum():
    raw=config().model_dump();raw['scene']['objects'][0]['velocity_ned_mps']=[0,1200,0]
    assert motion_error_report(Simulation.model_validate(raw))['metrics']['max_two_way_path_error_m']==0
    a=motion_error_report(config(),max_slots=4,fast_samples=2)
    b=motion_error_report(config(),max_slots=4,fast_samples=128)
    assert b['metrics']['max_unwrapped_phase_error_rad']>=a['metrics']['max_unwrapped_phase_error_rad']
    with pytest.raises(ValueError):motion_error_report(config(),fast_samples=1)


def test_sampled_acceleration_reference_has_analytic_quadratic_error():
    raw=config((0,0,0)).model_dump();acceleration=1e6;end=.002
    raw['platform']['trajectory']=dict(time_s=[0,end],position_ned_m=[[0,0,0],[.5*acceleration*end**2,0,0]],
        velocity_ned_mps=[[0,0,0],[acceleration*end,0,0]],quaternion_xyzw=[[0,0,0,1],[0,0,0,1]],
        provenance='analytic acceleration stress test, not observed motion')
    r=Simulation.model_validate(raw);result=motion_error_report(r,max_slots=1,fast_samples=128)
    fast=r.waveform.adc_start_s+(r.waveform.samples-1)/r.waveform.sample_rate_hz
    assert result['metrics']['max_two_way_path_error_m']==pytest.approx(acceleration*fast**2,abs=1e-12)
    assert result['status']=='risk'


def test_rotating_array_reference_uses_rotating_element_positions():
    from scipy.spatial.transform import Rotation
    raw=config((0,0,0)).model_dump();omega=100.;end=.002
    raw['array']=dict(layout='custom',num_tx=1,num_rx=1,tx_local_m=[[0,1,0]],rx_local_m=[[0,1,0]])
    raw['platform']['trajectory']=dict(time_s=[0,end],position_ned_m=[[0,0,0],[0,0,0]],
        velocity_ned_mps=[[0,0,0],[0,0,0]],quaternion_xyzw=Rotation.from_rotvec([[0,0,0],[0,0,omega*end]]).as_quat().tolist(),
        provenance='analytic rigid rotation stress test')
    r=Simulation.model_validate(raw);result=motion_error_report(r,max_slots=1,fast_samples=128)
    fast=r.waveform.adc_start_s+np.arange(r.waveform.samples)/r.waveform.sample_rate_hz
    exact=2*np.sqrt((10+np.sin(omega*fast))**2+np.cos(omega*fast)**2)
    linear=2*np.sqrt(101)+2*10*omega/np.sqrt(101)*fast
    assert result['metrics']['max_two_way_path_error_m']==pytest.approx(float(np.max(abs(exact-linear))),abs=1e-12)
