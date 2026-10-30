import numpy as np
import pytest
from radar_platform.unified.radiation import Pattern,TransmitAperture,element_voltage,directions,aperture_voltage
from radar_platform.unified.schema import Simulation
from radar_platform.unified.engine import simulate_frame
from radar_platform.unified.applicability import inspect_configuration


def aperture(n=8,steer=0):
    positions=np.zeros((n,3));positions[:,1]=(np.arange(n)-(n-1)/2)*299792458/10e9/2
    return TransmitAperture(positions_local_m=positions.tolist(),steering_azimuth_deg=steer)


def request(kind='pulsed_lfm',angle=0):
    wave=(dict(kind=kind,carrier_hz=10e9,bandwidth_hz=2e6,pulse_width_s=2e-6,sample_rate_hz=4e6,samples=128,loops=8,adc_start_s=0,slot_period_s=100e-6)
          if kind=='pulsed_lfm' else dict(kind=kind,start_hz=10e9,slope_hz_s=1e10,ramp_s=50e-6,sample_rate_hz=4e6,samples=128,loops=8,adc_start_s=10e-6,slot_period_s=100e-6))
    pos=(directions(np.array(angle),np.array(0.))*1000).tolist()
    return Simulation.model_validate(dict(waveform=wave,platform=dict(position_ned_m=[0,0,0],boresight_ned=[1,0,0],beam_half_angle_deg=10),
        array=dict(layout='siso',num_tx=1,num_rx=1),scene=dict(objects=[dict(kind='point',id='target',position_ned_m=pos)]),
        processing=dict(range_limits_m=[0,2000],azimuth_deg=[0.],elevation_deg=[0.])))


def test_element_power_is_square_of_voltage_and_not_applied_twice():
    pattern=Pattern(kind='cosine_power',power_exponent=2)
    v=element_voltage([1,.5,-1],pattern)
    np.testing.assert_allclose(v,[1,.5,0])
    assert (v[1]*v[1])**2==pytest.approx(.0625)
    np.testing.assert_array_equal(element_voltage([1,0,-1],Pattern()),[1,1,1])


def test_uniform_array_matches_closed_form_and_steering_sign():
    angles=np.linspace(-80,80,1001);a=aperture(8,20)
    gain=aperture_voltage(directions(angles,np.zeros_like(angles)),np.zeros_like(angles),a,10e9)
    psi=np.pi*(np.sin(np.deg2rad(angles))-np.sin(np.deg2rad(20)))
    expected=8*(np.sinc(8*psi/(2*np.pi))/np.sinc(psi/(2*np.pi)))**2
    np.testing.assert_allclose(abs(gain)**2,expected,rtol=1e-10,atol=1e-10)
    assert abs(angles[np.argmax(abs(gain))]-20)<.2
    peak=aperture_voltage(directions(np.array([20]),np.array([0])),np.array([0]),a,10e9)
    assert abs(peak[0])**2==pytest.approx(8)


@pytest.mark.parametrize('kind',['pulsed_lfm','fmcw'])
def test_complex_transmit_weights_reach_adc_with_correct_dechirp_conjugation(kind):
    r=request(kind);baseline=simulate_frame(r).adc;raw=r.model_dump()
    ap=aperture(2).model_dump();ap['phases_rad']=[0,np.pi/2]
    raw['radiation']={'transmit_aperture':ap}
    observed=simulate_frame(Simulation.model_validate(raw)).adc
    gain=(1+1j)/np.sqrt(2)
    if kind=='fmcw':gain=gain.conjugate()
    np.testing.assert_allclose(observed,baseline*gain,rtol=2e-6,atol=1e-7)


def test_radiation_replaces_old_cone_without_clipping_sidelobe_scene():
    r=request(angle=30);assert not np.any(simulate_frame(r).adc)
    raw=r.model_dump();raw['radiation']={}
    result=simulate_frame(Simulation.model_validate(raw))
    assert np.max(abs(result.adc))>0
    assert result.metadata['beam_response']['model']=='radiation_pattern'
    raw['radiation']['replace_legacy_cone']=False
    assert not np.any(simulate_frame(Simulation.model_validate(raw)).adc)


def test_scan_schedule_changes_global_time_response_without_moving_array():
    raw=request(angle=15).model_dump();raw['sequence']=dict(frame_count=2,frame_interval_s=.02)
    ap=aperture(16,-15).model_dump();ap['scan_schedule']=[dict(time_s=.02,azimuth_deg=15)]
    raw['radiation']={'transmit_aperture':ap};r=Simulation.model_validate(raw)
    a,b=simulate_frame(r,0),simulate_frame(r,1)
    assert np.max(abs(b.adc))>3*np.max(abs(a.adc))
    np.testing.assert_array_equal(a.metadata['array']['tx_local_m'],b.metadata['array']['tx_local_m'])
    raw['radiation']['transmit_aperture']['scan_schedule']=[dict(time_s=.0003,azimuth_deg=15)]
    check=inspect_configuration(raw)
    assert next(x for x in check['checks'] if x['code']=='transmit_scan')['status']=='risk'


def test_near_field_bandwidth_and_pattern_band_are_not_silently_accepted():
    raw=request().model_dump();ap=aperture(2).model_dump();ap['positions_local_m']=[[0,-.5,0],[0,.5,0]]
    raw['radiation']={'transmit_aperture':ap};raw['scene']['objects'][0]['position_ned_m']=[1,0,0]
    with pytest.raises(ValueError,match='near field'):simulate_frame(Simulation.model_validate(raw))
    raw['waveform']['bandwidth_hz']=100e6;raw['waveform']['sample_rate_hz']=200e6
    with pytest.raises(ValueError,match='narrowband'):Simulation.model_validate(raw)
    raw=request().model_dump();raw['radiation']={'tx_element':dict(kind='axisymmetric_table',angle_deg=[0,90,180],power_db=[0,-30,-60],frequency_min_hz=9e9,frequency_max_hz=9.5e9,evidence='assumed',source='test')}
    with pytest.raises(ValueError,match='frequency band'):Simulation.model_validate(raw)


def test_noncentered_aperture_is_rejected_not_silently_rebased():
    with pytest.raises(ValueError,match='phase center'):TransmitAperture(positions_local_m=[[0,1,0]])


def test_drive_rescaling_cannot_change_normalized_power_or_overflow():
    a=aperture();u=directions(np.array([0.,10.]),np.array([0.,0.]))
    expected=aperture_voltage(u,np.zeros(2),a,10e9)
    for scale in [1e-200,1e200]:
        raw=a.model_dump();raw['amplitudes']=[scale]*8
        np.testing.assert_allclose(aperture_voltage(u,np.zeros(2),TransmitAperture(**raw),10e9),expected)


def test_pattern_table_is_voltage_from_power_db_not_power_gain():
    p=Pattern(kind='axisymmetric_table',angle_deg=[0,90,180],power_db=[0,-20,-40],frequency_min_hz=9e9,frequency_max_hz=11e9,
              evidence='assumed',source='test profile, not measurement')
    np.testing.assert_allclose(element_voltage([1,0,-1],p),[1,.1,.01])


def test_scan_checks_are_json_serializable_through_session_and_api(tmp_path):
    import json
    from fastapi.testclient import TestClient
    from radar_platform.api import app
    from radar_platform.unified.session import SimulationSession
    raw=request().model_dump();ap=aperture().model_dump()
    ap['scan_schedule']=[dict(time_s=.0003,azimuth_deg=10)]
    raw['radiation']={'transmit_aperture':ap}
    session=SimulationSession(Simulation.model_validate(raw));session.save_inputs(tmp_path)
    saved=json.loads((tmp_path/'applicability.json').read_text())
    assert next(x for x in saved['checks'] if x['code']=='transmit_scan')['metrics']['schedule_changes_inside_cpis']==1
    with TestClient(app) as client:
        response=client.post('/api/v1/scene-check',json=raw)
        assert response.status_code==200,response.text
