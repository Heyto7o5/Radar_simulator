import numpy as np
import pytest
from pydantic import ValidationError
from scipy.spatial.transform import Rotation
from radar_platform.models import C0
from radar_platform.unified.trajectory import RigidTrajectory,TrajectoryEvaluator
from radar_platform.unified.schema import Simulation
from radar_platform.unified.engine import simulate_frame


def specification(speed=2.,omega=0.):
    t=np.array([0.,.01,.03])
    return dict(time_s=t.tolist(),position_ned_m=np.column_stack([speed*t,np.zeros((3,2))]).tolist(),
                velocity_ned_mps=np.tile([speed,0,0],(3,1)).tolist(),
                quaternion_xyzw=Rotation.from_rotvec(t[:,None]*[0,0,omega]).as_quat().tolist(),
                provenance='analytic test trajectory, not measured')


def request(kind='fmcw'):
    waveform=dict(kind='fmcw',start_hz=9e9,slope_hz_s=1e12,ramp_s=20e-6,slot_period_s=.001,
                  adc_start_s=2e-6,sample_rate_hz=4e6,samples=64,loops=16)
    if kind=='pulsed_lfm':waveform=dict(kind=kind,carrier_hz=9e9,bandwidth_hz=2e6,pulse_width_s=2e-6,
                                      slot_period_s=.001,adc_start_s=0.,sample_rate_hz=4e6,samples=64,loops=16)
    return Simulation.model_validate(dict(waveform=waveform,
        platform=dict(position_ned_m=[0,0,0],velocity_ned_mps=[2,0,0],boresight_ned=[1,0,0],right_hint_ned=[0,1,0],beam_half_angle_deg=80),
        array=dict(layout='siso',num_tx=1,num_rx=1),scene=dict(objects=[dict(kind='point',id='p',position_ned_m=[10,1,0])]),
        processing=dict(range_limits_m=[0,100],azimuth_deg=[0.],elevation_deg=[0.]),sequence=dict(frame_count=1)))


@pytest.mark.parametrize('kind',['fmcw','pulsed_lfm'])
def test_sampled_constant_motion_matches_existing_adc(kind):
    a=request(kind);baseline=simulate_frame(a)
    raw=a.model_dump();raw['platform']['trajectory']=specification()
    b=simulate_frame(Simulation.model_validate(raw))
    np.testing.assert_allclose(b.adc,baseline.adc,rtol=1e-6,atol=1e-6)
    assert b.metadata['platform_motion']=='sampled_Hermite_translation_SLERP_attitude'


def test_hermite_acceleration_rotation_and_no_extrapolation():
    raw=specification(speed=0,omega=5)
    t=np.array(raw['time_s']);raw['position_ned_m']=np.column_stack([t*t,np.zeros((3,2))]).tolist()
    raw['velocity_ned_mps']=np.column_stack([2*t,np.zeros((3,2))]).tolist()
    evaluator=TrajectoryEvaluator(RigidTrajectory.model_validate(raw))
    p,v,r,w=evaluator.at(.015)
    assert p[0]==pytest.approx(.015**2) and v[0]==pytest.approx(.03)
    np.testing.assert_allclose(r,Rotation.from_rotvec([0,0,.075]).as_matrix(),atol=1e-12)
    np.testing.assert_allclose(w,[0,0,5],atol=1e-12)
    with pytest.raises(ValueError):evaluator.at(.031)
    raw['quaternion_xyzw'][0]=[0,0,0,2]
    with pytest.raises(ValidationError):RigidTrajectory.model_validate(raw)


def test_rotating_element_phase_includes_rotational_velocity():
    a=request();raw=a.model_dump();raw['platform']['trajectory']=specification(0,10)
    raw['array']=dict(layout='custom',num_tx=1,num_rx=1,tx_local_m=[[0,1,0]],rx_local_m=[[0,-.5,0]])
    a=Simulation.model_validate(raw);frame=simulate_frame(a)
    k=7;t=k*.001;fast=a.waveform.adc_start_s+np.arange(64)/a.waveform.sample_rate_hz
    target=np.array([10,1,0]);rotation=Rotation.from_rotvec([0,0,10*t]).as_matrix()
    tx=rotation@np.array([0,1,0]);rx=rotation@np.array([0,-.5,0])
    dt=target-tx;dr=target-rx;rt=np.linalg.norm(dt);rr=np.linalg.norm(dr)
    rate=-dt@np.cross([0,0,10],tx)/rt-dr@np.cross([0,0,10],rx)/rr
    tau=(rt+rr+rate*fast)/C0;wave=a.waveform
    expected=a.scene.amplitude_reference_range_m**2/(rt*rr)*np.exp(2j*np.pi*(wave.start_hz*tau+wave.slope_hz_s*fast*tau-.5*wave.slope_hz_s*tau*tau))
    np.testing.assert_allclose(frame.adc[:,k,0,0],expected,rtol=1e-6)


def test_sequence_continuity_and_coverage_rejection():
    raw=request().model_dump();raw['platform']['trajectory']=specification(2,2)
    whole=simulate_frame(Simulation.model_validate(raw)).adc
    raw['waveform']['loops']=8;raw['sequence']={'frame_count':2,'frame_interval_s':.008}
    a=Simulation.model_validate(raw)
    joined=np.concatenate([simulate_frame(a,i).adc for i in range(2)],axis=1)
    np.testing.assert_allclose(joined,whole,rtol=1e-6,atol=1e-6)
    raw['sequence']['start_time_s']=.025
    with pytest.raises(ValidationError,match='cover every Tx'):Simulation.model_validate(raw)


def test_beam_visibility_follows_sampled_attitude_and_quaternion_sign_is_irrelevant():
    raw=request().model_dump();raw['platform']['beam_half_angle_deg']=10
    trajectory=specification(0,30);raw['platform']['trajectory']=trajectory
    before=simulate_frame(Simulation.model_validate(raw)).adc
    assert np.linalg.norm(before[:,0])>0
    assert np.linalg.norm(before[:,-1])==0
    trajectory['quaternion_xyzw'][1]=(-np.array(trajectory['quaternion_xyzw'][1])).tolist()
    after=simulate_frame(Simulation.model_validate(raw)).adc
    np.testing.assert_allclose(before,after,rtol=1e-6,atol=1e-6)


def test_engine_applies_two_way_beam_voltage_once():
    from radar_platform.unified.antenna import two_way_voltage_gain
    raw=request().model_dump();raw['platform']['beam_half_angle_deg']=8
    hard=simulate_frame(Simulation.model_validate(raw)).adc
    raw['platform'].update(beam_model='cosine_edge',beam_taper_deg=4)
    smooth=simulate_frame(Simulation.model_validate(raw)).adc
    starts=np.arange(16)*.001
    cosine=(10-2*starts)/np.sqrt((10-2*starts)**2+1)
    gain=two_way_voltage_gain(cosine,8,'cosine_edge',4)
    np.testing.assert_allclose(smooth,hard*gain[None,:,None,None],rtol=1e-6,atol=1e-6)
