import numpy as np

from radar_platform.navigation.acquisition import RadarWorld
from radar_platform.navigation.dynamics import State,Command,propagate
from radar_platform.navigation.timeline import covered_cpi_end
from radar_platform.navigation.controller import Controller,Estimate
from radar_platform.unified.schema import Simulation


def test_radar_frames_cross_32_seconds_without_extrapolation_or_changed_echo():
    config=Simulation(waveform=dict(kind='fmcw',start_hz=9.9e9,slope_hz_s=3.125e12,ramp_s=80e-6,
        slot_period_s=100e-6,adc_start_s=5e-6,sample_rate_hz=3e6,samples=192,loops=64),
        platform=dict(position_ned_m=[0,0,-30],boresight_ned=[0,0,1],right_hint_ned=[0,1,0],beam_half_angle_deg=60),
        array=dict(layout='ula',num_tx=4,num_rx=1),
        scene=dict(objects=[dict(kind='point',id='fixed',position_ned_m=[0,10,0])]),
        processing=dict(range_limits_m=[8,120],azimuth_deg=[-30,0,30],elevation_deg=[0]))
    world=RadarWorld(config);reference=None
    for start in [0.,31.75,32.,32.25,60.,64.]:
        state=State(start,np.array([0.,0.,30.]),np.array([3.,0.,0.]),.1)
        _,samples=propagate(state,Command(np.zeros(3),.4),covered_cpi_end(config,start))
        frame,_,_=world.acquire(samples,0,assessment=True)
        if reference is None:reference=frame.magnitude
        else:np.testing.assert_allclose(frame.magnitude,reference,atol=1e-5,rtol=1e-6)


def test_gentle_terminal_guidance_does_not_chase_small_position_error():
    controller=Controller(gentle_terminal=True)
    measurement=Estimate(0.,np.array([.4,-.3,-30.2]),np.zeros(3))
    command,info=controller.step(measurement)
    np.testing.assert_array_equal(command.acceleration_body_mps2,np.zeros(3))
    for t in [1.,2.,3.]:
        _,info=controller.step(Estimate(t,measurement.relative_position_body_m,np.zeros(3)))
        assert not info['estimated_arrival']
    _,info=controller.step(Estimate(4.,measurement.relative_position_body_m,np.zeros(3)))
    assert info['estimated_arrival']
    _,info=Controller(gentle_terminal=True).step(Estimate(0.,np.array([1.8,0,-30.]),np.zeros(3)))
    assert info['mode']=='approach'
