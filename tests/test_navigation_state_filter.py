import numpy as np
import pytest

from radar_platform.navigation.controller import Estimate
from radar_platform.navigation.dynamics import Command
from radar_platform.navigation.state_filter import RadarStateFilter


def test_prediction_uses_relative_commanded_yaw_without_world_pose():
    filt=RadarStateFilter(np.ones(6))
    r=np.array([20.,10.,-30.]);v=np.array([3.,0.,0.])
    filt.update(Estimate(0.,r,v),Command.coast())
    # Increasing yaw turns body axes clockwise, so stationary target coordinates rotate oppositely.
    out,_=filt.update(Estimate(.5,np.full(3,1e9),np.full(3,1e9),valid=False),Command(np.zeros(3),.5))
    angle=.25;q=np.array([[np.cos(angle),np.sin(angle),0],[-np.sin(angle),np.cos(angle),0],[0,0,1]])
    np.testing.assert_allclose(out.relative_position_body_m,q@(r-v*.5),atol=1e-12)
    np.testing.assert_allclose(out.velocity_body_mps,q@v,atol=1e-12)
    assert not out.valid


def test_stationary_radar_noise_is_smoothed_and_covariance_stays_positive():
    rng=np.random.default_rng(17);filt=RadarStateFilter([1.,1.,.6,.25,.25,.15])
    target=np.array([.5,-.7,-30.]);raw=[];filtered=[]
    for k in range(80):
        r=target+rng.normal(size=3)*[1,1,.6];v=rng.normal(size=3)*[.25,.25,.15]
        out,_=filt.update(Estimate(k*.5,r,v),Command.coast())
        raw.append(r-target);filtered.append(out.relative_position_body_m-target)
        assert np.linalg.eigvalsh(filt.covariance).min()>0
    assert np.mean(np.linalg.norm(filtered[10:],axis=1))<.75*np.mean(np.linalg.norm(raw[10:],axis=1))
    with pytest.raises(ValueError,match='increasing'):filt.update(Estimate(39.5,target,np.zeros(3)),Command.coast())


def test_no_initial_radar_state_is_fabricated_when_observation_invalid():
    filt=RadarStateFilter(np.ones(6));measurement=Estimate(0.,np.zeros(3),np.zeros(3),valid=False)
    result,diagnostic=filt.update(measurement,Command.coast())
    assert result is measurement and filt.mean is None and not diagnostic['initialized']
