import numpy as np
import pytest

from radar_platform.navigation.observation import Observation
from radar_platform.navigation.features_v2 import directions,coordinate_grid,encode_v2


def test_range_angle_mapping_has_correct_body_signs_and_units():
    unit=directions(np.array([-30.,0.,30.]),np.array([-30.,0.,30.]))
    np.testing.assert_allclose(np.linalg.norm(unit,axis=-1),1,atol=1e-12)
    np.testing.assert_allclose(unit[1,1],[0.,0.,-1.],atol=1e-12)
    assert unit[2,1,0]==pytest.approx(.5)
    assert unit[1,2,1]==pytest.approx(.5)
    grid=coordinate_grid([10.,100.],[-60.,60.],[-60.,60.])
    np.testing.assert_allclose(grid[0,10,10],[0.,0.,-10.])
    np.testing.assert_allclose(grid[-1,10,10],[0.,0.,-100.])


def test_native_doppler_moments_preserve_static_scene_velocity_projection():
    angles=np.linspace(-45,45,5);velocity=np.linspace(-15,15,64);u=directions(angles,angles)
    platform=np.array([2.,-1.,.4]);closing=u@platform
    power=np.zeros((2,8,64,5,5),np.float32)
    for e in range(5):
        for a in range(5):
            i=np.searchsorted(velocity,closing[e,a])-1
            fraction=(closing[e,a]-velocity[i])/(velocity[i+1]-velocity[i])
            power[:,:,i,e,a]=1-fraction;power[:,:,i+1,e,a]=fraction
    o=Observation(np.sqrt(power),np.arange(8)+10,velocity,angles,angles,np.array([0.,.25]),np.array([.026,.276]))
    features,baseline=encode_v2(o)
    assert features.shape==(2,20,96,21,21)
    assert np.isfinite(features).all()
    np.testing.assert_allclose(baseline,np.tile(platform,(2,1)),atol=.03,rtol=0)
