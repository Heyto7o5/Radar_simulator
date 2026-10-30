import numpy as np
import pytest
from radar_platform.xyz_features import encode_rda,FEATURE_SHAPE


def test_rda_encoding_retains_frame_order_and_removes_uniform_gain():
    rng=np.random.default_rng(4)
    a=rng.uniform(.1,2,(2,11,9,7)).astype('float32')
    x=encode_rda(a)
    assert x.shape==FEATURE_SHAPE and x.dtype==np.float32
    np.testing.assert_allclose(x,encode_rda(a*3),rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(x[::-1],encode_rda(a[::-1]),rtol=1e-6,atol=1e-6)
    assert not np.allclose(x[0],x[1])


@pytest.mark.parametrize('value',[np.zeros((2,4,4,4)),np.ones((1,4,4,4)),np.full((2,4,4,4),np.nan),-np.ones((2,4,4,4))])
def test_invalid_observation_rejected(value):
    with pytest.raises(ValueError):encode_rda(value)
