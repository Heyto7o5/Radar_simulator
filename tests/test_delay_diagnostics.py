import numpy as np
import pytest
from radar_platform.delay_diagnostics import profile_shift

def test_relative_delay_sign_and_unknown_complex_gain():
    r=np.arange(0,10,.005)
    a=np.exp(-.5*((r-4)/.04)**2)*np.exp(2j*r)
    for shift in [-.3,0,.3]:
        b=2.5*np.exp(.7j)*np.exp(-.5*((r-4-shift)/.04)**2)*np.exp(2j*(r-shift))
        m=profile_shift(a,b,r,(1,8))
        assert m['shift_m']==pytest.approx(shift,abs=.005)
        assert m['normalized_complex_correlation']>.999

def test_invalid_axis_and_empty_signal_not_reported_as_a_good_delay():
    r=np.arange(10,dtype=float);a=np.ones(10,complex)
    with pytest.raises(ValueError):profile_shift(a,np.zeros(10,complex),r,(1,8))
    r[5]+=.1
    with pytest.raises(ValueError):profile_shift(a,a,r,(1,8))
