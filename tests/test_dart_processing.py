import numpy as np
import pytest
from radar_platform.dart_processing import published_to_closing_axis,rover_magnitude,fit_zero_doppler_profile,apply_zero_doppler_profile


def test_velocity_conversion_preserves_nyquist_coordinate_not_just_flip_labels():
    raw=np.arange(-4,4)*.1
    values=np.arange(8)[None,:]
    v,out=published_to_closing_axis(raw,values)
    assert np.allclose(v,[-.3,-.2,-.1,0,.1,.2,.3,.4])
    assert out.tolist()==[[7,6,5,4,3,2,1,0]]


def test_known_tone_range_doppler_angle_and_parseval():
    sample=np.arange(512)[:,None,None,None]
    loop=np.arange(256)[None,:,None,None]
    channel=np.arange(8).reshape(2,4).T[None,None,:,:]
    adc=np.exp(2j*np.pi*(sample*20/512+loop*5/256+channel*2/8))
    magnitude=rover_magnitude(adc)
    assert np.unravel_index(magnitude.argmax(),magnitude.shape)==(20,133,6)
    virtual=adc.transpose(0,1,3,2).reshape(512,256,8)
    rd=np.fft.fftshift(np.fft.fft2(virtual*np.hanning(512)[:,None,None]*np.hanning(256)[None,:,None],axes=(0,1)),axes=1)
    assert np.allclose(np.mean(magnitude**2,axis=2),np.sum(abs(rd)**2,axis=2),rtol=1e-10,atol=1e-8)


def test_background_profile_frozen_without_test_frame_fitting():
    calibration=np.ones((3,2,8,4))*2
    profile=fit_zero_doppler_profile(calibration)
    test=np.ones((2,8,4))*5
    before=test.copy();result=apply_zero_doppler_profile(test,profile)
    assert np.all(result[:,3:6]==3) and np.all(result[:,:3]==5)
    assert np.array_equal(test,before)
    with pytest.raises(ValueError):fit_zero_doppler_profile(calibration[:1])
