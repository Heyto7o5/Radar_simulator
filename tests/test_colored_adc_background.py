import numpy as np
import pytest
from radar_platform.colored_adc_background import colored_adc_noise,expected_windowed_range_power


def test_flat_spectrum_matches_white_variance_and_seed():
    q=np.full(64,2.5)
    a=colored_adc_noise((64,4096,1,1),q,42)
    assert np.mean(abs(a)**2)==pytest.approx(2.5,rel=.015)
    assert np.array_equal(a,colored_adc_noise(a.shape,q,42))
    assert not np.any(colored_adc_noise((4,3,1,1),np.zeros(4),42))
    expected=expected_windowed_range_power(q,np.hanning(64),np.hanning(16),8)
    np.testing.assert_allclose(expected,8*2.5*np.sum(np.hanning(64)**2)*np.sum(np.hanning(16)**2))


def test_colored_psd_and_hann_convolution_match_monte_carlo():
    k=np.arange(64);q=.1+3*np.exp(-((k-15)/4)**2)+np.exp(-((k-45)/2)**2)
    a=colored_adc_noise((64,8192,1,1),q,81)[:,:,0,0]
    measured=np.mean(abs(np.fft.fft(a*np.hanning(64)[:,None],axis=0))**2,axis=1)
    expected=expected_windowed_range_power(q,np.hanning(64),np.ones(1))
    np.testing.assert_allclose(measured,expected,rtol=.07)
    spectral=np.mean(abs(np.fft.fft(a,axis=0))**2,axis=1)/64
    np.testing.assert_allclose(spectral,q,rtol=.07)


def test_invalid_spectrum_not_silently_clipped():
    with pytest.raises(ValueError):colored_adc_noise((4,3,1,1),[1,-1,1,1],42)
    with pytest.raises(ValueError):colored_adc_noise((4,3,1,1),[1,2],42)
    with pytest.raises(ValueError):expected_windowed_range_power([1,2],[1],[1])


def test_unified_colored_noise_requires_provenance_and_replays():
    from pydantic import ValidationError
    from radar_platform.unified.schema import Simulation
    from radar_platform.unified.environment import receiver_noise
    raw={'scene':{'objects':[{'kind':'point','id':'p','position_ned_m':[0,0,0]}]},
         'receiver_noise':{'fast_time_spectral_power':[.2]*1024,'seed':21}}
    with pytest.raises(ValidationError,match='provenance'):Simulation.model_validate(raw)
    raw['receiver_noise']['provenance']='analytic PSD fixture'
    request=Simulation.model_validate(raw)
    a=receiver_noise(request,0)
    np.testing.assert_array_equal(a,receiver_noise(request,0))
    assert np.mean(abs(a)**2)==pytest.approx(.2,rel=.02)
    raw['receiver_noise']['variance']=1
    with pytest.raises(ValidationError,match='choose colored'):Simulation.model_validate(raw)
