import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('dart_metrics',Path(__file__).resolve().parents[1]/'scripts/dart_batch_metrics.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_frozen_gain_and_shared_reference_do_not_hide_amplitude_error():
    calibration=np.array([1.,2.]);sim=calibration/4;mask=np.ones(2,bool)
    gain=m.fit_power_gain(calibration,sim,mask)
    assert gain==4
    assert m.mae_db(np.ones(2),np.ones(2)*4,1.,mask)==pytest.approx(10*np.log10(4))
    assert m.mae_db(np.ones(2),np.ones(2)/4*gain,1.,mask)==0
    assert m.db_power(0,1)==-60
    with pytest.raises(ValueError):m.fit_power_gain(calibration,np.zeros(2),mask)
    with pytest.raises(ValueError):m.mae_db(calibration,sim,1.,np.zeros(2,bool))


def test_complex_noise_variance_seed_and_no_fake_floor():
    a=m.complex_white_noise((200000,),3.,42)
    assert np.mean(abs(a)**2)==pytest.approx(3.,rel=.02)
    assert np.array_equal(a,m.complex_white_noise((200000,),3.,42))
    assert not np.any(m.complex_white_noise((5,),0.,42))


def test_adc_noise_to_rover_power_scale_includes_eight_channels():
    from radar_platform.dart_processing import rover_magnitude
    variance=1.2
    magnitude=rover_magnitude(m.complex_white_noise((512,256,4,2),variance,42))
    expected=8*variance*np.sum(np.hanning(512)**2)*np.sum(np.hanning(256)**2)
    assert np.mean(magnitude**2)==pytest.approx(expected,rel=.02)


def test_gain_refit_does_not_change_reported_noise_realization():
    from radar_platform.colored_adc_background import colored_adc_noise
    q=np.linspace(.01,.2,32);old_gain=.4;new_gain=1.7
    first=colored_adc_noise((32,16,2,1),q,np.random.SeedSequence([11,32]))
    second=colored_adc_noise((32,16,2,1),q*old_gain/new_gain,np.random.SeedSequence([11,32]))
    np.testing.assert_allclose(np.sqrt(old_gain)*first,np.sqrt(new_gain)*second,rtol=2e-6,atol=1e-8)
