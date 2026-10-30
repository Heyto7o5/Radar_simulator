import numpy as np
import pytest
from pydantic import ValidationError
from radar_platform.unified.antenna import two_way_voltage_gain
from radar_platform.unified.schema import Platform


def test_hard_default_and_two_way_power_convention():
    angles=np.array([0,60,62.5,65,66])
    cosine=np.cos(np.deg2rad(angles))
    assert two_way_voltage_gain(cosine,65).tolist()==[1,1,1,1,0]
    gain=two_way_voltage_gain(cosine,65,'cosine_edge',5)
    np.testing.assert_allclose(gain,[1,1,.5,0,0],atol=1e-12)
    assert gain[2]**2==pytest.approx(.25)
    with pytest.raises(ValidationError):Platform(beam_model='cosine_edge',beam_half_angle_deg=2,beam_taper_deg=5)


def test_continuity_and_reduced_far_spectral_leakage_when_crossing_edge():
    angle=np.linspace(55,70,1024)
    hard=two_way_voltage_gain(np.cos(np.deg2rad(angle)),65)
    smooth=two_way_voltage_gain(np.cos(np.deg2rad(angle)),65,'cosine_edge',5)
    assert np.max(abs(np.diff(smooth)))<.005
    def leakage(envelope):
        power=abs(np.fft.fft(envelope*np.hanning(len(envelope))))**2
        freq=np.fft.fftfreq(len(envelope))
        return power[abs(freq)>.1].sum()/power.sum()
    assert leakage(smooth)<leakage(hard)*.01
