import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('phase_probe',Path(__file__).resolve().parents[1]/'scripts/phase_recurrence_probe.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


@pytest.mark.parametrize('kind',['fmcw','pulsed_lfm'])
def test_recurrence_matches_direct_with_fractional_delay_and_both_rates(kind):
    rng=np.random.default_rng(8)
    delay=rng.uniform(1e-8,4e-5,(3,9));rate=rng.uniform(-1e-5,1e-5,(3,9));phase=rng.uniform(-np.pi,np.pi,(3,9))
    fc=77e9;slope=5e12;start=5e-5;step=1/20e6;n=1024;width=5e-6
    abc=m.coefficients(kind,delay,rate,fc,slope,width,phase)
    actual=m.quadratic_phasor(*abc,start,step,n)
    expected=m.direct(kind,delay,rate,fc,slope,start,step,n,width,phase)
    np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-8)


def test_constant_phase_single_sample_and_input_guard():
    z=np.zeros((2,3));a=np.ones((2,3))*.3
    np.testing.assert_allclose(m.quadratic_phasor(a,z,z,1.,.2,1),np.exp(.3j))
    with pytest.raises(ValueError):m.quadratic_phasor(a,z,z,0.,0.,10)
