from dataclasses import asdict,replace
import numpy as np
import pytest
from fastapi.testclient import TestClient
from radar_platform.statistical_background import BackgroundRecipe,SpectrumComponent,gamma_texture,simulate_background,fit_background
from radar_platform.distributed_clutter import rd_blocks

def recipe(family='compound_gamma',white=.05):
    return BackgroundRecipe(1000.,(20.,35.),(1.,2.),(SpectrumComponent(1.,30.,10.),),white,2.,.1,
        ((1.,.35),(.35,1.)),((0.,.2),(-.2,0.)),family)

def test_covariance_and_parameters_are_validated():
    for kwargs in [{'texture_shape':float('nan')},{'family':'unknown'},{'white_fraction':2},
                   {'covariance_real':((1.,2.),(2.,1.))},{'covariance_imag':((0.,.2),(.2,0.))}]:
        with pytest.raises(ValueError):replace(recipe(),**kwargs)

def test_gamma_stationary_marginal_and_correlation_not_smoothed_away():
    x=gamma_texture(4096,128,2.,.02,1000.,np.random.default_rng(11))
    assert x.mean()==pytest.approx(1,rel=.035)
    assert x.var()==pytest.approx(.5,rel=.09)
    rho=np.mean((x[:-10]-1)*(x[10:]-1))/np.mean((x-1)**2)
    assert rho==pytest.approx(np.exp(-10/(1000*.02)),abs=.035)

@pytest.mark.parametrize('family',['white_gaussian','correlated_gaussian','compound_gamma'])
def test_seed_shape_finite_and_expected_ensemble_power(family):
    r=recipe(family);x=simulate_background(r,65536,7)
    np.testing.assert_array_equal(x,simulate_background(r,65536,7))
    assert x.shape==(65536,2) and np.isfinite(x).all() and np.iscomplexobj(x)
    assert not np.array_equal(x,simulate_background(r,65536,8))
    np.testing.assert_allclose(np.mean(abs(x)**2,axis=0),r.mean_power,rtol=.13)

def test_floor_is_independent_of_texture_and_not_scaled_by_it():
    a=simulate_background(recipe('correlated_gaussian',1.),8192,9)
    b=simulate_background(recipe('compound_gamma',1.),8192,9)
    np.testing.assert_array_equal(a,b)

def test_correlated_field_frequency_and_complex_covariance_sign():
    r=recipe('correlated_gaussian',0);x=simulate_background(r,131072,6)
    rd,f=rd_blocks(x,1000)
    peak=f[np.argmax(np.mean(abs(rd)**2,axis=(0,2)))]
    assert abs(peak-30)<4
    y=x/np.sqrt(r.mean_power);cov=y.T@y.conj()/len(y)
    np.testing.assert_allclose(cov,r.covariance,atol=.06)

def test_fitted_recipe_no_measured_template_and_calibration_mask_is_respected():
    r=recipe('correlated_gaussian',.1);x=simulate_background(r,32768,3)
    extended=np.c_[x,x[:,0]*100]
    fitted,meta=fit_background(extended,[20,35,50],1000,[True,True,False])
    assert len(fitted.components)==2
    assert fitted.mean_power[2]<5 # excluded large target not learned as background power
    np.testing.assert_allclose(fitted.covariance[2],[0,0,1])
    assert meta['domain']=='range_gated_iq' and meta['calibration_samples']==32768
    assert len(asdict(fitted))<15
    generated=simulate_background(fitted,8192,42)
    assert generated.shape==(8192,3)

def test_background_api_roundtrip_domain_and_invalid_config(monkeypatch,tmp_path):
    from radar_platform.api import app
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    with TestClient(app) as client:
        payload={'model':asdict(recipe()),'samples':1024,'seed':42}
        resp=client.post('/api/v1/background-simulations',json=payload)
        assert resp.status_code==200,resp.text
        meta=resp.json();assert meta['domain']=='range_gated_iq' and meta['is_real'] is False
        data=client.get(meta['rd_url']);a=np.frombuffer(data.content,dtype='<c8').reshape(meta['rd_shape'])
        assert a.shape==(2,512,2) and np.isfinite(a).all()
        payload['model']['white_fraction']=2
        assert client.post('/api/v1/background-simulations',json=payload).status_code==422
