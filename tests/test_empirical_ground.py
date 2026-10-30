from dataclasses import replace
import numpy as np
import pytest
from radar_platform.ground_scattering import Sigma0Observation,EmpiricalBackscatterCurve

def observations():
    return tuple(Sigma0Observation('synthetic_test',9e9,10e9,'vv',a,a-3,a+3,
        float(-10+2*10*np.log10(np.cos(np.deg2rad(a)))) ,1,-1) for a in [30,40,50,60])

def test_cosine_power_recovers_independent_analytic_law():
    m=EmpiricalBackscatterCurve(observations(),'cosine_power')
    angles=np.array([35,45,55])
    np.testing.assert_allclose(m.predict_db(9.4e9,angles,'vv'),-10+20*np.log10(np.cos(np.deg2rad(angles))),atol=1e-10)

def test_interpolation_and_patch_area_units():
    m=EmpiricalBackscatterCurve(observations())
    expected=(observations()[0].sigma0_db+observations()[1].sigma0_db)/2
    assert m.predict_db(9.4e9,35,'vv')==pytest.approx(expected)
    assert m.patch_rcs_m2(20,9.4e9,35,'vv')==pytest.approx(2*m.patch_rcs_m2(10,9.4e9,35,'vv'))

@pytest.mark.parametrize('f,a,p,c',[(1e9,40,'vv','alpine_meadow'),(9.4e9,0,'vv','alpine_meadow'),
                                  (9.4e9,40,'hh','alpine_meadow'),(9.4e9,40,'vv','sea')])
def test_no_silent_band_angle_polarization_or_land_cover_transfer(f,a,p,c):
    with pytest.raises(ValueError,match='extrapolation'):EmpiricalBackscatterCurve(observations()).predict_db(f,a,p,c)

def test_mixed_bands_and_duplicate_angles_rejected():
    x=observations()
    with pytest.raises(ValueError):EmpiricalBackscatterCurve((x[0],replace(x[1],polarization='hh')))
    with pytest.raises(ValueError):EmpiricalBackscatterCurve((x[0],x[0]))


def test_ground_response_api_keeps_coverage_and_prediction_boundary():
    from dataclasses import asdict
    from fastapi.testclient import TestClient
    from radar_platform.api import app
    request={'observations':[asdict(x) for x in observations()],'mode':'cosine_power',
             'carrier_hz':9.4e9,'incidence_deg':[35,45,55],'polarization':'vv','patch_area_m2':100.}
    with TestClient(app) as c:
        response=c.post('/api/v1/scattering/ground-response',json=request)
        assert response.status_code==200,response.text
        result=response.json();assert not result['is_real'] and result['not_a_dynamic_RD']
        assert len(result['sigma0_db'])==3
        request['incidence_deg']=[0.]
        assert c.post('/api/v1/scattering/ground-response',json=request).status_code==422
