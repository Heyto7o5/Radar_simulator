from pathlib import Path
import numpy as np
import pytest
from scipy.io import netcdf_file
from fastapi.testclient import TestClient
from radar_platform.ipix_dataset import load_ipix,IQCorrection
from radar_platform.distributed_clutter import ClutterRecipe,simulate_clutter_iq,rd_blocks
from radar_platform.maqu_dataset import load_citi
from radar_platform.physical_optics import rectangular_plate,triangle_quadrature,simulate_po_surface,sfcw_range_profile,range_gate
from radar_platform.models import C0
from radar_platform.instrument_response import SmoothInstrumentResponse


def test_ipix_unsigned_polarization_and_per_pol_prf(tmp_path):
    p=tmp_path/'test.cdf'
    with netcdf_file(p,'w') as f:
        for name,size in [('nsweep',4),('ntxpol',2),('nrange',2),('nadc',4)]:f.createDimension(name,size)
        for name,value in {'PRF':1000,'RF_frequency':9.39,'Pulse_length':200,'radar_elev':30,'antenna_beamwidth':.9,
            'adc_like_I':0,'adc_like_Q':1,'adc_cross_I':2,'adc_cross_Q':3,
            'like_stc_RF':0,'like_stc_IF':.8,'cross_stc_RF':0,'cross_stc_IF':.8}.items():
            v=f.createVariable(name,'f8',());v.data[...]=value
        f.variables['PRF'].long_name='Radar pulse repetition frequency per polarization'
        f.createVariable('range','f8',('nrange',))[:]=[2000,2015]
        for n in ['azimuth_angle','elevation_angle']:f.createVariable(n,'f8',('nsweep',))[:]=0
        a=np.zeros((4,2,2,4),dtype=np.uint8);a[:,1,:,0]=240;a[:,1,:,1]=200;a[:,0,:,2]=210
        f.createVariable('adc_data','b',('nsweep','ntxpol','nrange','nadc'))[:]=a.view(np.int8)
    record=load_ipix(p,'vv')
    np.testing.assert_array_equal(record.iq,np.full((4,2),240+200j))
    assert record.prf_per_polarization_hz==1000 # do not divide by number of polarizations
    assert load_ipix(p,'hv').iq[0,0]==210+0j


def test_iq_correction_recovers_orthogonal_calibration_without_refitting():
    rng=np.random.default_rng(42);i=rng.normal(size=(20000,2));q=rng.normal(size=i.shape)
    observed=(128+3*i)+1j*(140+5*(.3*i+np.sqrt(1-.3**2)*q))
    correction=IQCorrection.fit(observed)
    corrected=correction.apply(observed)
    assert abs(corrected.real.mean())<1e-5
    assert abs(np.corrcoef(corrected.real.ravel(),corrected.imag.ravel())[0,1])<1e-5
    frozen=correction.apply(observed+10)
    assert frozen.real.mean()>1 # future offset is not silently re-estimated away


def recipe():return ClutterRecipe(9.39e9,1000.,200e-9,(2000.,2015.),(1.,2.),30.,8.,.001,1000.)


def test_statistical_iq_seed_and_doppler_axis():
    config=recipe();x=simulate_clutter_iq(config,32768,42);y=simulate_clutter_iq(config,32768,42)
    np.testing.assert_array_equal(x,y)
    assert not np.array_equal(x,simulate_clutter_iq(config,32768,43))
    rd,f=rd_blocks(x,1000)
    peak=f[np.argmax(np.mean(abs(rd)**2,axis=(0,2)))]
    assert abs(peak-30)<4
    assert rd.shape==(64,512,2)
    assert f[1]-f[0]==pytest.approx(1000/512)


def test_citi_polarization_parser_and_axis_checks(tmp_path):
    text=''
    for i,j in [(1,1),(1,2),(2,1),(2,2)]:
        text+=f'CITIFILE A.01.01\nVAR Freq MAG 2\nDATA S[{i},{j}] RI\nVAR_LIST_BEGIN\n1000000000\n2000000000\nVAR_LIST_END\nBEGIN\n{i},{j}\n2,3\nEND\n'
    p=tmp_path/'test.cti';p.write_text(text)
    f,s=load_citi(p);np.testing.assert_array_equal(f,[1e9,2e9]);assert s['12'][0]==1+2j
    p.write_text(text.replace('DATA S[1,2] RI','DATA S[1,2] DB'))
    with pytest.raises(ValueError,match='real/imaginary'):load_citi(p)


def test_po_area_convergence_reciprocity_and_far_field_spreading():
    plate=rectangular_plate(30,.85,.65);p,w,n=triangle_quadrature(plate,12)
    assert w.sum()==pytest.approx(.85*.65)
    f=np.linspace(9e9,10e9,20)
    a=simulate_po_surface(plate,f,order=12);b=simulate_po_surface(plate,f,order=24)
    np.testing.assert_allclose(a,b,rtol=1e-7,atol=1e-10)
    reverse=simulate_po_surface(plate,f,tx_ned_m=(0,.2,0),rx_ned_m=(0,-.2,0))
    np.testing.assert_allclose(b,reverse,rtol=1e-10)
    near=simulate_po_surface(rectangular_plate(100,.1,.1),f)
    far=simulate_po_surface(rectangular_plate(200,.1,.1),f)
    np.testing.assert_allclose(abs(near)/abs(far),4,rtol=1e-4)


def test_range_gate_retains_one_path_rejects_other():
    f=np.linspace(.75e9,10.25e9,3201)
    first=np.exp(-4j*np.pi*f*20/C0);second=np.exp(-4j*np.pi*f*25/C0)
    gated=range_gate(first+second,f,20,.65,.2)
    response,r=sfcw_range_profile(gated,f)
    assert abs(r[np.argmax(abs(response))]-20)<.01
    assert np.max(abs(response[abs(r-25)<.1])) < .02*np.max(abs(response))


def test_both_platform_api_paths(monkeypatch,tmp_path):
    from radar_platform.api import app
    from dataclasses import asdict
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    client=TestClient(app)
    req={'model':asdict(recipe()),'samples':1024,'seed':42}
    response=client.post('/api/v1/clutter-simulations',json=req)
    assert response.status_code==200
    iq=np.frombuffer(client.get(response.json()['iq_url']).content,dtype='<c8').reshape(1024,2)
    np.testing.assert_array_equal(iq,simulate_clutter_iq(recipe(),1024,42))
    plate=rectangular_plate(30,.85,.65);f=np.linspace(9e9,10e9,32)
    req={'method':'physical_optics','max_bounces':1,'frequencies_hz':f.tolist(),
        'vertices_ned_m':plate.vertices_m.tolist(),'faces':plate.faces.tolist(),'tx_ned_m':[[0,-.2,0]],'rx_ned_m':[[0,.2,0]]}
    response=client.post('/api/v1/surface-simulations',json=req);assert response.status_code==200
    spec=np.frombuffer(client.get(response.json()['spectrum_url']).content,dtype='<c8')
    np.testing.assert_allclose(spec,simulate_po_surface(plate,f).astype(np.complex64),rtol=1e-6)
    req['max_bounces']=3
    assert client.post('/api/v1/surface-simulations',json=req).status_code==422


def test_instrument_response_has_six_parameters_and_generalizes_frequency():
    f=np.linspace(1.5e9,9.875e9,120)
    truth=SmoothInstrumentResponse((.002,-.04,.2,-30.),(-22.,.7))
    signal=np.exp(-2j*np.pi*f*2/ C0)
    measured=signal*truth.evaluate(f)
    mask=np.arange(len(f))%2==0
    fit=SmoothInstrumentResponse.fit(f,measured,signal,mask)
    assert len(fit.amplitude_db_coefficients)+len(fit.phase_rad_coefficients)==6
    np.testing.assert_allclose(fit.evaluate(f[~mask]),truth.evaluate(f[~mask]),rtol=1e-9)
