import json
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient
from radar_platform.api import app
from radar_platform.unified.session import session_from_config
from radar_platform.fmcw import simulate_fmcw_if

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('real',[False,True])
def test_reference_configuration_matches_analytic_formula_and_compatibility(real,tmp_path,monkeypatch):
    raw=json.loads((ROOT/'configs/unified/legacy_separable_if.json').read_text());raw['real_output']=real
    session=session_from_config(raw);r=session.request;w=r.waveform;t=r.targets[0]
    fast=w.adc_start_s+np.arange(w.samples)/w.sample_rate_hz
    slow=np.arange(w.chirps)*w.chirp_period_s
    fd=2*t.radial_velocity_mps/w.wavelength_m
    beat=2*w.slope_hz_s*t.range_m/299792458.+fd
    phase=2*np.pi*(beat*fast[:,None]+fd*slow[None,:])
    expected=(np.cos(phase) if real else np.exp(1j*phase)).astype(np.float32 if real else np.complex64)[:,:,None]
    frame=session.frame(0)
    np.testing.assert_array_equal(frame.if_cube,expected)
    np.testing.assert_array_equal(frame.if_cube,simulate_fmcw_if(w,r.targets,real_output=real))
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    with TestClient(app) as client:
        response=client.post('/api/v1/scene-simulations',json=raw)
        assert response.status_code==202,response.text
        job=response.json()['job_id']
        assert response.json()['data_origin']=='synthetic_legacy_approximation'
        with client.websocket_connect(f'/api/v1/scene-simulations/{job}/events') as ws:
            while True:
                event=ws.receive_json()
                if event['type']=='job_completed':break
                if event['type']=='snapshot' and event['job']['status']=='complete':break
                assert event['type']!='job_failed',event
        assert client.get(f'/api/v1/scene-simulations/{job}/frames/0/data/if').content==expected.tobytes()
    with pytest.raises(ValueError):session.frame(1)
