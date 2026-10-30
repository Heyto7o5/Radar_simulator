import time
import numpy as np
import pytest
from fastapi.testclient import TestClient

from radar_platform.api import app
from radar_platform.unified.schema import Simulation
from radar_platform.unified.jobs import manager


def small_request():
    return Simulation.model_validate({'waveform':{'kind':'fmcw','samples':128,'sample_rate_hz':2e6,
        'slope_hz_s':3e10,'adc_start_s':40e-6,'ramp_s':150e-6,'slot_period_s':200e-6,'loops':16},
        'array':{'layout':'ula','num_tx':2,'num_rx':2},
        'scene':{'objects':[{'kind':'point','id':'p'}]},
        'processing':{'range_limits_m':[4600,5400],'azimuth_deg':[-10.,0.,10.]}}).model_dump(mode='json')


def test_unified_rest_websocket_and_complex_data(monkeypatch,tmp_path):
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    with TestClient(app) as client:
        presets=client.get('/api/v1/scene-presets').json()
        assert len(presets)==4
        created=client.post('/api/v1/scene-simulations',json=small_request())
        assert created.status_code==202
        job=created.json()['job_id']
        with client.websocket_connect(f'/api/v1/scene-simulations/{job}/events') as ws:
            first=ws.receive_json()
            assert first['type']=='snapshot'
            events=[]
            while True:
                message=ws.receive_json();events.append(message['type'])
                if message['type'] in ('job_completed','job_failed'):break
            assert 'job_completed' in events
        result=client.get(f'/api/v1/scene-simulations/{job}').json()
        assert result['status']=='complete' and len(result['frames'])==1
        for name,description in result['frames'][0]['products'].items():
            response=client.get(f'/api/v1/scene-simulations/{job}/frames/0/data/{name}')
            assert response.status_code==200
            value=np.frombuffer(response.content,dtype=description['dtype']).reshape(description['shape'])
            assert np.isfinite(value).all()
            if name in ('adc','rd_channels','rad'):assert np.iscomplexobj(value)
        assert len(manager.get(job).subscribers)==0
        assert client.get(f'/api/v1/scene-simulations/{job}/frames/-1/data/adc').status_code==404
        assert client.get(f'/api/v1/scene-simulations/{job}/frames/0/data/missing').status_code==404


def test_unified_invalid_input_returns_422(monkeypatch,tmp_path):
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    with TestClient(app) as client:
        data=small_request();data['array']['tx_offsets_s']=[0,1e-8]
        assert client.post('/api/v1/scene-simulations',json=data).status_code==422
        data=small_request();data['processing']['range_limits_m']=[1e8,2e8]
        assert client.post('/api/v1/scene-simulations',json=data).status_code==422
        data=small_request();data['scene']['objects'][0]['amplitude']=float('inf')
        with pytest.raises(ValueError):client.post('/api/v1/scene-simulations',json=data)


def test_unified_cancel_does_not_require_frontend(monkeypatch,tmp_path):
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    data=small_request();data['sequence']['frame_count']=100
    with TestClient(app) as client:
        created=client.post('/api/v1/scene-simulations',json=data)
        job=created.json()['job_id']
        assert client.delete(f'/api/v1/scene-simulations/{job}').status_code==200
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            result=client.get(f'/api/v1/scene-simulations/{job}').json()
            if result['status'] not in ('queued','running'):break
            time.sleep(.01)
        assert result['status']=='cancelled'
        assert len(result['frames'])<100


def test_unified_error_is_reported_not_empty_success(monkeypatch,tmp_path):
    import radar_platform.unified.engine as engine
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    def fail(*args,**kwargs):raise ValueError('intentional integration failure')
    monkeypatch.setattr(engine,'simulate_frame',fail)
    with TestClient(app) as client:
        job=client.post('/api/v1/scene-simulations',json=small_request()).json()['job_id']
        with client.websocket_connect(f'/api/v1/scene-simulations/{job}/events') as ws:
            while True:
                message=ws.receive_json()
                if message['type']=='job_failed':break
        result=client.get(f'/api/v1/scene-simulations/{job}').json()
        assert result['status']=='failed' and 'intentional' in result['error']
