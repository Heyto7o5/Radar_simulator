import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from fastapi.testclient import TestClient
from radar_platform.api import app
from radar_platform.unified.session import session_from_config
from radar_platform.statistical_background import simulate_background
from radar_platform.distributed_clutter import rd_blocks

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('family',['white_gaussian','correlated_gaussian','compound_gamma'])
def test_background_common_entry_preserves_all_families(tmp_path,monkeypatch,family):
    raw=json.loads((ROOT/'configs/unified/statistical_background.json').read_text())
    raw['model']['family']=family
    session=session_from_config(raw);frame=session.frame(0)
    expected=simulate_background(session.request.model,1024,42)
    rd,_=rd_blocks(expected,1000,256)
    np.testing.assert_array_equal(frame.iq,expected)
    np.testing.assert_array_equal(frame.rd,rd)
    config=tmp_path/'input.json';config.write_text(json.dumps(raw));out=tmp_path/'out'
    subprocess.run([sys.executable,str(ROOT/'scripts/run_scene.py'),'--config',str(config),'--output',str(out)],
                   cwd=ROOT,check=True,capture_output=True,timeout=30)
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path/'api'))
    with TestClient(app) as client:
        created=client.post('/api/v1/scene-simulations',json=raw)
        assert created.status_code==202,created.text
        job=created.json()['job_id']
        assert created.json()['data_origin']=='synthetic_statistical_background'
        with client.websocket_connect(f'/api/v1/scene-simulations/{job}/events') as ws:
            while True:
                event=ws.receive_json()
                if event['type']=='job_completed':break
                if event['type']=='snapshot' and event['job']['status']=='complete':break
                assert event['type']!='job_failed',event
        legacy=client.post('/api/v1/background-simulations',json=raw)
        assert legacy.status_code==200,legacy.text
        for product,value in [('iq',expected),('rd',rd)]:
            binary=value.astype('<c8').tobytes()
            assert (out/'frame_0000'/(product+'.bin')).read_bytes()==binary
            assert client.get(f'/api/v1/scene-simulations/{job}/frames/0/data/{product}').content==binary
            assert client.get(legacy.json()[product+'_url']).content==binary
        assert client.get(f'/api/v1/scene-simulations/{job}/frames/0/data/adc').status_code==404
    assert frame.metadata['domain']=='range_gated_iq'
    assert frame.metadata['rd_axes']==['cpi','raw_slow_frequency','range']


def test_background_partial_cpi_reported_and_motion_rejected():
    raw=json.loads((ROOT/'configs/unified/statistical_background.json').read_text())
    raw['samples']=1031
    session=session_from_config(raw)
    frame=session.frame(0)
    assert frame.metadata['unused_tail_samples']==7
    assert frame.iq.shape[0]==1031 and frame.rd.shape[0]==4
    raw['platform']={'velocity_ned_mps':[1,0,0]}
    with pytest.raises(ValueError):session_from_config(raw)


@pytest.mark.parametrize('white',[False,True])
def test_block_texture_compatibility_preserves_model_and_all_cpis(tmp_path,monkeypatch,white):
    from radar_platform.distributed_clutter import simulate_clutter_iq
    from radar_platform.clutter_api import run_clutter_request
    raw=json.loads((ROOT/'configs/unified/block_texture_background.json').read_text())
    raw['white_baseline']=white
    session=session_from_config(raw);frame=session.frame(0)
    expected=simulate_clutter_iq(session.request.model,1024,42,white)
    np.testing.assert_array_equal(frame.iq,expected)
    expected_rd,_=rd_blocks(expected,1000)
    np.testing.assert_array_equal(frame.rd,expected_rd)
    assert frame.rd.shape[0]==2
    assert 'range_near_edge_m' in frame.metadata['axes']
    assert 'range_m' not in frame.metadata['axes']
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path))
    legacy=run_clutter_request(session.request)
    # Old transport exposes only the first CPI; common session preserves all.
    assert (tmp_path/'clutter_jobs'/legacy['job_id']/'rd.c64').read_bytes()==frame.rd[0].astype('<c8').tobytes()
    with TestClient(app) as client:
        response=client.post('/api/v1/scene-simulations',json=raw)
        assert response.status_code==202,response.text
        job=response.json()['job_id']
        with client.websocket_connect(f'/api/v1/scene-simulations/{job}/events') as ws:
            while True:
                event=ws.receive_json()
                if event['type']=='job_completed':break
                if event['type']=='snapshot' and event['job']['status']=='complete':break
                assert event['type']!='job_failed',event
        assert client.get(f'/api/v1/scene-simulations/{job}/frames/0/data/rd').content==frame.rd.astype('<c8').tobytes()
