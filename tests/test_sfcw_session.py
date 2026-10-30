import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from fastapi.testclient import TestClient
from radar_platform.api import app
from radar_platform.unified.session import session_from_config
from radar_platform.surface_api import run_surface_request

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('method',['image_paths','physical_optics'])
def test_static_common_cli_api_and_legacy_transport_are_identical(tmp_path,monkeypatch,method):
    raw=json.loads((ROOT/'configs/surface_plane.json').read_text())
    raw['method']=method
    config=tmp_path/'input.json';config.write_text(json.dumps(raw))
    output=tmp_path/'offline'
    subprocess.run([sys.executable,str(ROOT/'scripts/run_scene.py'),'--config',str(config),'--output',str(output)],
                   cwd=ROOT,check=True,capture_output=True,timeout=30)
    session=session_from_config(raw)
    expected=session.frame(0).signal.astype('<c8').tobytes()
    assert (output/'frame_0000/spectrum.bin').read_bytes()==expected
    manifest=json.loads((output/'frame_0000/metadata.json').read_text())
    assert not manifest['has_doppler_axis']
    assert set(manifest['products'])=={'spectrum'}
    monkeypatch.setenv('RADAR_RUNTIME_ROOT',str(tmp_path/'api'))
    with TestClient(app) as client:
        created=client.post('/api/v1/scene-simulations',json=raw)
        assert created.status_code==202,created.text
        job=created.json()['job_id']
        with client.websocket_connect(f'/api/v1/scene-simulations/{job}/events') as ws:
            while True:
                event=ws.receive_json()
                if event['type']=='job_completed':break
                if event['type']=='snapshot' and event['job']['status']=='complete':break
                assert event['type']!='job_failed',event
        response=client.get(f'/api/v1/scene-simulations/{job}/frames/0/data/spectrum')
        assert response.content==expected
        assert client.get(f'/api/v1/scene-simulations/{job}/frames/0/data/rd_channels').status_code==404
    legacy=run_surface_request(session.request)
    assert (tmp_path/'api/surface_jobs'/legacy['job_id']/'spectrum.c64').read_bytes()==expected
    with pytest.raises(ValueError):session.frame(1)


def test_static_config_rejects_motion_and_invalid_model():
    raw=json.loads((ROOT/'configs/surface_plane.json').read_text())
    raw['velocity_ned_mps']=[1,0,0]
    with pytest.raises(ValueError):session_from_config(raw)
    del raw['velocity_ned_mps']
    raw.update(method='physical_optics',max_bounces=3)
    with pytest.raises(ValueError):session_from_config(raw)


def test_static_three_reflections_remain_available_through_common_session():
    raw=dict(waveform='sfcw',frequencies_hz=[8e9,9e9,10e9],
             tx_ned_m=[[1,2,3]],rx_ned_m=[[3,2,1]],
             vertices_ned_m=[[0,0,0],[10,0,0],[0,10,0],[0,0,10]],
             faces=[[0,2,3],[0,1,3],[0,1,2]],max_bounces=3)
    session=session_from_config(raw)
    frame=session.frame(0)
    assert frame.metadata['path_counts']['3']>0
    from radar_platform.surface_echo import simulate_surface_spectrum
    expected=simulate_surface_spectrum(session.scene,session.radar)
    np.testing.assert_array_equal(frame.signal,expected.signal)
