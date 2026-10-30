import json
from pathlib import Path
import numpy as np
import pytest
import subprocess
import sys
import time
from fastapi.testclient import TestClient
from radar_platform.api import app
from radar_platform.unified.schema import Simulation
from radar_platform.unified.session import SimulationSession
from radar_platform.unified.engine import simulate_frame

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('config', [
    'unified/high_altitude_1ghz.json',
    'unified/lab_fmcw_mimo.json',
    'lab_moving_plate_po.json',
])
def test_same_session_different_configuration_matches_engine(config):
    request = Simulation.model_validate(json.loads((ROOT/'configs'/config).read_text()))
    session = SimulationSession(request)
    for i in range(request.sequence.frame_count):
        actual = session.frame(i)
        expected = simulate_frame(request, i)
        np.testing.assert_array_equal(actual.adc, expected.adc)
        np.testing.assert_array_equal(actual.products.rd_channels, expected.products.rd_channels)
        assert np.isfinite(actual.products.rd_power).all()
        assert np.max(actual.products.rd_power) > 0
    assert session.data_origin == 'ideal_simulation'
    with pytest.raises(ValueError):
        session.frame(request.sequence.frame_count)


def test_session_detaches_config_and_prepares_environment_once(monkeypatch):
    from radar_platform.unified import session as module
    request = Simulation.model_validate(json.loads((ROOT/'configs/unified/lab_fmcw_mimo.json').read_text()))
    raw = request.model_dump()
    raw['receiver_noise'] = {'variance': 0.01, 'seed': 7}
    request = Simulation.model_validate(raw)
    calls = []
    original = module.prepare_environment
    def prepare(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(module, 'prepare_environment', prepare)
    session = SimulationSession(request)
    request.scene.objects[0].amplitude = 99
    assert session.request.scene.objects[0].amplitude == 1
    first = session.frame(0)
    session.frame(1)
    np.testing.assert_array_equal(first.adc, session.frame(0).adc)
    assert len(calls) == 1
    assert session.data_origin == 'synthetic_composite_simulation'


def test_offline_and_api_produce_identical_complex_outputs(tmp_path, monkeypatch):
    config = ROOT/'configs/unified/lab_fmcw_mimo.json'
    output = tmp_path/'offline'
    subprocess.run([sys.executable, str(ROOT/'scripts/run_scene.py'),
                    '--config', str(config), '--output', str(output)],
                   cwd=ROOT, check=True, capture_output=True, timeout=30)
    monkeypatch.setenv('RADAR_RUNTIME_ROOT', str(tmp_path/'api'))
    with TestClient(app) as client:
        created = client.post('/api/v1/scene-simulations', json=json.loads(config.read_text()))
        assert created.status_code == 202
        job_id = created.json()['job_id']
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            status = client.get(f'/api/v1/scene-simulations/{job_id}').json()
            if status['status'] not in ('queued', 'running'):
                break
            time.sleep(.01)
        assert status['status'] == 'complete'
        for i in range(2):
            for product in ('adc', 'rd_channels', 'rad', 'rd_power', 'ra_power'):
                response = client.get(f'/api/v1/scene-simulations/{job_id}/frames/{i}/data/{product}')
                assert response.status_code == 200
                assert response.content == (output/f'frame_{i:04d}'/f'{product}.bin').read_bytes()
    manifest = json.loads((output/'run.json').read_text())
    assert manifest['status'] == 'complete'
    # Dependencies outside unified/ must be included in the reproducibility snapshot.
    assert 'backend/radar_platform/physical_optics.py' in manifest['source_sha256']
