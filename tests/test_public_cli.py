import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from fastapi.testclient import TestClient

from radar_platform.application import app
from radar_platform.capabilities import capabilities
from radar_platform.configuration import inspect_configuration
from radar_platform.runner import run_configuration

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('filename', [
    'unified/lab_fmcw_mimo.json', 'surface_plane.json',
    'unified/statistical_background.json', 'unified/block_texture_background.json',
    'unified/legacy_separable_if.json',
])
def test_public_check_covers_every_acquisition_family(filename, tmp_path):
    path = ROOT / 'configs' / filename
    raw = json.loads(path.read_text())
    expected = inspect_configuration(raw)
    assert expected['status'] in ('usable', 'risk')
    run = subprocess.run([sys.executable, '-m', 'radar_platform.cli', 'check', '--config', str(path)],
                         check=True, capture_output=True, text=True, timeout=30)
    assert json.loads(run.stdout) == expected
    with TestClient(app) as client:
        response = client.post('/api/v1/scene-check', json=raw)
        assert response.status_code == 200
        assert response.json() == expected


def test_capability_contract_shared_by_cli_api():
    run = subprocess.run([sys.executable, '-m', 'radar_platform.cli', 'capabilities'],
                         check=True, capture_output=True, text=True, timeout=30)
    assert json.loads(run.stdout) == capabilities()
    with TestClient(app) as client:
        assert client.get('/api/v1/capabilities').json() == capabilities()


def test_installed_runner_and_legacy_script_are_equivalent(tmp_path):
    config = ROOT / 'configs/unified/lab_fmcw_mimo.json'
    raw = json.loads(config.read_text())
    result = run_configuration(raw, tmp_path/'python')
    assert result['status'] == 'complete'
    subprocess.run([sys.executable, '-m', 'radar_platform.cli', 'run', '--config', str(config),
                    '--output', str(tmp_path/'cli')], check=True, capture_output=True, timeout=30)
    for path in (tmp_path/'python').glob('frame_*/*.bin'):
        assert path.read_bytes() == (tmp_path/'cli'/path.relative_to(tmp_path/'python')).read_bytes()
    with pytest.raises(FileExistsError):
        run_configuration(raw, tmp_path/'python')
    assert (tmp_path/'python/source.zip').is_file()


def test_failure_persists_readable_state_without_false_complete(tmp_path, monkeypatch):
    from radar_platform.unified.session import SimulationSession
    def fail(*args, **kwargs):
        raise RuntimeError('intentional execution failure')
    monkeypatch.setattr(SimulationSession, 'frame', fail)
    raw = json.loads((ROOT/'configs/unified/lab_fmcw_mimo.json').read_text())
    with pytest.raises(RuntimeError):
        run_configuration(raw, tmp_path/'failed')
    manifest = json.loads((tmp_path/'failed/run.json').read_text())
    assert manifest['status'] == 'failed'
    assert 'intentional execution failure' in manifest['error']
    assert not manifest['frames']


def test_bad_configuration_does_not_create_output(tmp_path):
    assert inspect_configuration({'waveform': 'unknown'})['status'] == 'unsupported'
    with pytest.raises(ValueError):
        run_configuration({'waveform': 'unknown'}, tmp_path/'invalid')
    assert not (tmp_path/'invalid').exists()
