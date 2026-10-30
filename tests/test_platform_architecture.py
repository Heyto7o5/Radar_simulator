import inspect
import subprocess
import sys
from pathlib import Path


def test_live_api_does_not_load_historical_forward_experiments():
    code="""import sys
import radar_platform.api
for name in ('ramp_comparison','legacy_simulator_reference','legacy_array_reference'):
    assert 'radar_platform.'+name not in sys.modules, name
"""
    subprocess.run([sys.executable,'-c',code],check=True,capture_output=True)


def test_compatibility_synthesis_entries_delegate_to_common_sessions():
    from radar_platform.simulator import simulate_frame
    from radar_platform.array_fmcw import simulate_motion_adc
    from radar_platform.fmcw import simulate_fmcw_if
    for fn in [simulate_frame,simulate_motion_adc,simulate_fmcw_if]:
        source=inspect.getsource(fn)
        assert 'Session' in source
        assert 'np.exp(' not in source


def test_common_dispatch_does_not_select_dataset_specific_simulators():
    from radar_platform.unified.session import session_from_config
    source=inspect.getsource(session_from_config).lower()
    for dataset in ['ramp','dart','maroon','ipix','maqu','tsms']:
        assert dataset not in source
    root=Path(__file__).resolve().parents[1]
    for file in ['engine.py','scene.py','processing.py']:
        text=(root/'backend/radar_platform/unified'/file).read_text()
        assert 'ramp_comparison' not in text
