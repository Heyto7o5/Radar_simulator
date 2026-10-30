import numpy as np
import pytest
from radar_platform.fertilizer_pilot import (terrain_landing, ideal_position_adjustment,
                                            RDACalibration, rda_features)
from radar_platform.unified.schema import Terrain


def ground(heights):
    return Terrain(kind='dem', north_m=[-100, 100], east_m=[-100, 100], elevation_up_m=heights)


def test_flat_ballistic_contact_and_position_adjustment():
    terrain = ground([[0, 0], [0, 0]])
    state = np.array([-10., 3., 4., -.5])
    shift = ideal_position_adjustment(state, observation_s=1, altitude_m=30, release_delay_s=.2)
    landing, t = terrain_landing(terrain, state[:2]+state[2:]+shift, state[2:], 30, .2)
    assert t == pytest.approx(np.sqrt(60/9.80665))
    np.testing.assert_allclose(landing, [0, 0], atol=1e-10)


def test_sloped_contact_matches_independent_quadratic():
    terrain = ground([[-10, 10], [-10, 10]])  # h = .1 north
    landing, t = terrain_landing(terrain, [2, 3], [4, 0], 30, .2)
    expected = (-.4+np.sqrt(.4**2+2*9.80665*(30-.28)))/9.80665
    assert t == pytest.approx(expected)
    assert landing[0] == pytest.approx(2.8+4*expected)


def test_rda_only_interface_and_gain_invariance():
    rng = np.random.default_rng(11)
    rda = rng.uniform(.1, 1, (2, 4, 4, 3)).astype('float32')
    np.testing.assert_allclose(rda_features(rda), rda_features(7*rda), atol=1e-7)
    state = [[-10, 0, 3, 0]]
    model = RDACalibration([rda_features(rda)], state)
    result, _ = model.predict(rda)
    np.testing.assert_allclose(result, state[0])
    with pytest.raises(ValueError, match='empty'):
        model.predict(np.zeros_like(rda))


def test_inference_cli_without_scene_or_truth_and_axis_mismatch(tmp_path):
    import json
    import subprocess
    import sys
    from pathlib import Path
    rda = np.ones((2, 4, 4, 3), dtype='float32')
    axes = dict(range_m=np.arange(4.), velocity_mps=np.arange(4.),
                azimuth_deg=np.arange(3.), frame_times_s=[0., 1.])
    model = tmp_path/'model.npz'
    observation = tmp_path/'observation.npz'
    output = tmp_path/'result.json'
    contract = dict(neighbors=1, altitude_m=30, observation_s=1, release_delay_s=.2,
                    scope='test-only fixed scene')
    np.savez(model, features=[rda_features(rda)], states=[[-10., 0., 3., 0.]],
             rda_shape=rda.shape, contract_json=np.array(json.dumps(contract)), **axes)
    np.savez(observation, magnitude=rda, **axes)
    script = Path(__file__).resolve().parents[1]/'scripts'/'predict_fertilizer_correction.py'
    command = [sys.executable, str(script), 'predict', '--model', str(model),
               '--rda', str(observation), '--output', str(output)]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    prediction = json.loads(output.read_text())
    assert prediction['estimated_velocity_ne_mps'] == [3, 0]
    axes['range_m'] = np.arange(4.)+1
    np.savez(observation, magnitude=rda, **axes)
    command[-1] = str(tmp_path/'bad_result.json')
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and 'range_m differs' in result.stderr
    assert not (tmp_path/'bad_result.json').exists()
