import inspect

import numpy as np
import pytest
torch = pytest.importorskip('torch', reason='install navigation extra for learning tests')

from radar_platform.navigation.io import read_json
from radar_platform.navigation_multidem.protocol import validate_regions, validate_records, tile_name
from radar_platform.navigation_multidem.data import records_for_plan
from radar_platform.navigation_multidem.model import MultiDEMRadarNet, RDAEstimator, encode_rda, FEATURE_SHAPE
from radar_platform.navigation_multidem.scene import rotated_footprint_heights
from radar_platform.unified.schema import Terrain


def plan():
    return read_json('configs/navigation_multidem_v1.json')


def test_whole_region_split_and_tile_overlap_rejection():
    p = plan()
    assert {k: len(v) for k, v in validate_regions(p).items()} == {'test': 2, 'train': 8, 'val': 2}
    rows = records_for_plan(p, train_per_region=2, val_per_region=2, test_per_region=2)
    rows[0]['split'] = 'test'
    with pytest.raises(ValueError, match='whole-region'):
        validate_records(rows, p)
    p['minimum_region_distance_km'] = 0
    p['regions'][-1].update(latitude_deg=48.7, longitude_deg=1.8)
    with pytest.raises(ValueError, match='tile shared'):
        validate_regions(p)
    assert tile_name(41.65, -4.65).endswith('N41_00_W005_00_DEM')
    assert tile_name(41.65, -4.65, worldcover=True) == 'N39W006'


def test_scene_variation_is_not_baked_into_model():
    rows = records_for_plan(plan(), train_per_region=2, val_per_region=2, test_per_region=2)
    assert len({r['building']['height_m'] for r in rows}) == len(rows)
    assert len({r['phase_seed'] for r in rows}) == len(rows)
    assert list(inspect.signature(MultiDEMRadarNet.forward).parameters) == ['self', 'rda_features']
    assert list(inspect.signature(RDAEstimator.predict).parameters) == ['self', 'rda_magnitude']
    model = MultiDEMRadarNet()
    assert set(dict(model.named_buffers())) == {'target_mean', 'target_std'}
    with pytest.raises(TypeError):
        model(torch.zeros(1, *FEATURE_SHAPE), roof_height_m=5)
    with pytest.raises(ValueError, match='sampling shape'):
        RDAEstimator(model).predict(np.ones((2, 12, 8, 5, 5), np.float32))


def test_rda_only_preprocessing_is_gain_invariant_and_rejects_invalid_input():
    rda = np.random.default_rng(2).uniform(.1, 2, size=(2, 12, 8, 5, 5)).astype('float32')
    features = encode_rda(rda)
    assert features.shape == FEATURE_SHAPE
    np.testing.assert_allclose(features, encode_rda(rda*17), rtol=1e-5, atol=1e-6)
    for bad in [rda*-1, np.full_like(rda, np.nan), np.zeros_like(rda), rda[:1]]:
        with pytest.raises(ValueError):
            encode_rda(bad)


def test_rotated_building_foundation_uses_actual_terrain():
    axis = np.linspace(-20, 20, 5)
    north, east = np.meshgrid(axis, axis)
    terrain = Terrain(kind='dem', north_m=axis.tolist(), east_m=axis.tolist(), elevation_up_m=(.2*north+.1*east).tolist())
    size = np.array([6., 10.])
    angle = np.deg2rad(37.)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]])*size/2@rotation.T
    heights = corners@np.array([.2, .1])
    low, high = rotated_footprint_heights(terrain, size, 37.)
    assert low == pytest.approx(heights.min())
    assert high == pytest.approx(heights.max())


def test_tensor_model_trains_without_geometry_or_motion_inputs():
    torch.set_num_threads(1)
    model = MultiDEMRadarNet()
    state, scale, visibility = model(torch.rand(1, *FEATURE_SHAPE))
    assert state.shape == scale.shape == (1, 6)
    assert visibility.shape == (1,)
    loss = state.square().mean()+scale.log().square().mean()+visibility.square().mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
