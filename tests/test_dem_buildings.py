import numpy as np
import pytest

from radar_platform.dem_buildings import BuildingSpec, add_dem_buildings, footprint_heights
from radar_platform.unified.schema import Simulation, Terrain
from radar_platform.unified.scene import object_mesh


def terrain(h):
    return Terrain(kind='dem', north_m=[0, 10, 20], east_m=[0, 10, 20], elevation_up_m=h)


def test_footprint_extrema_include_interior_peak():
    t = terrain([[0, 0, 0], [0, 8, 0], [0, 0, 0]])
    assert footprint_heights(t, [10, 10], [20, 20]) == (0, 8)
    # Clipped corners on the same diagonal must use triangular interpolation.
    assert footprint_heights(t, [2, 2], [2, 2]) == pytest.approx((.8, 2.4))


def test_sloped_foundation_ned_geometry_and_input_preserved():
    t = terrain([[0, 1, 2], [2, 3, 4], [4, 5, 6]])
    cfg = Simulation(scene=dict(terrain=t))
    before = cfg.model_dump()
    spec = BuildingSpec(id='pump', center_ne_m=(10, 10), footprint_ne_m=(10, 10),
                        height_above_highest_ground_m=4, max_foundation_relief_m=3,
                        relative_reflectivity=.03)
    result, report = add_dem_buildings(cfg, [spec])
    mesh = object_mesh(result.scene.objects[0])
    assert -mesh.vertices_m[:, 2].max() == pytest.approx(1.5)
    assert -mesh.vertices_m[:, 2].min() == pytest.approx(8.5)
    assert report[0]['foundation_relief_m'] == pytest.approx(3)
    assert cfg.model_dump() == before
    assert result.scene.terrain == t
    assert result.waveform == cfg.waveform and result.platform == cfg.platform
    assert result.visibility.occlusion_enabled and result.visibility.occluders == 'all_surfaces'
    with pytest.raises(ValueError, match='relief'):
        add_dem_buildings(cfg, [spec.model_copy(update=dict(max_foundation_relief_m=2))])
    with pytest.raises(ValueError, match='overlap'):
        add_dem_buildings(cfg, [spec, spec.model_copy(update=dict(id='second'))])
    with pytest.raises(ValueError, match='unique'):
        add_dem_buildings(result, [spec])


@pytest.mark.parametrize('center,size', [([0, 0], [2, 2]), ([10, 10], [-1, 2]), ([np.nan, 5], [2, 2])])
def test_invalid_footprints_rejected(center, size):
    with pytest.raises(ValueError):
        footprint_heights(terrain(np.zeros((3, 3)).tolist()), center, size)
