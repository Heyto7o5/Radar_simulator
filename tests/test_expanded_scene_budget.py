import pytest
from pydantic import ValidationError
from radar_platform.unified.schema import Scene
from radar_platform.unified.scene import compile_scene


def test_expanded_budget_opt_in_and_default_guard_preserved():
    cloud={'kind':'cloud','points_local_m':[(10.,0.,0.)]*4096,'amplitudes':[1.]*4096}
    objects=[dict(cloud,id=f'part_{i}') for i in range(5)]
    assert Scene(objects=objects).max_scatterers==4096
    with pytest.raises(ValueError,match='max_scatterers'):compile_scene(Scene(objects=objects))
    scene=compile_scene(Scene(objects=objects,max_scatterers=32768))
    assert len(scene.amplitudes)==20480
    assert Scene(objects=objects,max_scatterers=131072).max_scatterers==131072
    with pytest.raises(ValidationError):Scene(objects=objects,max_scatterers=131073)
