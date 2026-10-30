import numpy as np
import pytest
import json
import subprocess
import sys
from pathlib import Path
from radar_platform.ramp_alignment import track_configuration,simulate_track_adc
from radar_platform.unified.schema import Simulation
from radar_platform.unified.session import session_from_config


def test_ramp_labels_are_complete_reusable_input_without_post_scaling():
    labels=[dict(uid='a',x_m=2.,y_m=10.,range_m=np.hypot(2,10),class_id=2,velocity_ned_mps=[-1,0,0])]
    config=track_configuration(labels)
    assert config.scene.amplitude_reference_range_m==10.
    assert config.scene.objects[0].position_ned_m==(10.,2.,0.)
    frame=session_from_config(config.model_dump(mode='json')).frame(0)
    np.testing.assert_array_equal(frame.adc,simulate_track_adc(labels))
    assert np.max(abs(frame.adc))>0


def test_no_annotations_are_explicit_empty_scene_not_invented_targets():
    config=track_configuration([])
    assert config.scene.allow_empty and config.scene.objects==[]
    frame=session_from_config(config.model_dump(mode='json')).frame(0)
    assert frame.adc.shape==(128,255,4,2)
    assert not np.any(frame.adc) and not np.any(frame.products.rd_power)
    raw=config.model_dump();raw['scene']['allow_empty']=False
    with pytest.raises(ValueError):Simulation.model_validate(raw)


def test_prepare_script_only_exports_reusable_config(tmp_path):
    root=Path(__file__).resolve().parents[1]
    labels=tmp_path/'labels.json';output=tmp_path/'scene.json'
    labels.write_text(json.dumps([dict(uid='p',class_id=2,x_m=1,y_m=8,velocity_ned_mps=[0,0,0])]))
    command=[sys.executable,str(root/'scripts/prepare_ramp_scene.py'),'--labels',str(labels),'--output',str(output)]
    result=subprocess.run(command,cwd=root,check=True,capture_output=True,text=True)
    assert not json.loads(result.stdout)['simulation_performed']
    config=Simulation.model_validate_json(output.read_text())
    assert any('SHA256' in note for note in config.input_provenance)
    assert any('not vehicle CAD' in note for note in config.input_provenance)
    prior=output.read_bytes()
    assert subprocess.run(command,cwd=root,capture_output=True).returncode!=0
    assert output.read_bytes()==prior
