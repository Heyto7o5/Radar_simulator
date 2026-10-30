import json
from pathlib import Path
import h5py
import numpy as np
import pytest
from radar_platform.dart_scene_adapter import prepare_scene


def inputs(tmp_path):
    geometry=tmp_path/'mesh.npz'
    np.savez(geometry,vertices_flu_m=[[1,-1,0],[1,1,0],[1,0,1]],faces=[[0,1,2]])
    trajectory=tmp_path/'trajectory.h5'
    t=np.arange(11)*.1
    with h5py.File(trajectory,'w') as f:
        for name,value in dict(t=t,pos=np.zeros((11,3)),vel=np.zeros((11,3)),
            rot=np.tile(np.eye(3),(11,1,1)),valid=np.ones(11,dtype=bool),mask=np.ones(11,dtype=bool)).items():f[name]=value
    config=tmp_path/'radar.json'
    root=Path(__file__).resolve().parents[1]
    sample=json.loads((root/'configs/unified/lab_fmcw_mimo.json').read_text())
    config.write_text(json.dumps({'waveform':sample['waveform'],'array_timing_candidate':sample['array']}))
    return geometry,trajectory,config


def test_preparation_uses_only_geometry_poses_and_radar_config(tmp_path):
    paths=inputs(tmp_path)
    request=prepare_scene(*paths,valid_frame_index=5)
    assert request.platform.trajectory is not None
    assert request.scene.objects[0].vertices_local_m[0]==(1.,1.,0.)
    assert request.platform.beam_model=='cosine_edge'
    assert sum('SHA256' in x for x in request.input_provenance)==3
    assert not list(tmp_path.glob('*.bin'))
    with pytest.raises(ValueError,match='out of bounds'):prepare_scene(*paths,valid_frame_index=99)


def test_preparation_constant_motion_and_hard_beam_are_explicit(tmp_path):
    request=prepare_scene(*inputs(tmp_path),valid_frame_index=5,sampled_trajectory=False,smooth_beam=False)
    assert request.platform.trajectory is None
    assert request.platform.beam_model=='hard_cone'
