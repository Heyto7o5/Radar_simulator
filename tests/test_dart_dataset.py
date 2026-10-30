import json
import h5py
import numpy as np
import pytest
from radar_platform.dart_dataset import aligned_raw_indices, magnitude_to_rd_power, load_dart_frame, FLU_TO_NED


def test_alignment_is_mask_then_valid():
    a, r = aligned_raw_indices(np.array([0,1,0,1,1],bool),np.array([0,1,1],bool))
    assert a.tolist()==[1,2] and r.tolist()==[3,4]
    with pytest.raises(ValueError):aligned_raw_indices(np.array([1,0]),np.array([True]))
    with pytest.raises(ValueError):aligned_raw_indices(np.array([True]),np.array([True,True]))


def test_power_is_not_average_amplitude():
    x=np.array([[[1.,3.]]])
    assert magnitude_to_rd_power(x).item()==5.
    for invalid in [np.array([[1.]]),np.array([[[-1.]]]),np.array([[[np.nan]]])]:
        with pytest.raises(ValueError):magnitude_to_rd_power(invalid)


def test_fixture_pose_and_observation_correspondence(tmp_path):
    (tmp_path/'sensor.json').write_text(json.dumps({'r':[1,2,2],'d':[-1,1,3]}))
    with h5py.File(tmp_path/'radar.h5','w') as f:
        f['rad']=np.arange(4)[:,None,None,None]*np.ones((4,2,3,8))
    with h5py.File(tmp_path/'trajectory.h5','w') as f:
        f['mask']=np.array([0,1,1,1],bool);f['valid']=np.array([0,1,1],bool)
        f['pos']=np.array([[0,0,0],[1,2,3],[2,3,4]])
        f['vel']=np.ones((3,3));f['rot']=np.tile(np.eye(3),(3,1,1));f['t']=[10,11,12]
    frame=load_dart_frame(tmp_path,0)
    assert frame.metadata['raw_frame_index']==2 and frame.timestamp_s==11
    assert np.all(frame.magnitude==2) and np.all(frame.rd_power==4)
    assert frame.position_ned_m.tolist()==[1,-2,-3]
    assert np.linalg.det(FLU_TO_NED)==1
    assert frame.metadata['velocity_sign_verified'] is False
    with pytest.raises(IndexError):load_dart_frame(tmp_path,2)
