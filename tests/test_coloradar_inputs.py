import numpy as np
import pytest
from radar_platform.coloradar_inputs import load_source_poses,validate_adc_size


def test_pose_format_keeps_source_coordinates_and_requires_time_units(tmp_path):
    folder=tmp_path/'groundtruth';folder.mkdir()
    (folder/'timestamps.txt').write_text('1000\n2000\n')
    (folder/'groundtruth_poses.txt').write_text('1 2 3 0 0 0 1\n4 5 6 0 0 0 -1\n')
    p=load_source_poses(tmp_path,timestamp_scale_s=.001)
    np.testing.assert_array_equal(p.time_s,[1,2])
    np.testing.assert_array_equal(p.position_source_m,[[1,2,3],[4,5,6]])
    assert p.metadata['radar_extrinsics_applied'] is False
    with pytest.raises(ValueError):load_source_poses(tmp_path,timestamp_scale_s=0)


def test_bad_pose_and_nonmonotonic_time_are_rejected(tmp_path):
    folder=tmp_path/'groundtruth';folder.mkdir()
    (folder/'timestamps.txt').write_text('1\n1\n')
    (folder/'groundtruth_poses.txt').write_text('1 2 3 0 0 0 1\n4 5 6 0 0 0 1\n')
    with pytest.raises(ValueError,match='increase'):load_source_poses(tmp_path,timestamp_scale_s=1)
    (folder/'timestamps.txt').write_text('1\n2\n')
    (folder/'groundtruth_poses.txt').write_text('1 2 3 0 0 0 2\n4 5 6 0 0 0 1\n')
    with pytest.raises(ValueError,match='unit-length'):load_source_poses(tmp_path,timestamp_scale_s=1)


def test_adc_size_does_not_guess_shape_or_ignore_trailing_data(tmp_path):
    p=tmp_path/'frame.bin';p.write_bytes(bytes(4*8*4*2*3))
    result=validate_adc_size(p,samples=8,chirps=4,tx=2,rx=3)
    assert result['shape_inferred'] is False and result['complex_axis_order_verified'] is False
    with pytest.raises(ValueError,match='mismatch'):validate_adc_size(p,samples=8,chirps=4,tx=1,rx=3)
