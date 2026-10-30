"""Strict prerequisites from the official ColoRadar library's published format.

This does not infer radar calibration, frame conventions or ADC channel order.
Source: arpg/coloradar-library, revision b6cccb116036382746f78a7ddbb57eed78f0e55f,
src/run/coloradar_plus_run.cpp (readPoses/readTimestamps/readDatacube).
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class SourcePoses:
    time_s: np.ndarray
    position_source_m: np.ndarray
    quaternion_xyzw: np.ndarray
    metadata: dict


def load_source_poses(run_directory, *, timestamp_scale_s):
    if not np.isfinite(timestamp_scale_s) or timestamp_scale_s<=0:
        raise ValueError('explicit positive timestamp unit scale required')
    folder=Path(run_directory)/'groundtruth'
    times=np.loadtxt(folder/'timestamps.txt',ndmin=2)
    poses=np.loadtxt(folder/'groundtruth_poses.txt',ndmin=2)
    if times.shape[1]!=1 or poses.shape[1]!=7 or len(times)!=len(poses) or len(times)<2:
        raise ValueError('expected one timestamp and x y z qx qy qz qw per pose, with matching counts')
    if not np.isfinite(times).all() or not np.isfinite(poses).all():raise ValueError('non-finite pose data')
    t=times[:,0]*timestamp_scale_s
    if not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('pose times must increase strictly')
    quaternions=poses[:,3:];norms=np.linalg.norm(quaternions,axis=1)
    if np.any(abs(norms-1)>1e-3):raise ValueError('quaternions not unit-length; check format instead of guessing order')
    return SourcePoses(t,poses[:,:3].copy(),quaternions/norms[:,None],
        {'source_directory':str(folder),'timestamp_scale_s':timestamp_scale_s,
         'source_coordinate_frame_not_converted_to_ned':True,
         'radar_extrinsics_applied':False,'velocity_truth_provided':False,
         'quaternion_order':'xyzw','maximum_quaternion_norm_adjustment':float(abs(norms-1).max()),
         'scope':'format validation only; sequence-specific reference frame and sensor extrinsics still required'})


def validate_adc_size(path, *, samples, chirps, tx, rx):
    dimensions=(samples,chirps,tx,rx)
    if any(isinstance(n,bool) or not isinstance(n,(int,np.integer)) or n<=0 for n in dimensions):
        raise ValueError('positive integer dimensions from matching calibration required')
    expected=4*int(samples)*int(chirps)*int(tx)*int(rx)
    actual=Path(path).stat().st_size
    if actual!=expected:raise ValueError(f'ADC byte count mismatch: expected {expected}, got {actual}')
    return {'bytes':actual,'int16_values':expected//2,'shape_inferred':False,
            'complex_axis_order_verified':False,'scope':'int16 I/Q count only; no channel-order decoding or radar-config inference'}
