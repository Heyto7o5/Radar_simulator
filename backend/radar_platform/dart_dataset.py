"""DART processed magnitude/pose adapter; never substitutes complex ADC or RCS.

Frame alignment follows WiseLabCMU/dart tools/dataset.py: rad[mask][valid].
FLU -> NED here chooses north=world front, east=-world left, down=-world up.
This is a coordinate relabeling, not a surveyed geographic north calibration.
"""
from dataclasses import dataclass
import json
from pathlib import Path

import h5py
import numpy as np

FLU_TO_NED = np.diag([1., -1., -1.])


def aligned_raw_indices(mask, valid):
    mask, valid = np.asarray(mask), np.asarray(valid)
    if mask.dtype != bool or valid.dtype != bool or mask.ndim != 1 or valid.ndim != 1:
        raise ValueError('alignment masks must be one-dimensional boolean arrays')
    if int(mask.sum()) != len(valid):
        raise ValueError('mask count does not match aligned trajectory length')
    aligned = np.flatnonzero(valid)
    return aligned, np.flatnonzero(mask)[aligned]


def magnitude_to_rd_power(magnitude):
    """Relative power: mean |RAD|^2 over angle, not mean magnitude or log."""
    values = np.asarray(magnitude, dtype=np.float64)
    if values.ndim != 3 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('expected finite nonnegative [range,doppler,angle] magnitude')
    return np.mean(values**2, axis=-1)


@dataclass
class DARTFrame:
    magnitude: np.ndarray
    range_m: np.ndarray
    published_velocity_mps: np.ndarray
    position_ned_m: np.ndarray
    velocity_ned_mps: np.ndarray
    sensor_flu_to_world_ned: np.ndarray
    timestamp_s: float
    metadata: dict

    @property
    def rd_power(self):
        return magnitude_to_rd_power(self.magnitude)


def load_dart_frame(directory, valid_frame_index):
    directory = Path(directory)
    if not isinstance(valid_frame_index, int) or valid_frame_index < 0:
        raise ValueError('valid_frame_index must be a nonnegative integer')
    sensor = json.loads((directory/'sensor.json').read_text())
    with h5py.File(directory/'trajectory.h5') as trajectory, h5py.File(directory/'radar.h5') as radar:
        mask, valid = trajectory['mask'][:], trajectory['valid'][:]
        aligned, raw = aligned_raw_indices(mask, valid)
        if valid_frame_index >= len(raw): raise IndexError('valid frame index out of bounds')
        if len(radar['rad']) != len(mask): raise ValueError('radar/mask length mismatch')
        ai, ri = int(aligned[valid_frame_index]), int(raw[valid_frame_index])
        magnitude = radar['rad'][ri].astype(np.float32)
        magnitude_to_rd_power(magnitude)
        axes = []
        for key, expected in zip(['r', 'd'], magnitude.shape[:2]):
            start, end, count = sensor[key]
            if count != expected or not np.isfinite([start, end]).all() or end <= start:
                raise ValueError('sensor axis does not match measured tensor')
            axes.append(np.linspace(start, end, int(count)))
        position, velocity, rotation = [trajectory[key][ai] for key in ['pos', 'vel', 'rot']]
        timestamp = float(trajectory['t'][ai])
        if position.shape != (3,) or velocity.shape != (3,) or rotation.shape != (3, 3):
            raise ValueError('invalid pose dimensions')
        if not all(np.isfinite(x).all() for x in [position, velocity, rotation, timestamp]):
            raise ValueError('non-finite pose')
        if not np.allclose(rotation.T@rotation, np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4):
            raise ValueError('invalid proper rotation')
    return DARTFrame(magnitude, *axes, FLU_TO_NED@position, FLU_TO_NED@velocity,
                     FLU_TO_NED@rotation, timestamp,
                     {'is_real': True, 'raw_frame_index': ri, 'trajectory_index': ai,
                      'valid_frame_index': valid_frame_index, 'representation': 'processed magnitude RAD',
                      'absolute_power_calibrated': False, 'complex_adc_available': False,
                      'velocity_sign_verified': False, 'geographic_north_verified': False,
                      'source': 'https://zenodo.org/records/10938617',
                      'angle_bins_are_not_raw_rx_channels': True})
