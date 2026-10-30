"""Radar-local coordinates and explicit physical/virtual TDM channel layout."""
from dataclasses import dataclass
import numpy as np

from .schema import Platform, Array, Waveform


def sensor_basis(platform: Platform) -> np.ndarray:
    """Columns: forward, right, down in NED; valid at exact nadir."""
    forward = np.asarray(platform.boresight_ned, dtype=float)
    forward /= np.linalg.norm(forward)
    right = np.asarray(platform.right_hint_ned, dtype=float)
    right -= forward*np.dot(right, forward)
    right /= np.linalg.norm(right)
    return np.column_stack([forward, right, np.cross(forward, right)])


def scan_directions(azimuth_deg, elevation_deg):
    """Local unit directions [elevation,azimuth,xyz]; elevation positive up."""
    az, el = np.meshgrid(np.deg2rad(azimuth_deg), np.deg2rad(elevation_deg))
    return np.stack([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), -np.sin(el)], axis=-1)


def local_angles(direction_ned, basis):
    local = np.asarray(direction_ned)@basis
    return (np.rad2deg(np.arctan2(local[..., 1], local[..., 0])),
            np.rad2deg(np.arctan2(-local[..., 2], np.hypot(local[..., 0], local[..., 1]))))


@dataclass(frozen=True)
class Layout:
    tx_local_m: np.ndarray
    rx_local_m: np.ndarray
    tx_offsets_s: np.ndarray  # physical index, not chronological array order
    loop_period_s: float
    basis_ned: np.ndarray

    @property
    def tx_ned_m(self):
        return self.tx_local_m@self.basis_ned.T

    @property
    def rx_ned_m(self):
        return self.rx_local_m@self.basis_ned.T

    @property
    def virtual_local_m(self):
        return (self.tx_local_m[:, None, :]+self.rx_local_m[None, :, :]).reshape(-1, 3)

    def metadata(self):
        virtual = self.virtual_local_m
        spatial_rank = int(np.linalg.matrix_rank(virtual-virtual.mean(axis=0), tol=1e-10))
        unique = len(np.unique(np.round(virtual, decimals=10), axis=0))
        # Rank alone does not guarantee angle observability: endfire, scan plane,
        # aliasing and duplicate coordinates must be inspected too.
        return {'tx_local_m': self.tx_local_m.tolist(), 'rx_local_m': self.rx_local_m.tolist(),
                'virtual_local_m': virtual.tolist(), 'sensor_basis_columns_ned': self.basis_ned.tolist(),
                'tx_offsets_s': self.tx_offsets_s.tolist(), 'loop_period_s': self.loop_period_s,
                'chronological_tx_order': np.argsort(self.tx_offsets_s).tolist(),
                'channel_order': 'tx-major, rx-minor',
                'channel_map': [{'channel': t*len(self.rx_local_m)+r, 'tx': t, 'rx': r}
                                for t in range(len(self.tx_local_m)) for r in range(len(self.rx_local_m))],
                'virtual_aperture_rank': spatial_rank, 'unique_virtual_positions': unique,
                'two_dimensional_angle_geometry': spatial_rank >= 2,
                'angle_convention': 'sensor-local azimuth right-positive, elevation up-positive'}


def build_layout(array: Array, waveform: Waveform, platform: Platform) -> Layout:
    d = array.spacing_m or waveform.wavelength_m/2
    if array.layout == 'custom':
        tx, rx = np.array(array.tx_local_m), np.array(array.rx_local_m)
    else:
        tx, rx = np.zeros((array.num_tx, 3)), np.zeros((array.num_rx, 3))
        rx[:, 1] = (np.arange(array.num_rx)-(array.num_rx-1)/2)*d
        if array.layout == 'ura':
            tx[:, 2] = (np.arange(array.num_tx)-(array.num_tx-1)/2)*d
        else:
            tx[:, 1] = (np.arange(array.num_tx)-(array.num_tx-1)/2)*array.num_rx*d
    if array.tx_offsets_s is not None:
        offsets = np.array(array.tx_offsets_s, dtype=float)
    else:
        order = array.tx_order if array.tx_order is not None else list(range(array.num_tx))
        offsets = np.empty(array.num_tx)
        offsets[order] = np.arange(array.num_tx)*waveform.slot_period_s
    period = array.loop_period_s or array.num_tx*waveform.slot_period_s
    return Layout(tx, rx, offsets, period, sensor_basis(platform))
