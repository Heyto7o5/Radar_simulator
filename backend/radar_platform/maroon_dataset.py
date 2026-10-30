"""Adapter for the small MAROON-associated corner-reflector measurement.

File axes / indexing / optical x reflection follow the source dataset:
https://github.com/nihofm/polarized-radar-pathtracing
Only data conventions are reused; the forward solver is implemented locally.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .surface_scene import TriangleMesh, mesh_from_obj
from .surface_echo import SurfaceRadar

DEFAULT_ROOT=Path("/nas/datasets/hyt/datasets/radar_open/maroon_corner")
# Radar frame (x horizontal, y vertical, z toward scene) -> platform NED.
RADAR_TO_NED=np.array([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]])


def geometry_and_radar(root: Path = DEFAULT_ROOT, max_bounces: int = 3) -> tuple[TriangleMesh,SurfaceRadar,dict]:
    """Prepare forward inputs without opening signal.npz."""
    raw=Path(root)/"raw"
    alignment=np.asarray(json.loads((raw/"alignment.json").read_text())["photogrammetry2radar"],dtype=float)
    optical_x_flip=np.diag([-1.,1.,1.,1.])
    to_ned=np.eye(4); to_ned[:3,:3]=RADAR_TO_NED
    transform=to_ned@optical_x_flip@alignment
    mesh=mesh_from_obj(raw/"mesh_recreated.obj",transform)
    tx=np.loadtxt(raw/"Tx.dat",delimiter=",",skiprows=1)[1:95]@RADAR_TO_NED.T
    rx=np.loadtxt(raw/"Rx.dat",delimiter=",",skiprows=1)[1:95]@RADAR_TO_NED.T
    radar=SurfaceRadar(np.linspace(72e9,82e9,128),tx,rx,max_bounces)
    metadata={
        "coordinate_frame":"NED metres",
        "source_frame_to_ned":RADAR_TO_NED.tolist(),
        "optical_to_ned":transform.tolist(),
        "tx_rx_file_rows":"after CSV header, Python slice [1:95]",
        "source_photogrammetry_scale":float(np.linalg.norm(alignment[:3,0])),
        "has_slow_time":False,
        "waveform":"SFCW; frequency samples, not FMCW chirp samples",
        "nominal_frequencies_source":"source README: 72–82 GHz, 128 equally spaced steps",
    }
    return mesh,radar,metadata


def load_measurement(root: Path = DEFAULT_ROOT) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    with np.load(Path(root)/"raw/signal.npz",allow_pickle=False) as z:
        signal=np.asarray(z["signal"],dtype=np.complex64)
        f=np.asarray(z["fvec"],dtype=np.float64)
    if signal.shape!=(94,94,128) or f.shape!=(128,) or not np.iscomplexobj(signal):
        raise ValueError("unexpected MAROON [Tx,Rx,frequency] format")
    if not np.isfinite(signal).all() or not np.isfinite(f).all():
        raise ValueError("non-finite measured values")
    if not np.allclose(f,np.linspace(72e9,82e9,128),rtol=0,atol=1e-3):
        raise ValueError("measured frequencies differ from the independently declared config")
    valid=np.any(np.abs(signal)>0,axis=-1)
    return signal,f,valid
