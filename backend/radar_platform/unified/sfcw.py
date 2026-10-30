"""Static stepped-frequency configuration and session, independent of HTTP.

The spectrum is not dynamic RD. Existing image-path and scalar-PO solvers are
physics kernels selected by configuration, not dataset-specific simulators.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import time
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ..surface_scene import TriangleMesh, mesh_from_dem
from ..surface_echo import SurfaceRadar, simulate_surface_spectrum
from ..physical_optics import simulate_po_surface
from .storage import write_json

class SurfaceSimulationRequest(BaseModel):
    model_config=ConfigDict(extra="forbid",allow_inf_nan=False)
    name: str = 'Static stepped-frequency surface'
    waveform: Literal["sfcw"]="sfcw"
    coordinate_frame: Literal["NED_m"]="NED_m"
    method: Literal["image_paths","physical_optics"]="image_paths"
    quadrature_order: int=Field(24,ge=2,le=48)
    frequencies_hz: list[float]=Field(min_length=2,max_length=4096)
    tx_ned_m: list[tuple[float,float,float]]=Field(min_length=1,max_length=128)
    rx_ned_m: list[tuple[float,float,float]]=Field(min_length=1,max_length=128)
    vertices_ned_m: list[tuple[float,float,float]] | None=None
    faces: list[tuple[int,int,int]] | None=None
    dem_north_m: list[float] | None=None
    dem_east_m: list[float] | None=None
    dem_elevation_up_m: list[list[float]] | None=None
    max_bounces: Literal[1,2,3]=3

    def mesh(self) -> TriangleMesh:
        if self.vertices_ned_m is not None and self.faces is not None:
            return TriangleMesh(np.asarray(self.vertices_ned_m),np.asarray(self.faces),"request_mesh")
        return mesh_from_dem(self.dem_north_m,self.dem_east_m,self.dem_elevation_up_m)

    def radar(self) -> SurfaceRadar:
        return SurfaceRadar(self.frequencies_hz,self.tx_ned_m,self.rx_ned_m,self.max_bounces)

    @model_validator(mode="after")
    def validate_scene(self) -> "SurfaceSimulationRequest":
        mesh_fields=(self.vertices_ned_m,self.faces)
        dem_fields=(self.dem_north_m,self.dem_east_m,self.dem_elevation_up_m)
        is_mesh=all(v is not None for v in mesh_fields) and all(v is None for v in dem_fields)
        is_dem=all(v is not None for v in dem_fields) and all(v is None for v in mesh_fields)
        if not (is_mesh or is_dem):
            raise ValueError("supply exactly one complete mesh or DEM input")
        mesh=self.mesh(); self.radar()
        n=len(mesh.faces)
        if n>32 or sum(n*(n-1)**(k-1) for k in range(1,self.max_bounces+1))>5000:
            raise ValueError("CPU reference solver supports <=32 faces and <=5000 sequences")
        if self.method=='physical_optics' and (self.max_bounces!=1 or len(self.tx_ned_m)*len(self.rx_ned_m)>8):
            raise ValueError('scalar PO reference supports a single reflection and at most 8 antenna pairs')
        return self

@dataclass
class SpectrumFrame:
    signal: np.ndarray
    metadata: dict


class SFCWSession:
    frame_count = 1
    data_origin = 'ideal_simulation'

    def __init__(self, request):
        self.request = SurfaceSimulationRequest.model_validate(request.model_dump(mode='json'))
        self.scene = self.request.mesh()
        self.radar = self.request.radar()
        self.estimate = {'estimated_output_bytes': len(self.radar.tx_ned_m)*len(self.radar.rx_ned_m)*len(self.radar.frequencies_hz)*8,
                         'triangles': len(self.scene.faces)}

    def frame(self, index, *, cancelled=None):
        if index != 0:
            raise ValueError('static SFCW has exactly one measurement, no slow-time sequence')
        from .engine import SimulationCancelled
        if cancelled is not None and cancelled():
            raise SimulationCancelled()
        start = time.perf_counter()
        if self.request.method == 'image_paths':
            result = simulate_surface_spectrum(self.scene, self.radar)
            signal = result.signal
            counts = {str(k): int(v.sum()) for k, v in result.path_counts.items()}
            method = 'scalar PEC finite-triangle image-path baseline'
        else:
            signal = np.stack([np.stack([simulate_po_surface(self.scene, self.radar.frequencies_hz, tx, rx,
                self.request.quadrature_order) for rx in self.radar.rx_ned_m]) for tx in self.radar.tx_ned_m])
            counts = None
            method = 'scalar PEC physical-optics surface integral; no multi-bounce or occlusion'
        if cancelled is not None and cancelled():
            raise SimulationCancelled()
        return SpectrumFrame(signal, {'frame_index': 0, 'elapsed_s': time.perf_counter()-start,
            'warnings': ['Static frequency response, not RD; no motion or Doppler axis.'],
            'has_doppler_axis': False, 'absolute_power_calibrated': False, 'method': method,
            'path_counts': counts, 'axes': {'frequency_hz': self.request.frequencies_hz},
            'spectrum_axes': ['tx', 'rx', 'frequency'], 'coordinate_frame': 'NED_m'})

    def save_inputs(self, root):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        write_json(root/'request.json', self.request.model_dump(mode='json'))
        write_json(root/'scene.json', {'coordinate_frame': 'NED_m', 'triangles': len(self.scene.faces),
                                     'static': True, 'has_doppler_axis': False})
        np.savez_compressed(root/'geometry.npz', vertices_ned_m=self.scene.vertices_m, faces=self.scene.faces)

    def save_frame(self, root, frame):
        directory = Path(root)/'frame_0000'
        directory.mkdir(parents=True, exist_ok=False)
        path = directory/'spectrum.bin'
        np.asarray(frame.signal, dtype='<c8', order='C').tofile(path)
        metadata = {**frame.metadata, 'products': {'spectrum': {'file': path.name,
            'shape': list(frame.signal.shape), 'dtype': '<c8', 'order': 'C', 'bytes': path.stat().st_size}}}
        write_json(directory/'metadata.json', metadata)
        return metadata
