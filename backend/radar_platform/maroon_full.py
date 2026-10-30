"""Strict full-MAROON frame adapter with an allow-listed NumPy pickle reader.

Official .npy files wrap numeric arrays in a dictionary. Ordinary unrestricted
allow_pickle=True is deliberately not used for externally supplied files.
"""
import io
import json
from pathlib import Path
import pickle
import numpy as np
from numpy.lib import format as npformat
from .maroon_dataset import RADAR_TO_NED,DEFAULT_ROOT
from .surface_scene import mesh_from_obj
from .surface_echo import SurfaceRadar


class _NumpyOnlyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype}
        for prefix in ('numpy.core.multiarray','numpy._core.multiarray'):
            allowed[(prefix,'_reconstruct')]=np._core.multiarray._reconstruct
            allowed[(prefix,'scalar')]=np._core.multiarray.scalar
        if (module,name) not in allowed:raise ValueError(f'forbidden pickle global: {module}.{name}')
        return allowed[(module,name)]

    def persistent_load(self,pid):raise ValueError('persistent pickle references forbidden')


def load_full_frame(path):
    path=Path(path)
    if path.stat().st_size>25_000_000:raise ValueError('full MAROON frame exceeds limit')
    with path.open('rb') as handle:
        version=npformat.read_magic(handle)
        shape,order,dtype=npformat._read_array_header(handle,version)
        if shape!=() or dtype!=np.dtype('O'):raise ValueError('expected official scalar dictionary NPY')
        wrapped=_NumpyOnlyUnpickler(handle).load()
    if not isinstance(wrapped,np.ndarray) or wrapped.shape!=():raise ValueError('invalid frame dictionary wrapper')
    data=wrapped.item()
    if not isinstance(data,dict) or not {'fvec','xr','timestamp_us'}.issubset(data):raise ValueError('missing measured axes/signal/timestamp')
    signal=np.asarray(data['xr']);f=np.asarray(data['fvec']).reshape(-1)
    if signal.shape!=(96,96,128) or not np.iscomplexobj(signal) or f.shape!=(128,):raise ValueError('unrecognized full radar dimensions')
    if not np.isfinite(signal).all() or not np.isfinite(f).all():raise ValueError('nonfinite measured frame')
    if not np.allclose(f,np.linspace(72e9,82e9,128),rtol=0,atol=1e-2):raise ValueError('full frame frequency axis mismatch')
    # Original xr is [rx,tx,frequency]; official prepare_dataset.py transposes
    # it before exporting the small [tx,rx,frequency] signal.npz sample.
    selected=np.ascontiguousarray(signal[1:95,1:95].transpose(1,0,2),dtype=np.complex64)
    return selected,f,{'timestamp_us':int(data['timestamp_us']),'source_shape':list(signal.shape),
                       'selected_channels':'source Tx and Rx indices [1:95]',
                       'source_axes':['rx','tx','frequency'],'output_axes':['tx','rx','frequency'],
                       'number_of_measurements':int(data.get('number_of_measurements',1)),
                       'has_verified_slow_time':False}


def full_scene_geometry(scene):
    scene=Path(scene)
    alignment=np.asarray(json.loads((scene/'alignment.json').read_text())['photogrammetry2radar'])
    to_ned=np.eye(4);to_ned[:3,:3]=RADAR_TO_NED
    transform=to_ned@np.diag([-1.,1.,1.,1.])@alignment
    mesh=mesh_from_obj(scene/'photogrammetry/mesh_synthetic.obj',transform)
    tx=np.loadtxt(DEFAULT_ROOT/'raw/Tx.dat',delimiter=',',skiprows=1)[1:95]@RADAR_TO_NED.T
    rx=np.loadtxt(DEFAULT_ROOT/'raw/Rx.dat',delimiter=',',skiprows=1)[1:95]@RADAR_TO_NED.T
    radar=SurfaceRadar(np.linspace(72e9,82e9,128),tx,rx,3)
    return mesh,radar
