"""Explicit historical IF approximation in the common configuration workflow."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import time
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ..fmcw import FMCWConfig, FMCWPointTarget, process_fmcw_rd
from .separable_kernel import synthesize_separable_if
from .storage import write_json


class SeparableIFRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    kind: Literal['legacy_separable_if']='legacy_separable_if'
    name: str='Historical fixed-range separable FMCW approximation'
    waveform: FMCWConfig
    targets: list[FMCWPointTarget]=Field(min_length=1,max_length=128)
    receiver_positions_m: list[float]=Field(default_factory=lambda:[0.],min_length=1,max_length=32)
    real_output: bool=True
    noise_std: float=Field(0.,ge=0)
    seed: int=Field(42,ge=0,lt=2**32)

    @model_validator(mode='after')
    def budget(self):
        if self.waveform.samples*self.waveform.chirps*len(self.receiver_positions_m)>8_000_000:
            raise ValueError('IF cube exceeds 8 million samples')
        for t in self.targets:
            if not np.isfinite([t.range_m,t.radial_velocity_mps,t.amplitude,t.phase_rad,t.azimuth_deg]).all():
                raise ValueError('target parameters must be finite')
        return self


@dataclass
class SeparableIFFrame:
    if_cube: np.ndarray
    rd: np.ndarray
    metadata: dict


class SeparableIFSession:
    frame_count=1
    data_origin='synthetic_legacy_approximation'

    def __init__(self,request):
        self.request=SeparableIFRequest.model_validate(request.model_dump(mode='json'))
        w=self.request.waveform
        self.estimate={'estimated_output_bytes':16*w.samples*w.chirps*len(self.request.receiver_positions_m)}

    def synthesize(self):
        r=self.request
        return synthesize_separable_if(r.waveform,r.targets,r.receiver_positions_m,
            real_output=r.real_output,noise_std=r.noise_std,seed=r.seed)

    def frame(self,index,*,cancelled=None):
        from .engine import SimulationCancelled
        if index!=0:raise ValueError('separable IF is one fixed-range cube, not a continuous motion sequence')
        if cancelled is not None and cancelled():raise SimulationCancelled()
        start=time.perf_counter();cube=self.synthesize();w=self.request.waveform
        result=process_fmcw_rd(cube,w,range_fft_size=w.samples,doppler_fft_size=w.chirps)
        if cancelled is not None and cancelled():raise SimulationCancelled()
        return SeparableIFFrame(cube,result.rd,{'frame_index':0,'elapsed_s':time.perf_counter()-start,
            'if_axes':['sample','chirp','rx'],'rd_axes':['range','legacy_closing_doppler','rx'],
            'axes':{'range_m':result.range_axis_m.tolist(),'velocity_mps':result.velocity_axis_mps.tolist()},
            'warnings':['Historical fixed-range separable phase approximation, not moving-path tx*conj(rx).',
                        'No TDM-MIMO or cross-frame motion continuity; use standard FMCW for new physical simulation.',
                        'Real IF retains both FFT branches; full range axis includes the mirror branch.'],
            'processing':{'range_window':'rectangular','doppler_window':'rectangular'},
            'absolute_power_calibrated':False})

    def save_inputs(self,root):
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        write_json(root/'request.json',self.request.model_dump(mode='json'))
        write_json(root/'scene.json',{'model':'legacy_separable_if','targets':[vars(t) for t in self.request.targets]})

    def save_frame(self,root,frame):
        directory=Path(root)/'frame_0000';directory.mkdir(parents=True,exist_ok=False)
        products={}
        for name,array in [('if',frame.if_cube),('rd',frame.rd)]:
            dtype='<c8' if np.iscomplexobj(array) else '<f4'
            path=directory/(name+'.bin');np.asarray(array,dtype=dtype,order='C').tofile(path)
            products[name]={'file':path.name,'shape':list(array.shape),'dtype':dtype,'order':'C','bytes':path.stat().st_size}
        meta={**frame.metadata,'products':products};write_json(directory/'metadata.json',meta)
        return meta
