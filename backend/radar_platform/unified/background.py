"""Configured range-gated stochastic acquisition, never raw fast-time ADC."""
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import time
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from ..statistical_background import BackgroundRecipe, simulate_background
from ..distributed_clutter import rd_blocks, ClutterRecipe, simulate_clutter_iq
from .storage import write_json


class BackgroundRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    kind: Literal['statistical_background']='statistical_background'
    name: str='Range-gated statistical background'
    model: BackgroundRecipe
    samples: int=Field(8192,ge=512,le=131072)
    seed: int=Field(42,ge=0,lt=2**32)
    pulses_per_cpi: int=Field(512,ge=4,le=2048)

    @model_validator(mode='after')
    def bounds(self):
        if self.samples*len(self.model.range_m)>5_000_000:
            raise ValueError('background sample budget exceeded')
        if self.samples<self.pulses_per_cpi:
            raise ValueError('need at least one CPI')
        return self


class ClutterSimulationRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    kind: Literal['block_texture_background']='block_texture_background'
    name: str='Historical block-texture background'
    model: ClutterRecipe
    samples: int=Field(65536,ge=512,le=131072)
    seed: int=Field(42,ge=0,lt=2**32)
    white_baseline: bool=False
    pulses_per_cpi: Literal[512]=512

    @model_validator(mode='after')
    def limit(self):
        if self.samples*len(self.model.mean_power)>5_000_000:
            raise ValueError('CPU IQ limit exceeded')
        return self


@dataclass
class BackgroundFrame:
    iq: np.ndarray
    rd: np.ndarray
    metadata: dict


class BackgroundSession:
    # One stationary acquisition batch, containing multiple explicitly indexed CPIs.
    frame_count=1
    data_origin='synthetic_statistical_background'

    def __init__(self,request):
        self.request=type(request).model_validate(request.model_dump(mode='json'))
        n,r=self.request.samples,len(self.request.model.mean_power)
        self.estimate={'estimated_output_bytes':2*n*r*8,'range_gates':r,
                       'cpi_count':n//self.request.pulses_per_cpi}

    def frame(self,index,*,cancelled=None):
        from .engine import SimulationCancelled
        if index!=0:raise ValueError('background session contains one stationary acquisition batch')
        if cancelled is not None and cancelled():raise SimulationCancelled()
        start=time.perf_counter();r=self.request
        legacy=isinstance(r,ClutterSimulationRequest)
        iq=(simulate_clutter_iq(r.model,r.samples,r.seed,r.white_baseline) if legacy
            else simulate_background(r.model,r.samples,r.seed))
        rd,frequency=rd_blocks(iq,r.model.prf_hz,r.pulses_per_cpi)
        if cancelled is not None and cancelled():raise SimulationCancelled()
        return BackgroundFrame(iq,rd,{'frame_index':0,'elapsed_s':time.perf_counter()-start,
            'domain':'range_gated_iq','family':self.family,'iq_axes':['slow_time','range'],
            'rd_axes':['cpi','raw_slow_frequency','range'],
            'axes':{('range_near_edge_m' if legacy else 'range_m'):list(r.model.range_near_edge_m if legacy else r.model.range_m),'raw_slow_frequency_hz':frequency.tolist(),
                    'cpi_start_s':(np.arange(len(rd))*r.pulses_per_cpi/r.model.prf_hz).tolist()},
            'unused_tail_samples':r.samples%r.pulses_per_cpi,
            'warnings':['Range-gated IQ, not ADC; do not add directly to fast-time ADC.',
                        'Raw slow-time frequency has no assumed closing-velocity sign.',
                        'Statistical background, not an exact sea/ground geometry replay.']+
                       (['Historical block texture, not the stationary CIR-Gamma model.'] if legacy else []),
            'receiver_noise_figure_calibrated':False,'absolute_power_calibrated':False})

    def save_inputs(self,root):
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        write_json(root/'request.json',self.request.model_dump(mode='json'))
        write_json(root/'scene.json',{'domain':'range_gated_iq','has_geometry':False,
                                     'family':self.family})

    @property
    def family(self):
        if isinstance(self.request,ClutterSimulationRequest):
            return 'legacy_white_baseline' if self.request.white_baseline else 'legacy_block_texture'
        return self.request.model.family

    def save_frame(self,root,frame):
        directory=Path(root)/'frame_0000';directory.mkdir(parents=True,exist_ok=False)
        products={}
        for name,array in [('iq',frame.iq),('rd',frame.rd)]:
            path=directory/(name+'.bin');np.asarray(array,dtype='<c8',order='C').tofile(path)
            products[name]={'file':path.name,'shape':list(array.shape),'dtype':'<c8','order':'C','bytes':path.stat().st_size}
        meta={**frame.metadata,'products':products};write_json(directory/'metadata.json',meta)
        return meta
