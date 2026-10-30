"""Bounded CPU jobs for ideal moving-target FMCW and azimuth processing."""
import asyncio
import json
import os
from pathlib import Path
import uuid
import numpy as np
from fastapi import APIRouter,HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel,ConfigDict,Field,model_validator
from .fmcw import FMCWConfig
from .array_fmcw import TDMArray
from .models import TargetConfig
from .unified.array_adapter import motion_array_configuration
from .unified.session import SimulationSession

router=APIRouter(prefix='/api/v1/fmcw-array-simulations',tags=['ideal moving-target FMCW / RAD'])


class FMCWSettings(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    start_hz: float=76.5e9
    stop_hz: float=77.5e9
    ramp_s: float=50e-6
    chirp_period_s: float=60e-6
    samples: int=Field(128,ge=4,le=1024)
    chirps: int=Field(64,ge=4,le=512)
    sample_rate_hz: float=4e6
    adc_start_s: float=2e-6

    def config(self):return FMCWConfig(**self.model_dump())


class ArraySimulationRequest(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    radar: FMCWSettings=Field(default_factory=FMCWSettings)
    targets: list[TargetConfig]=Field(min_length=1,max_length=32)
    platform_position_ned_m: tuple[float,float,float]=(0.,0.,0.)
    platform_velocity_ned_mps: tuple[float,float,float]=(0.,0.,0.)
    boresight_ned: tuple[float,float,float]=(1.,0.,0.)
    beam_half_angle_deg: float=Field(89.,gt=0,le=180)
    frame_start_s: float=Field(0.,ge=0)
    num_tx: int=Field(2,ge=1,le=2)
    num_rx: int=Field(4,ge=1,le=16)
    tx_ned_m: list[tuple[float,float,float]] | None=None
    rx_ned_m: list[tuple[float,float,float]] | None=None
    tx_offsets_s: list[float] | None=None
    loop_period_s: float | None=None
    azimuth_deg: list[float]=Field(default_factory=lambda:list(np.linspace(-80,80,161)),min_length=2,max_length=361)
    tdm_compensation: bool=True

    def array(self):
        if self.tx_ned_m is None:return TDMArray.ideal_ula(self.radar.config(),self.num_tx,self.num_rx)
        return TDMArray(self.tx_ned_m,self.rx_ned_m,self.tx_offsets_s,self.loop_period_s)

    @model_validator(mode='after')
    def check(self):
        self.radar.config()
        fields=(self.tx_ned_m,self.rx_ned_m,self.tx_offsets_s,self.loop_period_s)
        if any(v is not None for v in fields) and not all(v is not None for v in fields):
            raise ValueError('custom geometry requires Tx, Rx, offsets and loop period together')
        a=self.array()
        if self.tx_ned_m is not None and (len(a.tx_ned_m)!=self.num_tx or len(a.rx_ned_m)!=self.num_rx):
            raise ValueError('declared counts and custom geometry differ')
        if self.radar.samples*self.radar.chirps*self.num_tx*self.num_rx>5_000_000 or self.radar.samples*self.radar.chirps*len(self.azimuth_deg)>8_000_000:
            raise ValueError('request exceeds CPU cube limits')
        if np.linalg.norm(self.boresight_ned)<=0:raise ValueError('boresight must be nonzero')
        if len(set(t.id for t in self.targets))!=len(self.targets):raise ValueError('unique target IDs required')
        if np.any(np.diff(self.azimuth_deg)<=0) or np.any(abs(np.array(self.azimuth_deg))>90):raise ValueError('increasing azimuth grid in [-90,90] required')
        period=self.radar.chirp_period_s
        if a.loop_period_s-a.tx_offsets_s[-1]<period-1e-12 or np.any(np.diff(a.tx_offsets_s)<period-1e-12):raise ValueError('overlapping TDM chirps')
        return self


def run_array_request(request: ArraySimulationRequest):
    cfg,array=request.radar.config(),request.array()
    common=motion_array_configuration(cfg,request.targets,array,platform_position_ned_m=request.platform_position_ned_m,
        platform_velocity_ned_mps=request.platform_velocity_ned_mps,frame_start_s=request.frame_start_s,
        boresight_ned=request.boresight_ned,beam_half_angle_deg=request.beam_half_angle_deg,
        azimuth_deg=request.azimuth_deg,tdm_compensation=request.tdm_compensation)
    frame=SimulationSession(common).frame(0)
    adc,result=frame.adc,frame.products
    job=uuid.uuid4().hex;root=Path(os.environ.get('RADAR_RUNTIME_ROOT','.runtime'))/'array_jobs'/job
    root.mkdir(parents=True,exist_ok=False)
    # Legacy transport has no elevation dimension; retain its shape only.
    values={'adc':adc,'rd_channels':result.rd_channels,'rad':result.rad[:,:,0,:]}
    for name,value in values.items():np.asarray(value,dtype='<c8').tofile(root/(name+'.c64'))
    summary={'job_id':job,'status':'complete','is_real':False,'data_origin':'ideal_simulation',
        **result.metadata,'shapes':{k:list(v.shape) for k,v in values.items()},'dtype':'complex64_le','order':'C',
        'range_m':result.range_m.tolist(),'velocity_mps':result.velocity_mps.tolist(),'azimuth_deg':result.azimuth_deg.tolist(),
        'tx_ned_m':array.tx_ned_m.tolist(),'rx_ned_m':array.rx_ned_m.tolist(),
        'urls':{k:f'/api/v1/fmcw-array-simulations/{job}/data/{k}' for k in values},
        'coordinate_frame':'NED_m','velocity_convention':'positive closing',
        'dechirp':'tx * conj(rx), positive range beat, negative closing slow-frequency',
        'calibration':{'fitted_parameters':[],'source':'none, ideal relative amplitude'},
        'frame_start_s':request.frame_start_s}
    summary['rad_axes']=['range','closing_doppler','azimuth']
    summary['azimuth_coordinate_frame']='sensor_local_right_positive'
    summary['velocity_reference_frequency_hz']=common.waveform.carrier_hz
    summary['processing_contract']='unified_sampled_band_center_sensor_local_v1'
    summary['warnings']=frame.metadata['warnings']
    (root/'request.json').write_text(request.model_dump_json(indent=2))
    (root/'simulation_config.json').write_text(common.model_dump_json(indent=2))
    summary['simulation_engine']='unified'
    summary['simulation_config_file']='simulation_config.json'
    summary['compatibility_notes']=[
        'Common engine linearizes paths within a chirp and gates out-of-band beats; not bitwise identical to archived instantaneous-path solver.',
        'Processing now uses sampled-band center frequency and sensor-local azimuth, not the archived full-ramp center/world-NED azimuth convention.',
        'simulation_config.json reproduces ADC, complex RD and RAD through the common CLI; the API squeezes the singleton elevation axis only.'
    ]
    (root/'metadata.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    return summary


def job_root(job_id):
    try:canonical=uuid.UUID(job_id).hex
    except ValueError as exc:raise HTTPException(404,'unknown array job') from exc
    root=Path(os.environ.get('RADAR_RUNTIME_ROOT','.runtime'))/'array_jobs'/canonical
    if not (root/'metadata.json').exists():raise HTTPException(404,'unknown array job')
    return root


@router.post('')
async def create(request: ArraySimulationRequest):
    try:return await asyncio.to_thread(run_array_request,request)
    except ValueError as exc:raise HTTPException(422,str(exc)) from exc


@router.get('/{job_id}')
def status(job_id: str):return json.loads((job_root(job_id)/'metadata.json').read_text())


@router.get('/{job_id}/data/{name}')
def data(job_id: str,name: str):
    if name not in ('adc','rd_channels','rad'):raise HTTPException(404,'unknown cube')
    root=job_root(job_id);meta=json.loads((root/'metadata.json').read_text())
    return FileResponse(root/(name+'.c64'),media_type='application/octet-stream',headers={
        'X-Shape':','.join(map(str,meta['shapes'][name])),'X-Dtype':'complex64_le','X-Order':'C'})
