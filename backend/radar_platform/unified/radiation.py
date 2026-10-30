"""Scalar element patterns and coherent narrowband transmit subapertures.

Each existing logical Tx may feed a co-phased physical subaperture. This does
not turn TDM channels into simultaneous independent transmitters. Propagation
from its phase center is retained, with a guarded far-field array factor.
"""
from typing import Literal
import numpy as np
from pydantic import BaseModel,ConfigDict,Field,model_validator
from ..models import C0


class Pattern(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    kind: Literal['isotropic','cosine_power','axisymmetric_table']='isotropic'
    power_exponent: float=Field(1.,ge=0,le=128)
    angle_deg: list[float] | None=None
    power_db: list[float] | None=None
    frequency_min_hz: float | None=Field(None,gt=0)
    frequency_max_hz: float | None=Field(None,gt=0)
    evidence: Literal['analytic','assumed','measured']='analytic'
    source: str='Ideal scalar element pattern; no measured calibration'

    @model_validator(mode='after')
    def validate_table(self):
        if self.kind=='axisymmetric_table':
            if self.angle_deg is None or self.power_db is None:raise ValueError('pattern table requires angles and power dB')
            a,p=np.asarray(self.angle_deg),np.asarray(self.power_db)
            if not 2<=len(a)<=721 or p.shape!=a.shape or np.any(np.diff(a)<=0) or a[0]!=0 or a[-1]!=180 or np.any(abs(p)>160):
                raise ValueError('pattern needs full increasing 0..180 degree coverage and matching finite power values')
            if self.frequency_min_hz is None or self.frequency_max_hz is None or self.frequency_max_hz<=self.frequency_min_hz:
                raise ValueError('pattern table requires a declared frequency band')
            if not self.source.strip():raise ValueError('pattern provenance required')
        elif self.angle_deg is not None or self.power_db is not None:
            raise ValueError('table values require axisymmetric_table kind')
        return self


class ScanStep(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    time_s: float=Field(ge=0)
    azimuth_deg: float=Field(ge=-89,le=89)
    elevation_deg: float=Field(0.,ge=-89,le=89)


class TransmitAperture(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    positions_local_m: list[tuple[float,float,float]]=Field(min_length=1,max_length=1024)
    amplitudes: list[float] | None=None
    phases_rad: list[float] | None=None
    steering_azimuth_deg: float=Field(0.,ge=-89,le=89)
    steering_elevation_deg: float=Field(0.,ge=-89,le=89)
    scan_schedule: list[ScanStep]=Field(default_factory=list,max_length=2048)
    max_wideband_phase_error_rad: float=Field(np.pi/8,gt=0,le=np.pi/2)

    @model_validator(mode='after')
    def elements(self):
        n=len(self.positions_local_m)
        if np.linalg.norm(np.mean(self.positions_local_m,axis=0))>1e-9:
            raise ValueError('subaperture element coordinates must be centered at its logical Tx phase center')
        if self.amplitudes is not None and (len(self.amplitudes)!=n or min(self.amplitudes)<0 or not any(self.amplitudes)):
            raise ValueError('one nonnegative drive amplitude per element and nonzero total drive required')
        if self.phases_rad is not None and len(self.phases_rad)!=n:raise ValueError('one phase per transmit element required')
        if self.scan_schedule and np.any(np.diff([s.time_s for s in self.scan_schedule])<=0):
            raise ValueError('scan times must strictly increase on the global simulation clock')
        return self


class Radiation(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    tx_element: Pattern=Field(default_factory=Pattern)
    rx_element: Pattern=Field(default_factory=Pattern)
    transmit_aperture: TransmitAperture | None=None
    replace_legacy_cone: bool=True


def element_voltage(cosine,pattern):
    cosine=np.clip(np.asarray(cosine,float),-1,1)
    if pattern.kind=='isotropic':return np.ones_like(cosine)
    if pattern.kind=='cosine_power':return np.where(cosine>0,np.maximum(cosine,0)**(pattern.power_exponent/2),0.)
    angle=np.rad2deg(np.arccos(cosine))
    return 10**(np.interp(angle,pattern.angle_deg,pattern.power_db)/20)


def directions(azimuth,elevation):
    az,el=np.deg2rad(azimuth),np.deg2rad(elevation)
    return np.stack([np.cos(el)*np.cos(az),np.cos(el)*np.sin(az),-np.sin(el)],axis=-1)


def aperture_voltage(unit_local,times,aperture,carrier_hz):
    times=np.asarray(times);az=np.full(times.shape,aperture.steering_azimuth_deg);el=np.full(times.shape,aperture.steering_elevation_deg)
    for step in aperture.scan_schedule:
        az=np.where(times>=step.time_s,step.azimuth_deg,az);el=np.where(times>=step.time_s,step.elevation_deg,el)
    look=directions(az,el)
    positions=np.asarray(aperture.positions_local_m)
    amplitudes=np.ones(len(positions)) if aperture.amplitudes is None else np.asarray(aperture.amplitudes)
    amplitudes=amplitudes/np.max(amplitudes)  # Stable relative drive normalization, including very small/large inputs.
    phases=np.zeros(len(positions)) if aperture.phases_rad is None else np.asarray(aperture.phases_rad)
    phase=2*np.pi*carrier_hz/C0*np.einsum('...k,ak->...a',unit_local-look,positions)+phases
    return np.sum(amplitudes*np.exp(1j*phase),axis=-1)/np.linalg.norm(amplitudes)


def local_unit(delta,basis):
    distances=np.linalg.norm(delta,axis=-1)
    if np.any(distances<=1e-6):raise ValueError('pattern queried at antenna origin')
    unit=delta/distances[...,None]
    return (unit@basis if basis.ndim==2 else np.einsum('slj,ljk->slk',unit,basis)),distances


def transmit_response(delta,basis,times,spec,waveform,aperture_diameter_m=None):
    unit,distance=local_unit(delta,basis)
    gain=element_voltage(unit[...,0],spec.tx_element).astype(complex)
    aperture=spec.transmit_aperture
    if aperture is not None:
        points=np.asarray(aperture.positions_local_m)
        diameter=(float(np.linalg.norm(points[:,None]-points[None,:],axis=-1).max())
                  if aperture_diameter_m is None else aperture_diameter_m)
        if np.any(distance<2*diameter**2/waveform.wavelength_m):
            raise ValueError('transmit subaperture enters near field; far-field pattern model is unsupported here')
        gain*=aperture_voltage(unit,times,aperture,waveform.carrier_hz)
    return gain


def receive_response(delta,basis,spec):
    unit,_=local_unit(delta,basis)
    return element_voltage(unit[...,0],spec.rx_element)
