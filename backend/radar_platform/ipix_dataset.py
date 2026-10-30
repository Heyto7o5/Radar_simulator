"""Original IPIX range-gated receiver samples, not a full fast-time ADC cube."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from scipy.io import netcdf_file


@dataclass
class IPIXRecord:
    iq: np.ndarray
    range_near_edge_m: np.ndarray
    carrier_hz: float
    prf_per_polarization_hz: float
    pulse_width_s: float
    polarization: str
    metadata: dict


def load_ipix(path: str|Path,polarization: str='vv') -> IPIXRecord:
    if polarization not in ('hh','hv','vv','vh'):raise ValueError('unknown polarization')
    with netcdf_file(path,mmap=False) as file:
        def scalar(name):return float(file.variables[name].data)
        variable=file.variables['adc_data']
        if variable.dimensions!=('nsweep','ntxpol','nrange','nadc'):raise ValueError('unexpected IPIX axes')
        tx=0 if polarization[0]=='h' else 1
        kind='like' if polarization[0]==polarization[1] else 'cross'
        ii=int(scalar(f'adc_{kind}_I'));qi=int(scalar(f'adc_{kind}_Q'))
        # Official ipixload.m explicitly repairs unsigned-byte interpretation.
        raw=np.asarray(variable.data[:,tx],dtype=np.uint8)
        iq=raw[:,:,ii].astype(np.float32)+1j*raw[:,:,qi].astype(np.float32)
        ranges=np.asarray(file.variables['range'].data,dtype=float).copy()
        prf=scalar('PRF')
        desc=file.variables['PRF']._attributes.get('long_name',b'').decode()
        if 'per polarization' not in desc:raise ValueError('PRF convention not confirmed')
        return IPIXRecord(iq.astype(np.complex64),ranges,scalar('RF_frequency')*1e9,prf,
            scalar('Pulse_length')*1e-9,polarization,
            {'source_file':Path(path).name,'adc_axes':list(variable.dimensions),'adc_shape':list(variable.shape),
             'adc_encoding':'unsigned 8-bit','PRF_description':desc,'altitude_m':scalar('radar_elev'),
             'receiver_controls_volts':{k:scalar(k) for k in ('like_stc_RF','like_stc_IF','cross_stc_RF','cross_stc_IF')},
             'beamwidth_deg':scalar('antenna_beamwidth'),'azimuth_deg':float(np.median(file.variables['azimuth_angle'].data)),
             'elevation_deg':float(np.median(file.variables['elevation_angle'].data)),
             'range_axis':'near edge of 30 m resolution cell; approximately 15 m sample spacing',
             'has_full_fast_time':False})


@dataclass(frozen=True)
class IQCorrection:
    mean_i: float
    mean_q: float
    std_i: float
    std_q: float
    sin_phase_imbalance: float

    @classmethod
    def fit(cls,calibration: np.ndarray):
        i=calibration.real.astype(float).ravel();q=calibration.imag.astype(float).ravel()
        mi,mq=float(i.mean()),float(q.mean());si,sq=float(i.std()),float(q.std())
        if min(si,sq)<=0:raise ValueError('degenerate I/Q calibration')
        rho=float(np.mean((i-mi)/si*(q-mq)/sq))
        if abs(rho)>=.999:raise ValueError('near singular I/Q covariance')
        return cls(mi,mq,si,sq,rho)

    def apply(self,raw: np.ndarray) -> np.ndarray:
        i=(raw.real-self.mean_i)/self.std_i;q=(raw.imag-self.mean_q)/self.std_q
        i=(i-q*self.sin_phase_imbalance)/np.sqrt(1-self.sin_phase_imbalance**2)
        return (i+1j*q).astype(np.complex64)
