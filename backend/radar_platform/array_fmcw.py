"""Compatibility adapter for common TDM FMCW synthesis and legacy processing.

NED: North is boresight, East is positive azimuth. The transmitted waveform is
an up-chirp. Mixing tx * conjugate(rx) gives positive range beat and NEGATIVE
slow-time frequency for a closing target. Output velocity is explicitly flipped
to closing-positive. This convention is independent of the legacy IF demo.
Synthesis delegates to SimulationSession, with first-order within-chirp paths.
Legacy processing below retains the original frequency-axis conventions.
The archived legacy_array_reference module is not a production dependency.
No absolute RCS, polarization, coupling, or target rotation is modelled.
"""
from dataclasses import dataclass
import numpy as np
from .fmcw import FMCWConfig
from .models import C0, TargetConfig


@dataclass(frozen=True)
class TDMArray:
    tx_ned_m: np.ndarray
    rx_ned_m: np.ndarray
    tx_offsets_s: np.ndarray
    loop_period_s: float

    def __post_init__(self):
        for name in ('tx_ned_m','rx_ned_m'):
            x=np.asarray(getattr(self,name),dtype=float)
            if x.ndim!=2 or x.shape[1]!=3 or not len(x) or not np.isfinite(x).all():
                raise ValueError('finite [element,3] array positions required')
            object.__setattr__(self,name,x)
        t=np.asarray(self.tx_offsets_s,dtype=float)
        if t.shape!=(len(self.tx_ned_m),) or not np.isfinite(t).all() or np.any(t<0) or np.any(np.diff(t)<=0):
            raise ValueError('one strictly increasing nonnegative offset per Tx required')
        if not np.isfinite(self.loop_period_s) or self.loop_period_s<=t[-1]:
            raise ValueError('Tx offsets must fit a positive loop period')
        object.__setattr__(self,'tx_offsets_s',t)

    @property
    def virtual_positions_ned_m(self):
        # Channel order is tx0_rx0, tx0_rx1, ..., tx1_rx0, ...
        return (self.tx_ned_m[:,None,:]+self.rx_ned_m[None,:,:]).reshape(-1,3)

    @classmethod
    def ideal_ula(cls,config: FMCWConfig,num_tx=2,num_rx=4):
        if num_tx not in (1,2) or num_rx<1 or num_rx>16:raise ValueError('reference ULA size unsupported')
        rx=np.zeros((num_rx,3));rx[:,1]=np.arange(num_rx)*config.wavelength_m/2
        tx=np.zeros((num_tx,3));tx[:,1]=np.arange(num_tx)*num_rx*config.wavelength_m/2
        return cls(tx,rx,np.arange(num_tx)*config.chirp_period_s,num_tx*config.chirp_period_s)


@dataclass
class ArrayRD:
    rd_channels: np.ndarray  # [range, closing-Doppler, rx, tx], uncompensated
    rad: np.ndarray         # [range, closing-Doppler, azimuth]
    range_m: np.ndarray
    velocity_mps: np.ndarray
    closing_doppler_hz: np.ndarray
    azimuth_deg: np.ndarray
    metadata: dict


def simulate_motion_adc(config: FMCWConfig,targets: list[TargetConfig],array: TDMArray,
                        *,platform_position_ned_m=(0.,0.,0.),platform_velocity_ned_mps=(0.,0.,0.),
                        frame_start_s=0.,boresight_ned=(1.,0.,0.),beam_half_angle_deg=89.):
    """Compatibility entry; forward simulation is owned by the common engine.

    The common engine linearizes path inside each chirp and gates out-of-band
    beats. Historical exact instantaneous-path results remain archived, not
    silently treated as numerically identical.
    """
    from .unified.array_adapter import motion_array_configuration
    from .unified.session import SimulationSession
    if config.samples*config.chirps*len(array.tx_ned_m)*len(array.rx_ned_m)>5_000_000:
        raise ValueError('ADC cube exceeds CPU reference limit')
    request = motion_array_configuration(config, targets, array,
        platform_position_ned_m=platform_position_ned_m,
        platform_velocity_ned_mps=platform_velocity_ned_mps,
        frame_start_s=frame_start_s, boresight_ned=boresight_ned,
        beam_half_angle_deg=beam_half_angle_deg)
    return SimulationSession(request).frame(0).adc


def process_array_adc(adc,config: FMCWConfig,array: TDMArray,*,azimuth_deg=None,
                       range_fft_size=None,doppler_fft_size=None,tdm_compensation=True,
                       closing_doppler_sign=-1,spatial_phase_sign=-1,remove_dc=False,
                       range_window='hann',doppler_window='hann',spatial_window='rectangular'):
    x=np.asarray(adc)
    if config.samples<3 or config.chirps<3:raise ValueError('Hann processing requires at least three samples and loops')
    shape=(config.samples,config.chirps,len(array.rx_ned_m),len(array.tx_ned_m))
    if x.shape!=shape or not np.iscomplexobj(x) or not np.isfinite(x).all():raise ValueError('complex [sample,loop,rx,tx] ADC required')
    if closing_doppler_sign not in (-1,1) or spatial_phase_sign not in (-1,1):raise ValueError('explicit +/-1 phase conventions required')
    nr=range_fft_size or config.samples;nd=doppler_fft_size or config.chirps
    if nr<config.samples or nd<config.chirps:raise ValueError('FFTs must not truncate ADC samples')
    angles=np.linspace(-80,80,161) if azimuth_deg is None else np.asarray(azimuth_deg,dtype=float)
    if angles.ndim!=1 or not len(angles) or not np.isfinite(angles).all() or np.any(abs(angles)>90) or np.any(np.diff(angles)<=0):
        raise ValueError('increasing azimuth grid in [-90,90] required')
    if nr*nd*len(angles)>8_000_000:raise ValueError('RAD exceeds CPU reference limit')
    if remove_dc:x=x-x.mean(axis=(0,1),keepdims=True)
    def window(name,count):
        factories={'hann':np.hanning,'hamming':np.hamming,'rectangular':np.ones}
        if name not in factories:raise ValueError('unsupported processing window')
        return factories[name](count)
    rw,dw=window(range_window,config.samples),window(doppler_window,config.chirps)
    rfft=np.fft.fft(x*rw[:,None,None,None],nr,axis=0)/rw.sum()
    rd=np.fft.fftshift(np.fft.fft(rfft*dw[None,:,None,None],nd,axis=1),axes=1)/dw.sum()
    fslow=np.fft.fftshift(np.fft.fftfreq(nd,array.loop_period_s))
    closing_frequency=closing_doppler_sign*fslow
    order=np.argsort(closing_frequency)
    fslow,closing_frequency=fslow[order],closing_frequency[order]
    rd=rd[:,order]
    compensated=rd.copy()
    if tdm_compensation:
        compensated*=np.exp(-2j*np.pi*fslow[None,:,None,None]*array.tx_offsets_s[None,None,None,:])
    virtual=np.transpose(compensated,(0,1,3,2)).reshape(nr,nd,-1)
    sw=window(spatial_window,virtual.shape[-1])
    # Conventional coherent beamforming (Bartlett). Supports declared physical
    # element positions without pretending an arbitrary array is uniform.
    directions=np.c_[np.cos(np.deg2rad(angles)),np.sin(np.deg2rad(angles)),np.zeros(len(angles))]
    steering=np.exp(spatial_phase_sign*2j*np.pi*directions@array.virtual_positions_ned_m.T/config.wavelength_m)
    rad=np.einsum('rdc,ac->rda',virtual*sw,steering.conj(),optimize=True)/sw.sum()
    ranges=np.arange(nr)*config.sample_rate_hz/nr*C0/(2*config.slope_hz_s)
    metadata={'adc_axes':['sample','loop','rx','tx'],'rd_axes':['range','closing_doppler','rx','tx'],
              'rad_axes':['range','closing_doppler','azimuth'],'virtual_order':'tx-major, rx-minor',
              'tdm_compensation':tdm_compensation,'closing_doppler_sign':closing_doppler_sign,
              'spatial_phase_sign':spatial_phase_sign,'angle_processor':'far-field coherent Bartlett',
              'range_coupling_corrected':False,'absolute_power_calibrated':False,
              'windows':{'range':range_window,'doppler':doppler_window,'spatial':spatial_window},
              'adc_duration_s':config.adc_duration_s,'ramp_s':config.ramp_s,'idle_s':config.idle_s,
              'loop_period_s':array.loop_period_s,'tx_offsets_s':array.tx_offsets_s.tolist(),
              'range_resolution_m':config.native_range_resolution_m,
              'velocity_cell_mps':config.wavelength_m/(2*config.chirps*array.loop_period_s),
              'limitations':['no Doppler ambiguity resolution','far-field azimuth only, no elevation',
                             'no channel calibration or polarized scattering','sub-chirp propagation motion neglected']}
    return ArrayRD(rd.astype(np.complex64),rad.astype(np.complex64),ranges,
                   closing_frequency*config.wavelength_m/2,closing_frequency,angles,metadata)
