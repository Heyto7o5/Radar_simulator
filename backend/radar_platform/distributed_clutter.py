"""Range-gated semi-empirical clutter IQ generator for RD statistics.

Independent random signal generation from a small, frozen parameter set.
No measured waveform/PSD template is sampled.  This is a statistical surface
cell model, NOT a hydrodynamic sea or exact floating-target digital twin.
"""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .models import C0


@dataclass(frozen=True)
class ClutterRecipe:
    carrier_hz: float
    prf_hz: float
    pulse_width_s: float
    range_near_edge_m: tuple[float,...]
    mean_power: tuple[float,...]
    doppler_center_hz: float
    doppler_std_hz: float
    white_power_fraction: float
    texture_shape: float
    texture_block_samples: int=1000

    def __post_init__(self):
        if min(self.carrier_hz,self.prf_hz,self.pulse_width_s,self.doppler_std_hz,self.texture_shape)<=0:raise ValueError('positive physical/model scales required')
        r=np.asarray(self.range_near_edge_m);p=np.asarray(self.mean_power)
        if len(r)<2 or len(r)!=len(p) or np.any(np.diff(r)<=0) or not np.isfinite(r).all() or not np.isfinite(p).all() or np.any(p<=0):raise ValueError('invalid range/power vectors')
        if not 0<=self.white_power_fraction<=1:raise ValueError('invalid white fraction')
        if self.texture_block_samples<1:raise ValueError('invalid texture time scale')


def rd_blocks(iq: np.ndarray,prf_hz: float,pulses: int=512) -> tuple[np.ndarray,np.ndarray]:
    x=np.asarray(iq)
    if x.ndim!=2 or len(x)<pulses or prf_hz<=0 or pulses<4:raise ValueError('need [slow_time,range] and valid CPI')
    count=len(x)//pulses
    windows=x[:count*pulses].reshape(count,pulses,x.shape[1])
    w=np.hanning(pulses)
    rd=np.fft.fftshift(np.fft.fft(windows*w[None,:,None],axis=1),axes=1)/np.sqrt(np.sum(w*w))
    return rd.astype(np.complex64),np.fft.fftshift(np.fft.fftfreq(pulses,1/prf_hz))


def fit_clutter_recipe(calibration_iq: np.ndarray,ranges_m: np.ndarray,carrier_hz: float,prf_hz: float,pulse_width_s: float,clutter_mask: np.ndarray) -> ClutterRecipe:
    x=np.asarray(calibration_iq);mask=np.asarray(clutter_mask,dtype=bool)
    if x.shape[1]!=len(mask) or not mask.any():raise ValueError('no clutter-only gates')
    observed_power=np.mean(abs(x)**2,axis=0)
    # Target-contaminated gates are filled from adjacent clutter gates only.
    powers=np.interp(ranges_m,ranges_m[mask],observed_power[mask])
    blocks,f=rd_blocks(x,prf_hz)
    normalized=abs(blocks[:,:,mask])**2/powers[mask][None,None,:]
    psd=np.mean(normalized,axis=(0,2))
    floor=float(np.median(psd[abs(f)>.35*prf_hz]))
    signal=np.maximum(psd-floor,0.)
    center=float(np.sum(f*signal)/np.sum(signal))
    std=float(np.sqrt(np.sum((f-center)**2*signal)/np.sum(signal)))
    white=float(np.clip(floor*len(f)/np.sum(psd),0,1))
    normalized_intensity=abs(x[:,mask])**2/powers[mask]
    second=float(np.mean(normalized_intensity**2))
    shape=float(np.clip(2/max(second-2,1e-5),.2,1000))
    return ClutterRecipe(carrier_hz,prf_hz,pulse_width_s,tuple(map(float,ranges_m)),tuple(map(float,powers)),center,max(std,prf_hz/512),white,shape)


def simulate_clutter_iq(recipe: ClutterRecipe,samples: int,seed: int=42,white_baseline: bool=False) -> np.ndarray:
    if samples<512 or samples*len(recipe.mean_power)>5_000_000:raise ValueError('sample count outside CPU limit')
    rng=np.random.default_rng(seed);nr=len(recipe.mean_power)
    noise=(rng.standard_normal((samples,nr))+1j*rng.standard_normal((samples,nr)))/np.sqrt(2.)
    if white_baseline:return (noise*np.sqrt(recipe.mean_power)).astype(np.complex64)
    # Rectangular pulse-gate overlap supplies a simple spatial covariance.
    ranges=np.asarray(recipe.range_near_edge_m)
    cell_width=C0*recipe.pulse_width_s/2
    cov=np.maximum(1-abs(ranges[:,None]-ranges[None,:])/cell_width,0)+np.eye(nr)*1e-8
    noise=noise@np.linalg.cholesky(cov).T
    f=np.fft.fftfreq(samples,1/recipe.prf_hz)
    delta=(f-recipe.doppler_center_hz+recipe.prf_hz/2)%recipe.prf_hz-recipe.prf_hz/2
    shaped=np.exp(-.5*(delta/recipe.doppler_std_hz)**2)
    shaped/=shaped.mean()
    spectrum=(1-recipe.white_power_fraction)*shaped+recipe.white_power_fraction
    field=np.fft.ifft(np.fft.fft(noise,axis=0)*np.sqrt(spectrum[:,None]),axis=0)
    blocks=(samples+recipe.texture_block_samples-1)//recipe.texture_block_samples
    texture=rng.gamma(recipe.texture_shape,1/recipe.texture_shape,size=(blocks,nr))
    texture=np.repeat(texture,recipe.texture_block_samples,axis=0)[:samples]
    # Fixed smoothing avoids abrupt block jumps; time constant is an explicit assumption.
    texture=gaussian_filter1d(texture,sigma=max(1,recipe.texture_block_samples/20),axis=0,mode='nearest')
    return (field*np.sqrt(texture*np.asarray(recipe.mean_power))).astype(np.complex64)
