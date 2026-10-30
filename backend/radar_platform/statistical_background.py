"""Explicit range-gated statistical backgrounds; never an ADC/material digital twin.

Three controlled families share the same frozen power inputs: independent
Gaussian, spectrum/range-correlated Gaussian, and Gamma-textured correlated
Gaussian plus an INDEPENDENT Gaussian floor. Parameters come from calibration
only. No measured waveform/PSD template or per-realization renormalization.
"""
from dataclasses import dataclass, replace
from typing import Literal
import numpy as np

from .distributed_clutter import rd_blocks


@dataclass(frozen=True)
class SpectrumComponent:
    weight: float
    center_hz: float
    std_hz: float


@dataclass(frozen=True)
class BackgroundRecipe:
    prf_hz: float
    range_m: tuple[float, ...]
    mean_power: tuple[float, ...]
    components: tuple[SpectrumComponent, ...]
    white_fraction: float
    texture_shape: float
    texture_tau_s: float
    covariance_real: tuple[tuple[float, ...], ...]
    covariance_imag: tuple[tuple[float, ...], ...]
    family: Literal['white_gaussian', 'correlated_gaussian', 'compound_gamma'] = 'compound_gamma'
    domain: Literal['range_gated_iq'] = 'range_gated_iq'

    def __post_init__(self):
        values=[self.prf_hz,self.white_fraction,self.texture_shape,self.texture_tau_s]
        if not np.isfinite(values).all() or min(self.prf_hz,self.texture_shape,self.texture_tau_s)<=0:
            raise ValueError('finite positive sampling and texture parameters required')
        if not 0<=self.white_fraction<=1 or self.domain!='range_gated_iq':raise ValueError('invalid fraction or signal domain')
        if self.family not in ('white_gaussian','correlated_gaussian','compound_gamma'):raise ValueError('unknown background family')
        r,p=np.asarray(self.range_m),np.asarray(self.mean_power)
        if r.ndim!=1 or not 1<=len(r)<=64 or p.shape!=r.shape or not np.isfinite(r).all() or not np.isfinite(p).all() or np.any(p<=0) or np.any(np.diff(r)<=0):
            raise ValueError('positive powers and strictly increasing ranges required')
        if not 1<=len(self.components)<=3:raise ValueError('one to three parametric spectral components required')
        for x in self.components:
            if not np.isfinite([x.weight,x.center_hz,x.std_hz]).all() or min(x.weight,x.std_hz)<=0:
                raise ValueError('invalid spectrum component')
            if abs(x.center_hz)>self.prf_hz/2 or x.std_hz>self.prf_hz/2:raise ValueError('spectrum component outside sampled interval')
        cov=self.covariance
        if cov.shape!=(len(r),len(r)) or not np.isfinite(cov).all():raise ValueError('invalid covariance shape or values')
        if not np.allclose(cov,cov.conj().T,atol=1e-8) or not np.allclose(cov.diagonal(),1,atol=1e-8):
            raise ValueError('covariance must be Hermitian with unit diagonal')
        if np.linalg.eigvalsh(cov).min() < -1e-8:raise ValueError('covariance must be positive semidefinite')

    @property
    def covariance(self):
        return np.asarray(self.covariance_real,dtype=float)+1j*np.asarray(self.covariance_imag,dtype=float)


def gamma_texture(samples, channels, shape, tau_s, prf_hz, rng):
    """Stationary Gamma marginal, exact CIR transition on the sample clock.

    E[T]=1, Var[T]=1/shape, Corr[T(n),T(n+k)]=exp(-k/(PRF*tau)).
    No block jumps, smoothing-induced marginal change or startup transient.
    """
    if min(samples,channels)<1 or not np.isfinite([shape,tau_s,prf_hz]).all() or min(shape,tau_s,prf_hz)<=0:
        raise ValueError('invalid Gamma process parameters')
    persistence=np.exp(-1/(prf_hz*tau_s));one_minus=-np.expm1(-1/(prf_hz*tau_s))
    scale=one_minus/(2*shape)
    x=np.empty((samples,channels),float);x[0]=rng.gamma(shape,1/shape,size=channels)
    for i in range(1,samples):
        noncentral=2*shape*persistence/one_minus*x[i-1]
        x[i]=scale*rng.noncentral_chisquare(2*shape,noncentral)
    return x


def simulate_background(recipe: BackgroundRecipe,samples: int,seed: int=42):
    if samples<512 or samples*len(recipe.range_m)>5_000_000:raise ValueError('background exceeds CPU sample bounds')
    if not 0<=seed<2**32:raise ValueError('seed must be uint32')
    nr=len(recipe.range_m)
    # Separate streams keep floor/field draws identical across model ablations.
    streams=np.random.SeedSequence(seed).spawn(3)
    rng,noise_rng,texture_rng=[np.random.default_rng(s) for s in streams]
    field=(rng.standard_normal((samples,nr))+1j*rng.standard_normal((samples,nr)))/np.sqrt(2)
    power=np.asarray(recipe.mean_power)
    if recipe.family=='white_gaussian':return (field*np.sqrt(power)).astype(np.complex64)
    vals,vec=np.linalg.eigh(recipe.covariance)
    factor=vec*np.sqrt(np.maximum(vals,0))[None,:]
    field=field@factor.T
    frequencies=np.fft.fftfreq(samples,1/recipe.prf_hz)
    spectrum=np.zeros(samples)
    for part in recipe.components:
        delta=(frequencies-part.center_hz+recipe.prf_hz/2)%recipe.prf_hz-recipe.prf_hz/2
        shape=np.exp(-.5*(delta/part.std_hz)**2);shape/=shape.mean()
        spectrum+=part.weight*shape
    spectrum/=sum(p.weight for p in recipe.components)
    field=np.fft.ifft(np.fft.fft(field,axis=0)*np.sqrt(spectrum[:,None]),axis=0)
    if recipe.family=='compound_gamma':
        field*=np.sqrt(gamma_texture(samples,nr,recipe.texture_shape,recipe.texture_tau_s,recipe.prf_hz,texture_rng))
    noise=(noise_rng.standard_normal((samples,nr))+1j*noise_rng.standard_normal((samples,nr)))/np.sqrt(2)
    # Receiver-like floor is not multiplied by scene texture.
    result=np.sqrt((1-recipe.white_fraction)*power)*field+np.sqrt(recipe.white_fraction*power)*noise
    return result.astype(np.complex64)


def fit_background(calibration_iq,ranges_m,prf_hz,clutter_mask,components=2):
    """Fit a small frozen recipe, recording assumptions and identifiable limits."""
    x=np.asarray(calibration_iq,dtype=np.complex128);mask=np.asarray(clutter_mask,dtype=bool);r=np.asarray(ranges_m,float)
    if x.ndim!=2 or x.shape[1]!=len(r) or mask.shape!=r.shape or not mask.any() or not np.isfinite(x).all() or not np.iscomplexobj(calibration_iq):
        raise ValueError('finite complex [time,range] and calibration-only clutter mask required')
    if len(x)<4096 or components not in (1,2):raise ValueError('4096 calibration sweeps and one/two mixture components required')
    powers=np.mean(abs(x)**2,axis=0)
    powers=np.interp(r,r[mask],powers[mask])
    if np.any(powers<=0):raise ValueError('degenerate calibration power')
    normalized=x/np.sqrt(powers)[None,:]
    rd,f=rd_blocks(x,prf_hz)
    psd=np.mean(abs(rd[:,:,mask])**2/powers[mask][None,None,:],axis=(0,2))
    floor=float(np.median(psd[abs(f)>.35*prf_hz]))
    white=float(np.clip(floor*len(f)/psd.sum(),0,1))
    weights=np.maximum(psd-floor,0)
    if weights.sum()<=0:weights=np.ones_like(weights)
    weights/=weights.sum()
    anchor=float(f[np.argmax(weights)])
    z=(f-anchor+prf_hz/2)%prf_hz-prf_hz/2
    center=float(z@weights);std=max(float(np.sqrt(((z-center)**2)@weights)),prf_hz/512)
    means=np.array([center] if components==1 else [center-.5*std,center+.5*std])
    sigmas=np.full(components,std);mix=np.full(components,1/components)
    for _ in range(40):
        likelihood=mix[None,:]/sigmas[None,:]*np.exp(-.5*((z[:,None]-means)/sigmas)**2)
        responsibilities=likelihood/np.maximum(likelihood.sum(1,keepdims=True),1e-250)
        mass=weights[:,None]*responsibilities
        mix=np.maximum(mass.sum(0),1e-6);mix/=mix.sum()
        means=(mass*z[:,None]).sum(0)/np.maximum(mass.sum(0),1e-12)
        sigmas=np.sqrt((mass*(z[:,None]-means)**2).sum(0)/np.maximum(mass.sum(0),1e-12))
        sigmas=np.clip(sigmas,prf_hz/512,prf_hz/2)
    parts=tuple(SpectrumComponent(float(w),float((m+anchor+prf_hz/2)%prf_hz-prf_hz/2),float(s)) for w,m,s in zip(mix,means,sigmas))
    intensity=abs(normalized[:,mask])**2
    excess=float(np.mean(intensity**2))-2
    shape=float(np.clip(2*(1-white)**2/max(excess,1e-6),.2,1000))
    # Long-lag intensity correlation probes slow texture, not just Doppler speckle.
    lags=np.array([64,128,256,512,1024,2048,4096,8192]);lags=lags[lags<len(x)//3]
    centered=intensity-intensity.mean(0);variance=float(np.mean(centered**2))
    corr=np.array([np.mean(centered[:-k]*centered[k:])/max(variance,1e-30) for k in lags])
    keep=(corr>.05)&(corr<.95);tau=.5;estimated=False
    if keep.sum()>=3:
        slope,_=np.polyfit(lags[keep]/prf_hz,np.log(corr[keep]),1)
        if slope<0:tau=float(np.clip(-1/slope,1/prf_hz,len(x)/prf_hz/4));estimated=True
    # Covariance only within uncontaminated gates; omitted gates stay independent.
    cov=np.eye(len(r),dtype=complex);cal=normalized[:,mask]
    empirical=cal.T@cal.conj()/len(cal)
    diag=np.sqrt(np.maximum(empirical.diagonal().real,1e-12))
    empirical=empirical/(diag[:,None]*diag[None,:])
    empirical=.9*empirical+.1*np.eye(len(empirical))
    cov[np.ix_(mask,mask)]=empirical
    result=BackgroundRecipe(prf_hz,tuple(r),tuple(powers),parts,white,shape,tau,
        tuple(map(tuple,cov.real)),tuple(map(tuple,cov.imag)))
    meta={'domain':'range_gated_iq','calibration_samples':len(x),'scoring_range_indices_zero_based':np.flatnonzero(mask).tolist(),
          'spectrum_model':'one/two wrapped Gaussian components, 40 fixed EM iterations; no PSD template',
          'floor_assumption':'median high-Doppler power; not an independently measured thermal noise figure',
          'texture_tau_estimated':estimated,'fallback_tau_s':None if estimated else .5,
          'texture_correlation_lags_samples':lags.tolist(),'texture_intensity_correlations':corr.tolist(),
          'covariance':'calibration complex correlation, 10% identity shrinkage; not independent antenna calibration',
          'limitations':['texture independent across range gates','common spectrum shape across gates',
                         'range-gated signal, not full fast-time ADC','stationary parameters per generated batch',
                         'mean powers absorb receiver gain and scene jointly','no geometry-driven sea state prediction']}
    return result,meta
