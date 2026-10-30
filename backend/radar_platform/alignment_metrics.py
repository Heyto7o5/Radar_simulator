"""Calibration-only alignment primitives with explicit target/background masks."""
from dataclasses import dataclass,asdict
import numpy as np
from scipy.optimize import minimize_scalar
from .models import C0


@dataclass(frozen=True)
class TargetPowerCalibration:
    power_gain: float
    background_power: float
    calibration_target_cells: int
    calibration_background_cells: int
    notes: str='relative power fitted on summed calibration ROI after mean background subtraction; not absolute RCS'

    def to_dict(self):return asdict(self)


def fit_target_power(measured_powers,simulated_powers,target_masks,background_masks):
    if not (len(measured_powers)==len(simulated_powers)==len(target_masks)==len(background_masks)) or not len(measured_powers):
        raise ValueError('matched nonempty calibration sets required')
    gains=[];floors=[];nt=nb=0
    for measured,simulated,tm,bm in zip(measured_powers,simulated_powers,target_masks,background_masks):
        m,s=np.asarray(measured,dtype=float),np.asarray(simulated,dtype=float)
        t,b=np.asarray(tm,dtype=bool),np.asarray(bm,dtype=bool)
        if m.shape!=s.shape or t.shape!=m.shape or b.shape!=m.shape or np.any(t&b):raise ValueError('disjoint target/background masks on matched grids required')
        if not np.isfinite(m).all() or not np.isfinite(s).all() or np.any(m<0) or np.any(s<0):raise ValueError('finite nonnegative calibration power required')
        if not t.any() or not b.any():continue
        # Subtract the background energy BEFORE truncating. Clipping each pixel
        # independently would turn zero-mean noise fluctuations into fake target
        # energy in a large ROI. Mean is appropriate for the constant-background
        # approximation; this is not a spatially calibrated environment model.
        floor=float(np.mean(m[b]));signal=float(m[t].sum()-floor*t.sum());pred=float(s[t].sum())
        if signal<=0 or pred<=0:continue
        gains.append(signal/pred);floors.append(floor);nt+=int(t.sum());nb+=int(b.sum())
    if not gains:raise ValueError('no identifiable positive target calibration')
    return TargetPowerCalibration(float(np.median(gains)),float(np.median(floors)),nt,nb)


def fit_global_delay_gain(simulated,measured,frequencies_hz,calibration_mask,*,max_equivalent_range_bias_m=.015):
    """One global delay plus one scalar gain, estimated only from supplied cal data.

    Calibration mask operates on leading channel axes. Zero model channels stay
    zero; no per-channel correction can fabricate missing paths. Search bound is
    declared before examining evaluation records.
    """
    s=np.asarray(simulated)[calibration_mask].astype(np.complex128)
    m=np.asarray(measured)[calibration_mask].astype(np.complex128)
    f=np.asarray(frequencies_hz,dtype=float)
    if s.shape!=m.shape or s.shape[-1]!=len(f) or not np.isfinite(s).all() or not np.isfinite(m).all():raise ValueError('invalid calibrated spectra')
    if max_equivalent_range_bias_m<=0:raise ValueError('positive prespecified delay bound required')
    spectral_cross=np.sum(s.conj()*m,axis=0);power=float(np.sum(abs(s)**2))
    if power<=0:raise ValueError('model is zero on every calibration channel')
    def gain(delta):return np.sum(spectral_cross*np.exp(4j*np.pi*f*delta/C0))/power
    # Coarse grid prevents optimizing an arbitrary local sidelobe of the delay
    # objective. Only a single delay is fitted, not a sampled frequency response.
    grid=np.linspace(-max_equivalent_range_bias_m,max_equivalent_range_bias_m,121)
    values=np.array([abs(gain(x))**2 for x in grid]);index=int(values.argmax())
    low,high=grid[max(0,index-1)],grid[min(len(grid)-1,index+1)]
    optimum=minimize_scalar(lambda x:-abs(gain(x))**2,bounds=(low,high),method='bounded',options={'xatol':1e-10})
    candidates=[(float(optimum.x),gain(optimum.x)),(float(grid[index]),gain(grid[index]))]
    delta,g=max(candidates,key=lambda x:abs(x[1]))
    return {'equivalent_range_bias_m':delta,'delay_s':2*delta/C0,'gain_real':float(g.real),'gain_imag':float(g.imag),
            'bound_m':max_equivalent_range_bias_m,'at_bound':bool(abs(delta)>=max_equivalent_range_bias_m*.999),
            'calibration_channels':int(np.sum(calibration_mask)),'fit_uses_evaluation_data':False}


def apply_delay_gain(signal,frequencies_hz,calibration):
    g=complex(calibration['gain_real'],calibration['gain_imag'])
    return g*np.asarray(signal)*np.exp(-2j*np.pi*np.asarray(frequencies_hz)*calibration['delay_s'])


def fit_relative_channel_response(observed_vectors,expected_steering,*,minimum_consistency=.8):
    """Empirical relative array response from calibration target vectors only.

    Extended-target/annotation errors can be absorbed: this is NOT independently
    measured hardware calibration. Consistency gates are frozen before evaluation.
    """
    x,s=np.asarray(observed_vectors,dtype=complex),np.asarray(expected_steering,dtype=complex)
    if x.shape!=s.shape or x.ndim!=2 or x.shape[0]<3 or not np.isfinite(x).all() or not np.isfinite(s).all():raise ValueError('matched calibration vectors required')
    common=np.sum(s.conj()*x,axis=1)/np.sum(abs(s)**2,axis=1)
    usable=abs(common)>1e-12
    if usable.sum()<3:raise ValueError('insufficient identifiable array calibration')
    ratio=x[usable]/(common[usable,None]*s[usable])
    # Normalize every observation to channel 0, removing target common phase/gain.
    valid=abs(ratio[:,0])>1e-12;ratio=ratio[valid]/ratio[valid,0,None]
    if len(ratio)<3:raise ValueError('reference channel is not identifiable')
    unit=ratio/np.maximum(abs(ratio),1e-30)
    mean_phasor=unit.mean(axis=0);consistency=abs(mean_phasor)
    amplitude=np.median(abs(ratio),axis=0)
    response=amplitude*np.exp(1j*np.angle(mean_phasor))
    stable=bool(np.all(consistency>=minimum_consistency) and np.all((amplitude>=.25)&(amplitude<=4)))
    return {'relative_response_real':response.real.tolist(),'relative_response_imag':response.imag.tolist(),
            'phase_consistency':consistency.tolist(),'relative_amplitude':amplitude.tolist(),
            'calibration_vectors':len(ratio),'minimum_consistency':minimum_consistency,'stable_enough_for_empirical_trial':stable,
            'scope':'calibration-label-conditioned estimate, not independently calibrated hardware'}


def fit_phase_curvature(observed_vectors,*,minimum_consistency=.8):
    """ULA phase curvature, invariant to target common phase and angle slope.

    A constant and a linear channel phase are intentionally NOT calibrated:
    these are confounded with source phase and azimuth zero. This estimates only
    identifiable non-affine phase, under a dominant-plane-wave approximation.
    """
    x=np.asarray(observed_vectors,dtype=complex)
    if x.ndim!=2 or x.shape[0]<3 or x.shape[1]<3 or not np.isfinite(x).all():raise ValueError('finite multi-channel vectors required')
    unit=x/np.maximum(abs(x),1e-30)
    samples=unit[:,2:]*unit[:,:-2]*unit[:,1:-1].conj()**2
    mean=samples.mean(axis=0);consistency=abs(mean);curvature=np.angle(mean)
    n=x.shape[1];difference=np.zeros((n-2,n))
    for i in range(n-2):difference[i,i:i+3]=[1,-2,1]
    index=np.arange(n)-(n-1)/2
    operator=np.vstack([difference,np.ones(n),index])
    phase=np.linalg.solve(operator,np.r_[curvature,0.,0.])
    stable=bool(np.all(consistency>=minimum_consistency) and np.max(abs(curvature))<np.pi/2)
    return {'phase_rad':phase.tolist(),'phase_curvature_rad':curvature.tolist(),'curvature_consistency':consistency.tolist(),
            'stable_enough_for_trial':stable,'calibration_vectors':len(x),'minimum_consistency':minimum_consistency,
            'unidentified_modes':['common phase','linear phase / absolute azimuth zero'],
            'scope':'non-affine ULA phase estimate; no amplitude fit, no per-test-frame calibration'}


def apply_channel_phase_response(adc,phase_rad):
    """Forward, unit-magnitude receiver phase in Tx-major/Rx-minor order.

    Raw measured ADC is never overwritten. Use negative phase only when making
    an explicitly labeled inverse-calibrated measurement product.
    """
    x=np.asarray(adc);phase=np.asarray(phase_rad,dtype=float)
    if x.ndim!=4 or not np.iscomplexobj(x) or phase.shape!=(x.shape[2]*x.shape[3],) or not np.isfinite(phase).all():
        raise ValueError('complex [sample,loop,rx,tx] ADC and one finite phase per virtual channel required')
    response=np.exp(1j*phase.reshape(x.shape[3],x.shape[2]).T)
    return (x*response[None,None]).astype(x.dtype)


def estimate_adc_noise_variance(rd_channels,off_target_mask,range_weights,doppler_weights):
    """Infer a diffuse noise floor from single-channel power, not averaged power.

    Uses the exponential-power median identity and known FFT window noise gain.
    It is an empirical circular-white-noise approximation, not a measured noise
    figure or a reconstruction of static clutter/multipath.
    """
    rd=np.asarray(rd_channels);mask=np.asarray(off_target_mask,dtype=bool)
    if rd.ndim!=4 or mask.shape!=rd.shape[:2] or mask.sum()<100:raise ValueError('enough off-target RD cells required')
    rw,dw=np.asarray(range_weights),np.asarray(doppler_weights)
    window_gain=(np.sum(rw**2)/np.sum(rw)**2)*(np.sum(dw**2)/np.sum(dw)**2)
    if not np.isfinite(window_gain) or window_gain<=0:raise ValueError('valid windows required')
    power=abs(rd.astype(np.complex128))**2
    rx_tx=np.median(power[mask],axis=0)/(np.log(2)*window_gain)
    return rx_tx.T.reshape(-1)


def add_calibrated_noise(adc,variance_tx_major,seed):
    x=np.asarray(adc);variance=np.asarray(variance_tx_major,dtype=float)
    if x.ndim!=4 or variance.shape!=(x.shape[2]*x.shape[3],) or not np.isfinite(variance).all() or np.any(variance<0):raise ValueError('nonnegative per-channel noise variance required')
    rng=np.random.default_rng(seed)
    scale=np.sqrt(variance.reshape(x.shape[3],x.shape[2]).T/2)
    noise=(rng.standard_normal(x.shape)+1j*rng.standard_normal(x.shape))*scale[None,None]
    return (x+noise).astype(x.dtype)
