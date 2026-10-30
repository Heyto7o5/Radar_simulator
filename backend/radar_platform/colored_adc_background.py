"""Fast-time colored circular Gaussian ADC background, no material/RCS claim.

spectral_power[k] is the discrete ADC power spectrum: constant q gives
E|adc|^2=q. Slow-time and receiver channels remain independent. The model can
represent an empirical range-dependent floor but cannot reproduce deterministic
targets, transient phase jumps or general structured clutter dynamics.
"""
import numpy as np


def colored_adc_noise(shape,spectral_power,seed):
    q=np.asarray(spectral_power,dtype=float)
    if len(shape)!=4 or any(int(x)!=x or x<=0 for x in shape):raise ValueError('positive [sample,loop,rx,tx] shape required')
    if q.shape!=(shape[0],) or not np.isfinite(q).all() or np.any(q<0):raise ValueError('one finite nonnegative spectral power per fast FFT bin required')
    rng=np.random.default_rng(seed)
    white=(rng.standard_normal(shape)+1j*rng.standard_normal(shape))/np.sqrt(2)
    colored=np.fft.ifft(np.fft.fft(white,axis=0)*np.sqrt(q)[:,None,None,None],axis=0)
    return colored.astype(np.complex64)


def expected_windowed_range_power(spectral_power,range_window,slow_window,channels=1):
    """Expected angle-mean squared magnitude after unnormalized FFTs.

    Fast-time windowing convolves the PSD; it is NOT correct to equate a desired
    range profile directly with pre-window ADC PSD. This returns the full native
    range FFT axis before cropping, calibration offsets or display resampling.
    """
    q=np.asarray(spectral_power,dtype=float);wr=np.asarray(range_window,dtype=float);wd=np.asarray(slow_window,dtype=float)
    if q.ndim!=1 or len(q)==0 or wr.shape!=q.shape or wd.ndim!=1 or len(wd)==0 or not all(np.isfinite(x).all() for x in [q,wr,wd]) or np.any(q<0) or channels<1 or int(channels)!=channels:
        raise ValueError('invalid PSD, windows or channel count')
    kernel=abs(np.fft.fft(wr))**2
    convolved=np.fft.ifft(np.fft.fft(q)*np.fft.fft(kernel)).real
    return np.maximum(convolved,0)*channels*np.sum(wd**2)/len(q)
