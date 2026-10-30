"""Explicit DART/Rover magnitude processing and physical-axis mapping.

Sources: Rover processing/process/radar.py and DART dart/pose.py project_angle.
The published Doppler coordinate is recession-positive: t dot v = -d.
No intensity fitting or measured peak-dependent alignment occurs here.
"""
import numpy as np


def published_to_closing_axis(velocity_mps, values, axis=1):
    velocity=np.asarray(velocity_mps,dtype=float)
    values=np.asarray(values)
    if velocity.ndim!=1 or not np.isfinite(velocity).all() or np.any(np.diff(velocity)<=0):
        raise ValueError('finite increasing published velocity axis required')
    if values.shape[axis]!=len(velocity):raise ValueError('velocity axis length mismatch')
    closing=-velocity
    order=np.argsort(closing)
    return closing[order],np.take(values,order,axis=axis)


def rover_magnitude(adc):
    """[sample,loop,rx,tx] -> [range,raw Doppler,angle], unscaled magnitude.

    Retains full range FFT. No range-zero correction for ideal geometry and no
    hardware-artifact removal inferred from the current test frame. TX-major
    RX-minor is the official storage order, not a physical-array calibration.
    """
    x=np.asarray(adc)
    if x.shape!=(512,256,4,2) or not np.iscomplexobj(x) or not np.isfinite(x).all():
        raise ValueError('finite complex [512,256,4,2] ADC required')
    window=np.hanning(512)[:,None]*np.hanning(256)[None,:]
    virtual=x.transpose(0,1,3,2).reshape(512,256,8)
    rd=np.fft.fft2(virtual*window[:,:,None],axes=(0,1))
    rad=np.fft.fft(rd,axis=2)
    return np.abs(np.fft.fftshift(rad,axes=(1,2)))


def fit_zero_doppler_profile(calibration_magnitudes):
    """Freeze profile from explicitly supplied calibration frames ONLY."""
    x=np.asarray(calibration_magnitudes,dtype=float)
    if x.ndim!=4 or len(x)<2 or x.shape[2]<3 or not np.isfinite(x).all() or np.any(x<0):
        raise ValueError('at least two nonnegative calibration magnitude frames required')
    center=x.shape[2]//2
    return np.median(x[:,:,center-1:center+2,:],axis=0)


def apply_zero_doppler_profile(magnitude,profile):
    x=np.asarray(magnitude,dtype=float).copy();p=np.asarray(profile,dtype=float)
    if x.ndim!=3 or p.shape!=(x.shape[0],3,x.shape[2]) or not np.isfinite(x).all() or not np.isfinite(p).all() or np.any(x<0) or np.any(p<0):
        raise ValueError('magnitude/profile mismatch or invalid values')
    center=x.shape[1]//2
    x[:,center-1:center+2,:]=np.maximum(x[:,center-1:center+2,:]-p,0)
    return x
