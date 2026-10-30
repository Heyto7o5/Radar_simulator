"""Calibration-only scaling and shared-reference scores; no per-frame fitting."""
import numpy as np


def fit_power_gain(real,simulated,mask):
    real=np.asarray(real,float);simulated=np.asarray(simulated,float);mask=np.asarray(mask,bool)
    if real.shape!=simulated.shape or mask.shape!=real.shape or not mask.any():raise ValueError('calibration arrays/mask mismatch')
    if not np.isfinite(real[mask]).all() or not np.isfinite(simulated[mask]).all() or np.any(real[mask]<0) or np.any(simulated[mask]<0):raise ValueError('invalid calibration power')
    numerator=real[mask].sum();denominator=simulated[mask].sum()
    if numerator<=0 or denominator<=0:raise ValueError('calibration contains no power')
    return float(numerator/denominator)


def db_power(power,reference):
    if not np.isfinite(reference) or reference<=0:raise ValueError('positive shared power reference required')
    return 10*np.log10(np.maximum(np.asarray(power,dtype=float)/reference,1e-6))


def mae_db(real,simulated,reference,mask):
    real=np.asarray(real);simulated=np.asarray(simulated);mask=np.asarray(mask,bool)
    if real.shape!=simulated.shape or mask.shape!=real.shape or not mask.any():raise ValueError('score arrays/mask mismatch')
    if not np.isfinite(real[mask]).all() or not np.isfinite(simulated[mask]).all() or np.any(real[mask]<0) or np.any(simulated[mask]<0):raise ValueError('invalid evaluation power')
    return float(np.mean(abs(db_power(simulated[mask],reference)-db_power(real[mask],reference))))


def complex_white_noise(shape,variance,seed):
    if not np.isfinite(variance) or variance<0:raise ValueError('nonnegative finite complex variance required')
    rng=np.random.default_rng(seed)
    return (np.sqrt(variance/2)*(rng.standard_normal(shape)+1j*rng.standard_normal(shape))).astype(np.complex64)
