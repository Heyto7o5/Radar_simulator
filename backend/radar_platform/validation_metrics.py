"""Scalar calibration and complex-signal validation; no per-channel tuning."""
from __future__ import annotations
import numpy as np
from scipy.stats import wasserstein_distance


def fit_complex_gain(simulated: np.ndarray, measured: np.ndarray, calibration_mask: np.ndarray) -> complex:
    s=np.asarray(simulated)[calibration_mask].ravel().astype(np.complex128)
    m=np.asarray(measured)[calibration_mask].ravel().astype(np.complex128)
    power=float(np.vdot(s,s).real)
    if power<=1e-24: return 0j
    return complex(np.vdot(s,m)/power)


def complex_metrics(simulated: np.ndarray, measured: np.ndarray, mask: np.ndarray) -> dict:
    s=np.asarray(simulated)[mask].ravel().astype(np.complex128)
    m=np.asarray(measured)[mask].ravel().astype(np.complex128)
    power=float(np.vdot(m,m).real)
    if power<=0: raise ValueError("reference has no signal energy")
    pred_power=float(np.vdot(s,s).real)
    coherence=float(abs(np.vdot(s,m))/np.sqrt(power*pred_power)) if pred_power>0 else 0.
    nmse=float(np.vdot(s-m,s-m).real/power)
    return {"complex_coherence":coherence,"complex_nmse":nmse,
            "complex_nmse_db":float(10*np.log10(max(nmse,1e-30))),
            "relative_power_db":float(10*np.log10(max(pred_power/power,1e-30)))}


def profile_metrics(sim_power: np.ndarray, measured_power: np.ndarray, ranges_m: np.ndarray) -> dict:
    s=np.asarray(sim_power,dtype=float); m=np.asarray(measured_power,dtype=float)
    if s.sum()<=0 or m.sum()<=0: raise ValueError("profile has no positive energy")
    s=s/s.sum();m=m/m.sum()
    return {"profile_power_correlation":float(np.corrcoef(s,m)[0,1]),
        "profile_wasserstein_m":float(wasserstein_distance(ranges_m,ranges_m,u_weights=s,v_weights=m)),
        "mean_profile_peak_error_m":float(abs(ranges_m[s.argmax()]-ranges_m[m.argmax()])),
        "measured_mean_profile_peak_m":float(ranges_m[m.argmax()]),
        "simulated_mean_profile_peak_m":float(ranges_m[s.argmax()])}
