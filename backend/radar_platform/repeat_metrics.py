"""Descriptive, not independent-trial, metrics for repeated channel/frame checks."""
import numpy as np


def peak_width(axis, power):
    """Global peak and interpolated contiguous half-power width; no extrapolation.

    Half-power is -3.0103 dB. Zero signals and clipped lobes have no width.
    Interpolation improves numerical measurement, not physical resolution.
    """
    x = np.asarray(axis, dtype=float)
    y = np.asarray(power, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or len(x) < 3 or np.any(np.diff(x) <= 0):
        raise ValueError("need increasing 1D axis and matching power")
    if not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("power must be finite and nonnegative")
    if y.max() <= 0:
        return {"peak": None, "width": None}
    peak = int(np.argmax(y))
    level = y[peak] / 2
    left = peak
    while left > 0 and y[left] > level:
        left -= 1
    right = peak
    while right < len(y) - 1 and y[right] > level:
        right += 1
    width = None
    if left < peak < right and y[left] <= level and y[right] <= level:
        xl = x[left] + (level-y[left])/(y[left+1]-y[left])*(x[left+1]-x[left])
        xr = x[right-1] + (level-y[right-1])/(y[right]-y[right-1])*(x[right]-x[right-1])
        width = float(xr-xl)
    return {"peak": float(x[peak]), "width": width}


def distribution(values):
    """Descriptive distribution; never turn correlated channels into a CI."""
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(finite):
        return {"n": 0, "mean": None, "median": None, "p10": None, "p90": None, "min": None, "max": None}
    return {"n": len(finite), "mean": float(finite.mean()), "median": float(np.median(finite)),
            "p10": float(np.quantile(finite, .1)), "p90": float(np.quantile(finite, .9)),
            "min": float(finite.min()), "max": float(finite.max())}


def common_safe_gates(count, training_excluded, evaluation_excluded):
    """Exclude BOTH train-infilled gates and published test target gates."""
    mask = np.ones(count, dtype=bool)
    for gate in set(training_excluded) | set(evaluation_excluded):
        if gate < 1 or gate > count:
            raise ValueError("one-based gate outside range")
        mask[gate-1] = False
    if not mask.any():
        raise ValueError("no shared clutter-only gates")
    return mask
