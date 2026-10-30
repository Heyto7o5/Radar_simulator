"""Common-reference power comparisons. No per-image peak fitting or shifting."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PowerComparison:
    measured_db: np.ndarray
    simulated_db: np.ndarray
    difference_db: np.ndarray
    metrics: dict


def power_db(power, reference_power: float, floor_db: float = -60.) -> np.ndarray:
    p=np.asarray(power,dtype=np.float64)
    if not np.isfinite(p).all() or np.any(p<0):
        raise ValueError('power must be finite and nonnegative')
    if not np.isfinite(reference_power) or reference_power<=0 or not np.isfinite(floor_db) or floor_db>=0:
        raise ValueError('positive fixed reference and negative floor required')
    # No upper clipping: a prediction that is too strong must remain visible.
    return 10*np.log10(np.maximum(p/reference_power,10.**(floor_db/10)))


def compare_power(measured, simulated, *, reference_power: float, floor_db: float=-60.,
                  valid_mask=None, target_mask=None) -> PowerComparison:
    m,s=np.asarray(measured),np.asarray(simulated)
    if m.shape!=s.shape or not m.size:raise ValueError('matched nonempty shapes required; align axes before scoring')
    valid=np.ones(m.shape,dtype=bool) if valid_mask is None else np.asarray(valid_mask,dtype=bool)
    if valid.shape!=m.shape or not valid.any():raise ValueError('invalid scoring mask')
    a,b=power_db(m,reference_power,floor_db),power_db(s,reference_power,floor_db)
    diff=b-a
    def score(mask):return float(np.mean(abs(diff[mask]))) if mask.any() else None
    threshold=reference_power*10.**(floor_db/10)
    measured_floor=m<=threshold;simulated_floor=s<=threshold
    support=valid & ~(measured_floor & simulated_floor)
    metrics={'mae_db':score(valid),'scored_cells':int(valid.sum()),
             'reference_power':float(reference_power),'floor_db':float(floor_db),
             'difference':'simulated_db - measured_db','amplitude_calibration':'relative, common reference',
             'simulation_zero_power_fraction':float(np.mean(s[valid]==0)),
             'measured_floor_fraction':float(np.mean(measured_floor[valid])),
             'simulated_floor_fraction':float(np.mean(simulated_floor[valid])),
             'joint_floor_fraction':float(np.mean((measured_floor&simulated_floor)[valid])),
             'signal_support_mae_db':score(support),'signal_support_cells':int(support.sum()),
             'linear_power_mae_reference_units':float(np.mean(abs(s[valid]-m[valid]))/reference_power),
             'target_mae_db':None,'target_scored_cells':0}
    if target_mask is not None:
        target=np.asarray(target_mask,dtype=bool).copy()
        if target.shape!=m.shape:raise ValueError('ROI must already be aligned')
        target &= valid
        metrics['target_mae_db']=score(target)
        metrics['target_scored_cells']=int(target.sum())
        metrics['target_signal_support_mae_db']=score(target&support)
        metrics['target_joint_floor_fraction']=float(np.mean((measured_floor&simulated_floor)[target])) if target.any() else None
    return PowerComparison(a,b,diff,metrics)


def average_rd_power(complex_rd) -> np.ndarray:
    """[CPI, Doppler, range] -> average LINEAR power, never average log power."""
    x=np.asarray(complex_rd)
    if x.ndim!=3 or not np.iscomplexobj(x) or not len(x):
        raise ValueError('need complex [CPI,Doppler,range]')
    return np.mean(abs(x.astype(np.complex128))**2,axis=0)


def representative_indices(errors) -> list[int]:
    """Three fixed quantile positions in stable case order, plus the worst case."""
    x=np.asarray(errors,dtype=float)
    if x.ndim!=1 or not len(x) or not np.isfinite(x).all():raise ValueError('finite ordered case errors required')
    fixed=[int(round((len(x)-1)*q)) for q in (.25,.5,.75)]
    return list(dict.fromkeys([*fixed,int(np.argmax(x))]))
