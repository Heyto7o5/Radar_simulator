"""Read-only relative-delay diagnostics; never correct validation targets in-place."""
import numpy as np
from scipy.signal import correlate,correlation_lags


def profile_shift(reference,observation,range_m,window_m,max_shift_m=1.):
    a,b,r=map(np.asarray,[reference,observation,range_m])
    if a.ndim!=1 or a.shape!=b.shape or a.shape!=r.shape or len(a)<4 or not np.iscomplexobj(a) or not np.iscomplexobj(b):
        raise ValueError('complex profiles on one matching range axis required')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(r).all() or not np.all(np.diff(r)>0) or not np.allclose(np.diff(r),np.diff(r)[0]):
        raise ValueError('finite profiles and uniform increasing axis required')
    if not 0<max_shift_m<(window_m[1]-window_m[0])/2:raise ValueError('invalid diagnostic window/lag limit')
    mask=(r>=window_m[0])&(r<=window_m[1]);x,y=a[mask],b[mask]
    if len(x)<4:raise ValueError('window contains too few samples')
    norm=np.sqrt(np.vdot(x,x).real*np.vdot(y,y).real)
    if norm<=1e-30:raise ValueError('zero-energy diagnostic window')
    corr=correlate(y,x,mode='full',method='fft')
    lags=correlation_lags(len(y),len(x))*np.diff(r)[0]
    eligible=np.flatnonzero(abs(lags)<=max_shift_m)
    k=eligible[np.argmax(abs(corr[eligible]))]
    return {'shift_m':float(lags[k]),'normalized_complex_correlation':float(np.clip(abs(corr[k])/norm,0,1)),
            'zero_lag_correlation':float(abs(corr[np.argmin(abs(lags))])/norm),
            'profile_norm_ratio':float(np.linalg.norm(y)/np.linalg.norm(x)),
            'range_sample_interval_m':float(np.diff(r)[0]),'statistical_confidence_interval':None,
            'scope':'relative profile-shape displacement; not independent range truth or automatic clock calibration'}
