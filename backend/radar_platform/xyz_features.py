"""RDA-only fixed-acquisition preprocessing; contains no scene/pose lookup."""
import numpy as np
from scipy.ndimage import zoom

FEATURE_SHAPE = (2, 64, 16, 24)
XYZ_CENTER = np.array([0., 0., 32.], dtype=np.float32)
XYZ_SCALE = np.array([18., 12., 10.], dtype=np.float32)


def encode_rda(magnitude):
    a = np.asarray(magnitude, dtype=np.float32)
    if a.ndim != 4 or len(a) != 2 or not np.isfinite(a).all() or np.any(a < 0):
        raise ValueError('two finite nonnegative RDA magnitude frames required')
    power = a*a
    mean = power.mean(axis=(1,2,3), keepdims=True)
    if np.any(mean <= 0):
        raise ValueError('empty RDA frame')
    x = np.log1p(power/mean)
    x = zoom(x, np.array(FEATURE_SHAPE)/np.array(x.shape), order=1, prefilter=False)
    if x.shape != FEATURE_SHAPE:
        raise ValueError('unexpected resampled shape')
    return x.astype(np.float32)
