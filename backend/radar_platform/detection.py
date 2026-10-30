"""Truth-independent peak candidates. Not a calibrated CFAR detector."""
import numpy as np
from .models import BlindDetection,C0


def detect_rd_peaks(power, ranges_m, doppler_hz, wavelength_m, chirp_slope_hz_s,
                    *, range_guard=3, doppler_guard=2, max_peaks=20, threshold_db=-25.):
    p=np.asarray(power,dtype=float)
    ranges,f=np.asarray(ranges_m),np.asarray(doppler_hz)
    if p.shape!=(len(f),len(ranges)) or not np.isfinite(p).all() or np.any(p<0):
        raise ValueError('finite nonnegative RD power [Doppler,range] required')
    maximum=float(p.max())
    if maximum<=0:return []
    remaining=p.copy();threshold=maximum*10**(threshold_db/10);out=[]
    for _ in range(max_peaks):
        d,r=np.unravel_index(np.argmax(remaining),p.shape)
        peak=remaining[d,r]
        if peak<threshold or peak<=0:break
        bias=C0*f[d]/(2*chirp_slope_hz_s)
        out.append(BlindDetection(doppler_bin=int(d),range_bin=int(r),raw_range_m=float(ranges[r]),
            apparent_velocity_mps=float(f[d]*wavelength_m/2),relative_power_db=float(10*np.log10(peak/maximum)),
            range_using_apparent_doppler_m=float(ranges[r]+bias)))
        # Doppler is circular at the PRF boundary. Range is not circular.
        ds=(np.arange(d-doppler_guard,d+doppler_guard+1)%len(f)).astype(int)
        remaining[ds,max(0,r-range_guard):min(len(ranges),r+range_guard+1)]=0
    return out
