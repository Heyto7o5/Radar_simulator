"""Relative co-boresighted monostatic beam approximation, not measured gain."""
import numpy as np


def two_way_voltage_gain(cos_angle,half_angle_deg,model='hard_cone',taper_deg=5.):
    cosine=np.asarray(cos_angle,dtype=float)
    edge=np.cos(np.deg2rad(half_angle_deg))
    if model=='hard_cone':return (cosine>=edge).astype(float)
    if model!='cosine_edge' or not 0<taper_deg<=half_angle_deg:
        raise ValueError('cosine edge requires 0 < taper width <= half angle')
    angle=np.rad2deg(np.arccos(np.clip(cosine,-1,1)))
    fraction=np.clip((angle-(half_angle_deg-taper_deg))/taper_deg,0,1)
    # One-way relative power P=(1+cos(pi*f))/2. Identical Tx/Rx
    # voltage patterns sqrt(P) give two-way complex-amplitude factor P.
    # Thus received POWER is multiplied by P**2, not P.
    return .5*(1+np.cos(np.pi*fraction))
