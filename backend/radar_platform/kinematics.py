"""Linear world-coordinate motion, shared by the pulsed and FMCW paths."""
import numpy as np


def position_at(initial_position, velocity, time_s):
    p,v=np.asarray(initial_position,dtype=float),np.asarray(velocity,dtype=float)
    t=np.asarray(time_s,dtype=float)
    if p.shape!=(3,) or v.shape!=(3,) or not np.isfinite(p).all() or not np.isfinite(v).all() or not np.isfinite(t).all():
        raise ValueError('finite NED position/velocity vectors and times required')
    return p+t[...,None]*v


def relative_geometry(radar_position, radar_velocity, target_position, target_velocity=(0.,0.,0.)):
    los=np.asarray(target_position,dtype=float)-np.asarray(radar_position,dtype=float)
    distance=np.linalg.norm(los,axis=-1)
    if np.any(distance<=0) or not np.isfinite(distance).all():raise ValueError('target intersects radar or is non-finite')
    direction=los/distance[...,None]
    closing=np.sum((np.asarray(radar_velocity)-np.asarray(target_velocity))*direction,axis=-1)
    return distance,direction,closing
