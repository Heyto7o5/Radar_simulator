"""Sampled rigid-platform motion: Hermite translation and shortest-arc SLERP.

Times share the simulation global clock. Rotations map sensor forward/right/down
coordinates into world NED. No out-of-range extrapolation is permitted.
"""
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation, Slerp


class RigidTrajectory(BaseModel):
    model_config=ConfigDict(extra='forbid',allow_inf_nan=False)
    time_s:list[float]=Field(min_length=2,max_length=2048)
    position_ned_m:list[tuple[float,float,float]]
    velocity_ned_mps:list[tuple[float,float,float]]
    quaternion_xyzw:list[tuple[float,float,float,float]]
    provenance:str=Field(min_length=1,max_length=1000)

    @model_validator(mode='after')
    def valid_samples(self):
        n=len(self.time_s)
        if any(len(x)!=n for x in [self.position_ned_m,self.velocity_ned_mps,self.quaternion_xyzw]):
            raise ValueError('one position, velocity and quaternion per trajectory time required')
        if np.any(np.diff(self.time_s)<=0):raise ValueError('trajectory times must increase strictly')
        q=np.asarray(self.quaternion_xyzw)
        if not np.allclose(np.linalg.norm(q,axis=1),1,atol=1e-5,rtol=0):raise ValueError('trajectory quaternions must be normalized')
        return self


class TrajectoryEvaluator:
    def __init__(self,spec:RigidTrajectory):
        self.times=np.asarray(spec.time_s)
        self.translation=CubicHermiteSpline(self.times,spec.position_ned_m,spec.velocity_ned_mps,extrapolate=False)
        rotations=Rotation.from_quat(spec.quaternion_xyzw)
        self.rotation=Slerp(self.times,rotations)
        relative=(rotations[:-1].inv()*rotations[1:]).as_rotvec()/np.diff(self.times)[:,None]
        self.angular_velocity=rotations[:-1].apply(relative)

    def at(self,time_s):
        times=np.asarray(time_s,dtype=float)
        if not np.isfinite(times).all() or np.any(times<self.times[0]) or np.any(times>self.times[-1]):
            raise ValueError('requested time outside sampled trajectory; extrapolation forbidden')
        flat=times.reshape(-1)
        intervals=np.clip(np.searchsorted(self.times,flat,side='right')-1,0,len(self.times)-2)
        shape=times.shape
        return (self.translation(times),self.translation(times,1),
                self.rotation(flat).as_matrix().reshape(*shape,3,3),
                self.angular_velocity[intervals].reshape(*shape,3))
