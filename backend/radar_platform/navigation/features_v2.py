"""Radar-only spectral features and native Doppler moments, without pose reads."""
import numpy as np
from scipy.ndimage import zoom

from .observation import encode

FEATURE_SHAPE_V2=(2,20,96,21,21)


def directions(elevation_deg,azimuth_deg):
    az,el=np.meshgrid(np.deg2rad(azimuth_deg),np.deg2rad(elevation_deg))
    # Radar-local down/right/aft -> aircraft forward/right/up.
    return np.stack([np.sin(el),np.cos(el)*np.sin(az),-np.cos(el)*np.cos(az)],axis=-1)


def coordinate_grid(range_m,elevation_deg,azimuth_deg):
    ran=np.linspace(range_m[0],range_m[-1],96)
    elevation=np.linspace(elevation_deg[0],elevation_deg[-1],21)
    azimuth=np.linspace(azimuth_deg[0],azimuth_deg[-1],21)
    return (ran[:,None,None,None]*directions(elevation,azimuth)[None]).astype(np.float32)


def encode_v2(observation):
    observation.validate()
    spectral=encode(observation)
    power=np.square(np.asarray(observation.magnitude,np.float32))
    velocity=np.asarray(observation.velocity_mps,np.float32)
    total=power.sum(axis=2);denominator=np.maximum(total,1e-25)
    first=np.einsum('trdea,d->trea',power,velocity,optimize=True)
    second=np.einsum('trdea,d->trea',power,velocity**2,optimize=True)
    mean=first/denominator
    std=np.sqrt(np.maximum(0,second/denominator-mean**2))
    normalizer=total.mean(axis=(1,2,3),keepdims=True)
    peak=power.max(axis=2)
    vmax=max(float(np.max(abs(velocity))),1e-6)
    moment_maps=np.stack([np.log1p(total/normalizer),
                         np.log1p(peak/(normalizer/len(velocity))),mean/vmax,std/vmax],axis=1)
    desired=np.array([2,4,96,21,21]);moment_maps=zoom(moment_maps,desired/moment_maps.shape,order=1,prefilter=False)
    features=np.concatenate([spectral,moment_maps],axis=1).astype(np.float32)
    # Static terrain/building radial velocity is the platform-velocity projection.
    # Bartlett smearing makes this a baseline, subsequently corrected by learning.
    u=directions(observation.elevation_deg,observation.azimuth_deg)
    support=(-u[...,2])>=.5  # fixed 60-degree physical beam, independent of any pose
    baseline=[]
    for t in range(2):
        weights=total[t].sum(axis=0)*support
        moment=first[t].sum(axis=0)*support
        matrix=np.einsum('ea,eai,eaj->ij',weights,u,u)
        rhs=np.einsum('ea,eai->i',moment,u)
        ridge=max(float(np.trace(matrix))*1e-3,1e-20)
        baseline.append(np.linalg.solve(matrix+ridge*np.eye(3),rhs))
    if features.shape!=FEATURE_SHAPE_V2:raise ValueError('unexpected v2 feature shape')
    return features,np.asarray(baseline,np.float32)
