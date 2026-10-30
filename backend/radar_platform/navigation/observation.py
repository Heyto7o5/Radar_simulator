"""Truth-free radar observation contract and fixed numeric preprocessing."""
from dataclasses import dataclass
import numpy as np
from scipy.ndimage import zoom

FEATURE_SHAPE = (2, 16, 96, 21, 21)  # time, Doppler channels, range, elevation, azimuth


@dataclass(frozen=True)
class Observation:
    magnitude: np.ndarray  # [time, range, Doppler, elevation, azimuth]
    range_m: np.ndarray
    velocity_mps: np.ndarray
    elevation_deg: np.ndarray
    azimuth_deg: np.ndarray
    frame_start_s: np.ndarray
    frame_end_s: np.ndarray

    def validate(self):
        a = np.asarray(self.magnitude)
        axes = (self.range_m, self.velocity_mps, self.elevation_deg, self.azimuth_deg)
        if a.shape != (2, *(len(x) for x in axes)):
            raise ValueError('two radar volumes [time, range, Doppler, elevation, azimuth] required')
        if not np.isfinite(a).all() or np.any(a < 0):
            raise ValueError('finite nonnegative radar magnitudes required')
        for axis in axes:
            if not np.isfinite(axis).all() or np.any(np.diff(axis)<=0):
                raise ValueError('finite increasing acquisition axes required')
        start, end = map(np.asarray, (self.frame_start_s, self.frame_end_s))
        if start.shape!=(2,) or end.shape!=(2,) or not np.isfinite(np.r_[start,end]).all():
            raise ValueError('two finite frame timestamps required')
        if np.any(end<=start) or end[0]>start[1] or np.any(np.diff(start)<=0):
            raise ValueError('non-overlapping chronological acquired frames required')
        if np.any(np.sum(a*a, axis=(1,2,3,4))<=0):
            raise ValueError('empty radar frame')

    def arrays(self):
        return {key:np.asarray(getattr(self,key)) for key in self.__dataclass_fields__}

    @classmethod
    def load(cls, path):
        # No scene, trajectory, labels, case id, or truth metadata is accepted.
        with np.load(path, allow_pickle=False) as data:
            if set(data.files)!=set(cls.__dataclass_fields__):
                raise ValueError('observation contains missing or unauthorized fields')
            observation = cls(**{k:data[k] for k in data.files})
        observation.validate()
        return observation


def encode(observation):
    observation.validate()
    a = np.asarray(observation.magnitude, np.float32)
    power = np.square(a)
    x = np.log1p(power/np.mean(power, axis=(1,2,3,4), keepdims=True))
    x = x.transpose(0,2,1,3,4)
    x = zoom(x, np.array(FEATURE_SHAPE)/x.shape, order=1, prefilter=False)
    if x.shape != FEATURE_SHAPE:
        raise ValueError('unexpected preprocessed shape')
    return x.astype(np.float32)
