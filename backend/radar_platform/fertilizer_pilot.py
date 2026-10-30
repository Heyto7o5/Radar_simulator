"""Fixed-scene RDA-only calibration baseline and explicit ideal drop model.

Inference receives numeric RDA magnitudes only. Fixed calibration weights encode
the scene/target and acquisition, so this is not unknown-scene localization.
"""
import numpy as np
from scipy.optimize import brentq

from .dem_los_reference import height_at


def rda_features(magnitudes):
    x = np.asarray(magnitudes, dtype=np.float32)
    if x.ndim != 4 or not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError('finite nonnegative [frame,range,doppler,azimuth] magnitudes required')
    if np.any(np.sum(x*x, axis=(1, 2, 3)) == 0):
        raise ValueError('empty radar frame')
    # Fixed transform, no sample truth, scene geometry or test-set statistics.
    x = np.sqrt(x)
    x = x.reshape(len(x), -1)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x.ravel() / np.sqrt(len(x))


class RDACalibration:
    def __init__(self, features, states, neighbors=4):
        self.features = np.asarray(features, dtype=np.float32)
        self.states = np.asarray(states, dtype=float)
        self.neighbors = neighbors
        if self.features.ndim != 2 or self.states.shape != (len(self.features), 4):
            raise ValueError('calibration states must be [north,east,v_north,v_east]')
        if not np.isfinite(self.features).all() or not np.isfinite(self.states).all():
            raise ValueError('finite calibration required')

    def predict(self, rda):
        feature = rda_features(rda)
        if feature.shape != self.features.shape[1:]:
            raise ValueError('RDA shape differs from calibration')
        distance = np.maximum(0., np.sum(self.features**2, axis=1) + np.dot(feature, feature)
                              - 2*self.features@feature)
        indices = np.argsort(distance)[:self.neighbors]
        weights = 1/np.maximum(distance[indices], 1e-6)**2
        state = np.average(self.states[indices], axis=0, weights=weights)
        return state, dict(nearest_distance_squared=float(distance[indices[0]]),
                           nearest_state=self.states[indices[0]].tolist())


def ideal_position_adjustment(state_at_start, *, observation_s, altitude_m,
                              release_delay_s, gravity_mps2=9.80665):
    """Translation before release; constant horizontal velocity, zero vertical v.

    Target is (0,0,0). This is an ideal position-setpoint correction, not actuator
    dynamics. Altitude/delay/gravity are fixed declared scenario constants.
    """
    state = np.asarray(state_at_start, dtype=float)
    if state.shape != (4,) or not np.isfinite(state).all():
        raise ValueError('finite four-dimensional state required')
    if altitude_m <= 0 or release_delay_s < 0 or observation_s < 0 or gravity_mps2 <= 0:
        raise ValueError('invalid drop constants')
    flight = np.sqrt(2*altitude_m/gravity_mps2)
    return -(state[:2]+state[2:]*(observation_s+release_delay_s+flight))


def terrain_landing(terrain, position_ne, velocity_ne, altitude_m, release_delay_s,
                    gravity_mps2=9.80665):
    """Evaluation truth: first ballistic intersection with piecewise-linear DEM.

    No drag/wind, horizontal inherited velocity, zero vertical release velocity.
    Exact interval boundaries include all DEM grid and diagonal crossings. In
    each interval vertical clearance is a concave quadratic, so the first sign
    crossing identifies the first contact (positive starting clearance).
    """
    n, e, h = map(np.asarray, (terrain.north_m, terrain.east_m, terrain.elevation_up_m))
    p = np.asarray(position_ne, float) + np.asarray(velocity_ne, float)*release_delay_s
    v = np.asarray(velocity_ne, float)
    limit = np.sqrt(2*(altitude_m-float(h.min()))/gravity_mps2)+1e-6
    events = [0., limit]
    for axis, grid in enumerate((n, e)):
        if v[axis] != 0:
            events.extend(t for t in (grid-p[axis])/v[axis] if 0 < t < limit)
    events = sorted(set(events))
    # Diagonals of each traversed DEM cell must also split the quadratic.
    extra = []
    for left, right in zip(events[:-1], events[1:]):
        xy = p+v*(left+right)/2
        if not (n[0] <= xy[0] <= n[-1] and e[0] <= xy[1] <= e[-1]):
            continue
        i = np.clip(np.searchsorted(n, xy[0])-1, 0, len(n)-2)
        j = np.clip(np.searchsorted(e, xy[1])-1, 0, len(e)-2)
        dn, de = n[i+1]-n[i], e[j+1]-e[j]
        slope = v[0]/dn-v[1]/de
        if abs(slope) > 1e-15:
            t = ((p[1]-e[j])/de-(p[0]-n[i])/dn)/slope
            if left < t < right:
                extra.append(t)
    events = sorted(set(events+extra))

    def clearance(t):
        xy = p+v*t
        return altitude_m-.5*gravity_mps2*t*t-float(height_at(n, e, h, *xy))

    if clearance(0) <= 0:
        raise ValueError('release must be above terrain')
    for left, right in zip(events[:-1], events[1:]):
        if clearance(right) <= 0:
            t = brentq(clearance, left, right, xtol=1e-12)
            return p+v*t, t
    raise ValueError('no ground contact inside DEM')
