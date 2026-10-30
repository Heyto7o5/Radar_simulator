"""Bounded kinematic plant. State is simulation truth, never an estimator input."""
from dataclasses import dataclass, replace
import numpy as np
from scipy.spatial.transform import Rotation

from ..unified.trajectory import RigidTrajectory


def body_to_neu(yaw_rad):
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])


def radar_to_ned(yaw_rad):
    # Radar x points down; y points aircraft-right; z points aircraft-aft.
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[0.,-s,-c],[0.,c,-s],[1.,0.,0.]])


@dataclass(frozen=True)
class State:
    time_s: float
    position_neu_m: np.ndarray
    velocity_neu_mps: np.ndarray
    yaw_rad: float

    def label(self):
        rotation = body_to_neu(self.yaw_rad)
        return np.r_[rotation.T @ -self.position_neu_m, rotation.T @ self.velocity_neu_mps]


@dataclass(frozen=True)
class Limits:
    horizontal_speed_mps: float = 3.
    vertical_speed_mps: float = 1.
    horizontal_acceleration_mps2: float = 2.
    vertical_acceleration_mps2: float = 1.
    yaw_rate_deg_s: float = 45.
    integration_step_s: float = .02


@dataclass(frozen=True)
class Command:
    acceleration_body_mps2: np.ndarray
    yaw_rate_rad_s: float

    @classmethod
    def coast(cls):
        return cls(np.zeros(3), 0.)


def clip_horizontal(vector, maximum):
    vector = np.array(vector, float, copy=True)
    norm = np.linalg.norm(vector[:2])
    if norm > maximum:
        vector[:2] *= maximum/norm
    return vector


def propagate(state, command, end_s, limits=Limits()):
    if not np.isfinite(end_s) or end_s <= state.time_s:
        raise ValueError('plant must advance forward in time')
    acceleration = clip_horizontal(command.acceleration_body_mps2, limits.horizontal_acceleration_mps2)
    acceleration[2] = np.clip(acceleration[2], -limits.vertical_acceleration_mps2, limits.vertical_acceleration_mps2)
    yaw_rate = float(np.clip(command.yaw_rate_rad_s, -np.deg2rad(limits.yaw_rate_deg_s), np.deg2rad(limits.yaw_rate_deg_s)))
    samples = [state]
    steps = max(1, int(np.ceil((end_s-state.time_s)/limits.integration_step_s)))
    times = np.linspace(state.time_s, end_s, steps+1)
    for start, end in zip(times[:-1], times[1:]):
        dt = end-start
        world_a = body_to_neu(state.yaw_rad+yaw_rate*dt/2) @ acceleration
        velocity = state.velocity_neu_mps+world_a*dt
        # Only truncate a commanded acceleration that would exceed the cap;
        # an initial over-speed state decelerates continuously, without teleportation.
        cap = max(limits.horizontal_speed_mps, np.linalg.norm(state.velocity_neu_mps[:2]))
        velocity = clip_horizontal(velocity, cap)
        zcap = max(limits.vertical_speed_mps, abs(state.velocity_neu_mps[2]))
        velocity[2] = np.clip(velocity[2], -zcap, zcap)
        state = State(float(end), state.position_neu_m+(state.velocity_neu_mps+velocity)*dt/2,
                      velocity, state.yaw_rad+yaw_rate*dt)
        samples.append(state)
    return state, samples


def sampled_trajectory(states):
    times = np.array([s.time_s for s in states])
    if len(states)<2 or np.any(np.diff(times)<=0):
        raise ValueError('strictly increasing plant samples required')
    sign = np.array([1.,1.,-1.])
    return RigidTrajectory(time_s=times.tolist(),
        position_ned_m=[(s.position_neu_m*sign).tolist() for s in states],
        velocity_ned_mps=[(s.velocity_neu_mps*sign).tolist() for s in states],
        quaternion_xyzw=Rotation.from_matrix(np.stack([radar_to_ned(s.yaw_rad) for s in states])).as_quat().tolist(),
        provenance='Continuously integrated bounded civil kinematic plant, absolute simulation clock; no full rotor dynamics')
