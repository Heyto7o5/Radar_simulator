"""Guidance reads estimates and its own history, with no world-state argument."""
from dataclasses import dataclass
import numpy as np

from .dynamics import Command, Limits, clip_horizontal


@dataclass(frozen=True)
class Estimate:
    time_s: float
    relative_position_body_m: np.ndarray
    velocity_body_mps: np.ndarray
    valid: bool = True
    error_scale_m: float = 0.
    reason: str = 'ok'


class Controller:
    def __init__(self, limits=Limits(), target_height_m=30.,gentle_terminal=False):
        self.limits = limits
        self.target_height_m = target_height_m
        self.last_valid_s = None
        self.last_velocity = np.zeros(3)
        self.settled_since = None
        self.gentle_terminal=gentle_terminal
        self.position_gain=.25 if gentle_terminal else .55
        self.arrival_radius=1. if gentle_terminal else 2.
        self.arrival_height=.75 if gentle_terminal else 1.5
        self.arrival_speed=.15 if gentle_terminal else .3
        self.arrival_dwell_s=4. if gentle_terminal else 3.

    def step(self, estimate):
        if not estimate.valid:
            self.settled_since = None
            acceleration = clip_horizontal(-self.last_velocity/.6,self.limits.horizontal_acceleration_mps2)
            acceleration[2] = np.clip(acceleration[2],-self.limits.vertical_acceleration_mps2,self.limits.vertical_acceleration_mps2)
            command = Command(acceleration,0.)
            return command, dict(mode='reacquire', estimated_arrival=False)
        r = np.asarray(estimate.relative_position_body_m,float)
        velocity = np.asarray(estimate.velocity_body_mps,float)
        if r.shape!=(3,) or velocity.shape!=(3,) or not np.isfinite(np.r_[r,velocity]).all():
            raise ValueError('finite three-dimensional estimate required')
        self.last_velocity = velocity.copy()
        self.last_valid_s = estimate.time_s
        error = r+np.array([0.,0.,self.target_height_m])
        desired = clip_horizontal(self.position_gain*error,self.limits.horizontal_speed_mps)
        desired[2] = np.clip(self.position_gain*error[2],-self.limits.vertical_speed_mps,self.limits.vertical_speed_mps)
        if self.gentle_terminal:
            if np.linalg.norm(error[:2])<=.75:desired[:2]=0.
            if abs(error[2])<=.3:desired[2]=0.
        acceleration = (desired-velocity)/.7
        acceleration = clip_horizontal(acceleration,self.limits.horizontal_acceleration_mps2)
        acceleration[2] = np.clip(acceleration[2],-self.limits.vertical_acceleration_mps2,self.limits.vertical_acceleration_mps2)
        yaw_rate = 0. if np.linalg.norm(error[:2])<3. else 1.2*np.arctan2(error[1],error[0])
        yaw_rate = float(np.clip(yaw_rate,-np.deg2rad(self.limits.yaw_rate_deg_s),np.deg2rad(self.limits.yaw_rate_deg_s)))
        settled = np.linalg.norm(error[:2])<=self.arrival_radius and abs(error[2])<=self.arrival_height and np.linalg.norm(velocity)<=self.arrival_speed
        if not settled:
            self.settled_since = None
        elif self.settled_since is None:
            self.settled_since = estimate.time_s
        arrived = self.settled_since is not None and estimate.time_s-self.settled_since>=self.arrival_dwell_s
        return Command(acceleration,yaw_rate),dict(mode='hold' if settled else 'approach',
            estimated_arrival=bool(arrived),error_body_m=error.tolist(),desired_velocity_body_mps=desired.tolist())
