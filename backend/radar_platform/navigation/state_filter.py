"""Temporal radar state estimate using measured RDA and command history only.

Prediction is expressed in the previous *estimated body frame*. It constructs
no global pose and receives no simulator state or measured attitude/odometry.
"""
import numpy as np

from .controller import Estimate
from .dynamics import State,Limits,body_to_neu,propagate

FILTER_VERSION='radar_command_kalman_v1'


class RadarStateFilter:
    def __init__(self,measurement_std,limits=Limits()):
        std=np.asarray(measurement_std,float)
        if std.shape!=(6,) or not np.isfinite(std).all() or np.any(std<0):
            raise ValueError('six finite validation-derived measurement standard deviations required')
        self.std=np.maximum(std,[.3,.3,.2,.12,.12,.12])
        self.measurement_covariance=np.diag(self.std**2)
        self.limits=limits;self.mean=None;self.covariance=None;self.time_s=None

    def update(self,measurement,previous_command):
        if self.mean is None:
            if not measurement.valid:return measurement,dict(version=FILTER_VERSION,initialized=False)
            self.mean=np.r_[measurement.relative_position_body_m,measurement.velocity_body_mps].astype(float)
            self.covariance=self.measurement_covariance.copy();self.time_s=measurement.time_s
            return measurement,dict(version=FILTER_VERSION,initialized=True,measurement_used=True)
        dt=measurement.time_s-self.time_s
        if not np.isfinite(dt) or dt<=0:raise ValueError('radar filter requires increasing observation time')
        # Zero origin/yaw define an internal local frame, not knowledge of world pose.
        predicted,_=propagate(State(0.,np.zeros(3),self.mean[3:].copy(),0.),previous_command,dt,self.limits)
        rotation=body_to_neu(predicted.yaw_rad).T
        prior=np.r_[rotation@(self.mean[:3]-predicted.position_neu_m),rotation@predicted.velocity_neu_mps]
        transition=np.block([[rotation,-dt*rotation],[np.zeros((3,3)),rotation]])
        intensity=np.diag([.25,.25,.09])
        process=np.block([[dt**3/3*intensity,-dt**2/2*intensity],[-dt**2/2*intensity,dt*intensity]])
        covariance=transition@self.covariance@transition.T+process
        innovation=None
        if measurement.valid:
            observed=np.r_[measurement.relative_position_body_m,measurement.velocity_body_mps]
            if not np.isfinite(observed).all():raise ValueError('finite radar measurement required')
            innovation=observed-prior
            gain=np.linalg.solve(covariance+self.measurement_covariance,covariance).T
            prior=prior+gain@innovation
            identity=np.eye(6)-gain
            covariance=identity@covariance@identity.T+gain@self.measurement_covariance@gain.T
        self.mean=prior;self.covariance=(covariance+covariance.T)/2;self.time_s=measurement.time_s
        estimate=Estimate(measurement.time_s,prior[:3].copy(),prior[3:].copy(),measurement.valid,
                          measurement.error_scale_m,'radar_filtered' if measurement.valid else 'prediction_without_usable_radar')
        diagnostic=dict(version=FILTER_VERSION,initialized=True,measurement_used=measurement.valid,
            innovation=None if innovation is None else innovation.tolist(),
            state_std=np.sqrt(np.maximum(np.diag(self.covariance),0)).tolist(),
            input='RDA estimates, timestamps and previously issued bounded body-frame command only')
        return estimate,diagnostic
