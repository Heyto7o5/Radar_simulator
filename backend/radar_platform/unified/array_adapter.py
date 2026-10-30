"""Translate the legacy moving FMCW array API to a dataset-independent scene.

Array coordinates in that API are NED offsets, not sensor-local coordinates.
They must be rotated into the configured sensor basis without moving antennas.
"""
import numpy as np
from ..models import C0
from .schema import Simulation, Platform
from .geometry import sensor_basis


def motion_array_configuration(config, targets, array, *,
                               platform_position_ned_m=(0., 0., 0.),
                               platform_velocity_ned_mps=(0., 0., 0.),
                               frame_start_s=0., boresight_ned=(1., 0., 0.),
                               beam_half_angle_deg=89., azimuth_deg=None,
                               tdm_compensation=True, amplitude_reference_range_m=None,
                               allow_empty_scene=False):
    if not targets and not allow_empty_scene:
        raise ValueError('at least one target required')
    direction = np.asarray(boresight_ned, dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) <= 0:
        raise ValueError('nonzero finite boresight required')
    direction = direction/np.linalg.norm(direction)
    hint = [0., 1., 0.] if abs(direction[1]) < .99 else [1., 0., 0.]
    platform = Platform(position_ned_m=platform_position_ned_m,
                        velocity_ned_mps=platform_velocity_ned_mps,
                        boresight_ned=tuple(direction), right_hint_ned=tuple(hint),
                        beam_half_angle_deg=beam_half_angle_deg)
    basis = sensor_basis(platform)
    reference = amplitude_reference_range_m
    if reference is None:
        if not targets:raise ValueError('empty scene needs an explicit amplitude reference')
        reference = min(np.linalg.norm(np.asarray(t.position_ned_m)-platform_position_ned_m) for t in targets)
    if reference <= 0:
        raise ValueError('target intersects radar at reference time')
    return Simulation.model_validate(dict(
        name='Migrated moving FMCW array scene',
        waveform=dict(kind='fmcw', start_hz=config.start_hz, slope_hz_s=config.slope_hz_s,
                      ramp_s=config.ramp_s, slot_period_s=config.chirp_period_s,
                      adc_start_s=config.adc_start_s, sample_rate_hz=config.sample_rate_hz,
                      samples=config.samples, loops=config.chirps),
        platform=platform.model_dump(mode='json'),
        array=dict(layout='custom', num_tx=len(array.tx_ned_m), num_rx=len(array.rx_ned_m),
                   tx_local_m=(array.tx_ned_m@basis).tolist(),
                   rx_local_m=(array.rx_ned_m@basis).tolist(),
                   tx_offsets_s=array.tx_offsets_s.tolist(), loop_period_s=array.loop_period_s),
        scene=dict(amplitude_reference_range_m=float(reference),allow_empty=allow_empty_scene, objects=[
            dict(kind='point', id=t.id, position_ned_m=t.position_ned_m,
                 velocity_ned_mps=t.velocity_ned_mps, amplitude=t.amplitude) for t in targets]),
        processing=dict(range_limits_m=[0., config.sample_rate_hz*C0/(2*config.slope_hz_s)],
                        azimuth_deg=[0.] if azimuth_deg is None else list(azimuth_deg),
                        elevation_deg=[0.], tdm_compensation=tdm_compensation, save_adc=True),
        sequence=dict(frame_count=1, start_time_s=frame_start_s)))
