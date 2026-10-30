"""Translate the first V2 high-altitude inputs, not its echo implementation.

This is an explicit migration: the common engine includes within-pulse delay
motion and uses normalized matched-filter/FFT outputs. Bitwise legacy echo
equivalence is neither expected nor claimed.
"""
import math
import numpy as np
from ..models import SimulationRequest, C0
from ..physics import platform_velocity_ned, platform_boresight_ned
from .schema import Simulation


def convert_legacy(request: SimulationRequest) -> tuple[Simulation, list[str]]:
    radar, platform = request.radar, request.platform
    pri = 1/radar.prf_hz
    samples = int(math.floor(radar.sample_rate_hz*pri+1e-9))
    boresight = platform_boresight_ned(platform)
    position = np.asarray(platform.position_ned_m)
    right = np.array([0., 1., 0.])
    if np.linalg.norm(np.cross(boresight, right)) < 1e-6:
        right = np.array([1., 0., 0.])
    reference = max(1., -position[2]/max(boresight[2], 1e-9))
    objects = [dict(kind='point', id=t.id, position_ned_m=t.position_ned_m,
                    velocity_ned_mps=t.velocity_ned_mps, amplitude=t.amplitude)
               for t in request.scene.targets]
    notes = [
        'Migrated input parameters, not a bitwise reproduction of the legacy engine.',
        'Common engine includes within-pulse delay motion; legacy engine froze the chirp envelope per pulse.',
        'Common range/Doppler transforms are window-sum normalized; absolute legacy RD amplitude is not preserved.',
        'Truth-assisted legacy range corrections are not applied to common RD products.',
    ]
    if samples != math.ceil(radar.sample_rate_hz*pri):
        notes.append('ADC count rounded down to keep the receive gate within one PRI; less than one sample removed.')
    if request.scene.simplified_clutter_enabled:
        notes.append('Legacy clutter retained as seeded world-fixed point scatterers, not a calibrated surface model.')
        if boresight[2] > 0 and position[2] < 0:
            ground_distance = -position[2]/boresight[2]
            center = position[:2]+ground_distance*boresight[:2]
            radius = min(ground_distance*math.tan(math.radians(radar.beam_half_angle_deg)), 750.)
            coords = np.arange(-radius, radius+50., 50.)
            points = [[float(center[0]+dx), float(center[1]+dy), 0.]
                      for dx in coords for dy in coords if dx*dx+dy*dy <= radius*radius]
            if points:
                name = 'legacy_ground_clutter'
                while name in {obj['id'] for obj in objects}:
                    name += '_'
                phases = np.random.default_rng(request.scene.seed).uniform(-math.pi, math.pi, len(points))
                objects.append(dict(kind='cloud', id=name, points_local_m=points,
                                    amplitudes=[.012]*len(points), phases_rad=phases.tolist()))
    result = Simulation.model_validate(dict(
        name='Migrated legacy high-altitude scene',
        waveform=dict(kind='pulsed_lfm', carrier_hz=radar.carrier_hz,
                      bandwidth_hz=radar.bandwidth_hz, sample_rate_hz=radar.sample_rate_hz,
                      pulse_width_s=radar.pulse_width_s, slot_period_s=pri,
                      adc_start_s=0., samples=samples, loops=radar.pulses_per_cpi),
        platform=dict(position_ned_m=position.tolist(), velocity_ned_mps=platform_velocity_ned(platform).tolist(),
                      boresight_ned=boresight.tolist(), right_hint_ned=right.tolist(),
                      beam_half_angle_deg=radar.beam_half_angle_deg),
        array=dict(layout='siso', num_tx=1, num_rx=1),
        scene=dict(objects=objects, seed=request.scene.seed, amplitude_reference_range_m=reference),
        processing=dict(range_limits_m=[0., (samples-1)/radar.sample_rate_hz*C0/2],
                        range_window='rectangular', doppler_window='hann',
                        azimuth_deg=[0.], elevation_deg=[0.], save_adc=True),
        sequence=request.sequence.model_dump()))
    return result, notes
