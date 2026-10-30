"""Compatibility at the boundary, not dataset-specific logic in the simulator."""
import numpy as np
from .schema import Simulation
from .geometry import sensor_basis


def from_pulsed_request(old):
    """Convert legacy point/flat-patch scene; retain the historical endpoint itself."""
    from ..physics import platform_velocity_ned, platform_boresight_ned
    from ..simulator import _clutter_scatterers
    w = old.radar
    direction = platform_boresight_ned(old.platform)
    right = np.eye(3)[np.argmin(abs(direction))]
    objects = [{'kind': 'point', **t.model_dump(mode='json')} for t in old.scene.targets]
    background = _clutter_scatterers(old, np.array(old.platform.position_ned_m))
    if background:
        objects.append({'kind':'cloud', 'id':'legacy_ground_patch',
            'points_local_m':[s.position_ned_m.tolist() for s in background],
            'amplitudes':[s.amplitude for s in background], 'phases_rad':[s.phase_rad for s in background]})
    samples = int(np.ceil(w.sample_rate_hz/w.prf_hz))
    return Simulation.model_validate({
        'name': 'converted legacy pulsed scene',
        'waveform': {'kind':'pulsed_lfm', 'carrier_hz':w.carrier_hz, 'bandwidth_hz':w.bandwidth_hz,
            'sample_rate_hz':w.sample_rate_hz,'pulse_width_s':w.pulse_width_s,'slot_period_s':1/w.prf_hz,
            'adc_start_s':0.,'samples':samples,'loops':w.pulses_per_cpi},
        'platform':{'position_ned_m':old.platform.position_ned_m,'velocity_ned_mps':platform_velocity_ned(old.platform).tolist(),
            'boresight_ned':direction.tolist(),'right_hint_ned':right.tolist(),'beam_half_angle_deg':w.beam_half_angle_deg},
        'array':{'layout':'siso','num_tx':1,'num_rx':1},
        'scene':{'objects':objects,'seed':old.scene.seed,
                 'amplitude_reference_range_m':max(1.,-old.platform.position_ned_m[2]/max(direction[2],1e-9))},
        'processing':{'range_limits_m':[0,299792458/(2*w.prf_hz)],'azimuth_deg':[0.],'elevation_deg':[0.],
                      'range_window':'rectangular'},
        'sequence':{'frame_count':old.sequence.frame_count,'frame_interval_s':old.sequence.frame_interval_s},
    })


def from_array_request(old):
    """Legacy array positions are NED offsets; rotate into the declared sensor basis."""
    from .schema import Platform
    w, a = old.radar.config(), old.array()
    direction = np.asarray(old.boresight_ned, dtype=float)
    direction /= np.linalg.norm(direction)
    hint = (0.,1.,0.) if abs(direction[1]) < .99 else (0.,0.,1.)
    if old.beam_half_angle_deg > 89.:
        raise ValueError('legacy beam wider than unified forward hemisphere; choose an explicit forward beam before conversion')
    platform = Platform(position_ned_m=old.platform_position_ned_m, velocity_ned_mps=old.platform_velocity_ned_mps,
                        boresight_ned=tuple(direction),right_hint_ned=hint,beam_half_angle_deg=old.beam_half_angle_deg)
    basis = sensor_basis(platform)
    radius = min(np.linalg.norm(np.array(t.position_ned_m)-np.array(old.platform_position_ned_m)) for t in old.targets)
    return Simulation.model_validate({
        'name':'converted legacy FMCW array scene',
        'waveform':{'kind':'fmcw','start_hz':w.start_hz,'slope_hz_s':w.slope_hz_s,'ramp_s':w.ramp_s,
                    'slot_period_s':w.chirp_period_s,'adc_start_s':w.adc_start_s,'sample_rate_hz':w.sample_rate_hz,
                    'samples':w.samples,'loops':w.chirps},
        'platform':platform.model_dump(),
        'array':{'layout':'custom','num_tx':old.num_tx,'num_rx':old.num_rx,'tx_local_m':(a.tx_ned_m@basis).tolist(),
                 'rx_local_m':(a.rx_ned_m@basis).tolist(),'tx_offsets_s':a.tx_offsets_s.tolist(),'loop_period_s':a.loop_period_s},
        'scene':{'objects':[{'kind':'point',**t.model_dump()} for t in old.targets], 'amplitude_reference_range_m':radius},
        'processing':{'range_limits_m':[0,w.sample_rate_hz*299792458/(2*w.slope_hz_s)],
                      'azimuth_deg':old.azimuth_deg,'elevation_deg':[0.],'tdm_compensation':old.tdm_compensation},
        'sequence':{'start_time_s':old.frame_start_s},
    })
