"""Reusable scenarios, never measured-data-derived special cases."""
from .schema import Simulation


def high_altitude(waveform='pulsed_lfm', terrain='hills', planar=True):
    return Simulation.model_validate({
        'name': f'10GHz high-altitude mixed scene / {waveform} / {terrain}',
        'waveform': {'kind': waveform},
        'platform': {'position_ned_m': [0, 0, -5000], 'velocity_ned_mps': [20, 0, 0],
                     'boresight_ned': [.15, 0, 1], 'beam_half_angle_deg': 12},
        'array': {'layout': 'ura' if planar else 'ula', 'num_tx': 4 if planar else 2, 'num_rx': 4},
        'scene': {
            'amplitude_reference_range_m': 5000,
            'terrain': {'kind': terrain, 'north_bounds_m': [350, 1150], 'east_bounds_m': [-400, 400],
                        'spacing_m': 100, 'relative_reflectivity': .003,
                        **({'hills': [{'center_ne_m': [550, -160], 'height_m': 120, 'sigma_ne_m': [160, 180]}]}
                           if terrain == 'hills' else {})},
            'objects': [
                {'kind': 'box', 'id': 'building', 'position_ned_m': [950, 160, -30], 'size_m': [100, 70, 60],
                 'relative_reflectivity': .08, 'divisions': 3},
                {'kind': 'point', 'id': 'moving_target', 'position_ned_m': [800, -200, -45],
                 'velocity_ned_mps': [-10, 8, 0], 'amplitude': 3},
                {'kind': 'plate', 'id': 'reference_plate', 'position_ned_m': [1050, -200, -3],
                 'size_m': [20, 20], 'relative_reflectivity': .12, 'divisions': 2},
            ],
        },
        'processing': {'range_limits_m': [4800, 5250], 'azimuth_deg': list(range(-12, 13, 2)),
                       'elevation_deg': list(range(-12, 13, 2)) if planar else [0]},
        'sequence': {'frame_count': 4, 'frame_interval_s': .5},
    })


def presets():
    return {f'high_altitude_{waveform}_{terrain}': high_altitude(waveform, terrain).model_dump(mode='json')
            for waveform in ['pulsed_lfm', 'fmcw'] for terrain in ['flat', 'hills']}
