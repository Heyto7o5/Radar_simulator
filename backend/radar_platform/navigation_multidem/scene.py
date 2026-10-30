"""Real geography and variable cuboids, accessible only to the simulator."""
import numpy as np

from ..dem_buildings import _clip
from ..dem_los_reference import height_at
from ..unified.schema import Simulation, FMCW, Box
from ..unified.scene import terrain_mesh
from ..navigation.io import read_json, local_path


def rotated_footprint_heights(terrain, footprint, yaw_deg):
    """Clip actual DEM triangles in the rotated building's horizontal frame."""
    size = np.asarray(footprint, float)
    angle = np.deg2rad(yaw_deg)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    corners = np.array([[-1, -1], [-1, 1], [1, 1], [1, -1]]) * size/2 @ rotation.T
    if (corners[:, 0].min() < terrain.north_m[0] or corners[:, 0].max() > terrain.north_m[-1]
            or corners[:, 1].min() < terrain.east_m[0] or corners[:, 1].max() > terrain.east_m[-1]):
        raise ValueError('rotated building footprint exceeds real DEM context')
    mesh, _ = terrain_mesh(terrain)
    triangles = mesh.triangles.copy()
    triangles[..., :2] = triangles[..., :2] @ rotation
    triangles[..., 2] *= -1  # NED down -> terrain elevation
    lo, hi = -size/2, size/2
    selected = np.all(triangles[..., :2].max(axis=1) >= lo, axis=1) & np.all(triangles[..., :2].min(axis=1) <= hi, axis=1)
    values = []
    for triangle in triangles[selected]:
        polygon = list(triangle)
        for axis in (0, 1):
            polygon = _clip(polygon, axis, lo[axis], 1)
            polygon = _clip(polygon, axis, hi[axis], -1)
        values.extend(point[2] for point in polygon)
    if not values:
        raise ValueError('building footprint has no real terrain support')
    return float(min(values)), float(max(values))


def make_scene(geography, record, *, numerical_spacing_m=10.):
    root = local_path(geography)
    source = root/'regions'/record['region_id']
    parent, provenance = read_json(source/'parent.json'), read_json(source/'provenance.json')
    if record['split'] != provenance['region']['split']:
        raise ValueError('scene split differs from frozen geographic region')
    t = parent['scene']['terrain']
    north, east, heights = map(np.asarray, (t['north_m'], t['east_m'], t['elevation_up_m']))
    origin = np.asarray(provenance['selected_origin_ne_m'])
    h0 = float(height_at(north, east, heights, origin[0], origin[1]))
    axis = np.arange(-210., 210.+numerical_spacing_m/2, numerical_spacing_m)
    nn, ee = np.meshgrid(axis+origin[0], axis+origin[1])
    if nn.min() < north[0] or nn.max() > north[-1] or ee.min() < east[0] or ee.max() > east[-1]:
        raise ValueError('scene requires terrain outside downloaded context')
    local_heights = height_at(north, east, heights, nn, ee)-h0
    waveform = FMCW(start_hz=9.9e9, slope_hz_s=3.125e12, ramp_s=80e-6,
                    slot_period_s=100e-6, adc_start_s=5e-6, sample_rate_hz=3e6, samples=192, loops=64)
    d = waveform.wavelength_m/2
    config = Simulation(name='Multiple real DSM regions, RDA tensor-only learning', waveform=waveform,
        array=dict(layout='custom', num_tx=4, num_rx=16,
                   tx_local_m=[[0., y*d, z*d] for z in [-2., 2.] for y in [-2., 2.]],
                   rx_local_m=[[0., y*d, z*d] for z in [-1.5, -.5, .5, 1.5] for y in [-1.5, -.5, .5, 1.5]]),
        platform=dict(position_ned_m=[-24, -18, -30], velocity_ned_mps=[0, 3, 0],
                      boresight_ned=[0, 0, 1], right_hint_ned=[0, 1, 0], beam_half_angle_deg=60),
        scene=dict(seed=record['phase_seed'], max_scatterers=16384, amplitude_reference_range_m=30,
                   terrain=dict(kind='dem', max_grid_vertices=16384, north_m=axis.tolist(), east_m=axis.tolist(),
                                elevation_up_m=local_heights.tolist(), relative_reflectivity=record['terrain_reflectivity'])),
        visibility=dict(occlusion_enabled=True, report_coverage=True, occluders='all_surfaces',
                        accelerator='bvh', bvh_kernel='numba', max_ray_triangle_tests=500_000_000),
        processing=dict(range_limits_m=[8, 120], azimuth_deg=list(range(-60, 61, 6)),
                        elevation_deg=list(range(-60, 61, 6)), save_adc=True),
        sequence=dict(frame_count=1, frame_interval_s=.25),
        receiver_noise=dict(variance=record['noise_variance'], seed=record['noise_seed'],
                            provenance='Assumed independent complex ADC noise; not per-scene signal-normalized'),
        save_components=False, input_provenance=parent['input_provenance'] + [
            'Selected real cropland context; measured DSM unchanged apart from origin subtraction and interpolation.',
            'Numerical 10m surface tessellation is not new measured terrain resolution.',
            'Variable building dimensions are simulator truth only and never used by the online estimator.'])
    b = record['building']
    low, high = rotated_footprint_heights(config.scene.terrain, b['footprint_m'], b['yaw_deg'])
    roof = high+b['height_m']
    config.scene.objects.append(Box(id='center_landmark', position_ned_m=[0, 0, -(low+roof)/2],
        size_m=[*b['footprint_m'], roof-low], yaw_deg=b['yaw_deg'], divisions=2,
        relative_reflectivity=b['relative_reflectivity'], scattering='diffuse'))
    metadata = dict(region_id=record['region_id'], split=record['split'], ground_target_neu_m=[0, 0, 0],
                    parent_origin_ne_m=origin.tolist(), ground_min_m=low, ground_max_m=high, roof_m=roof,
                    building=b, inference_use='none: all geometry remains simulation/evaluation side')
    return config, metadata
