"""Frozen civil navigation scene from a locally embedded real DSM crop."""
import hashlib
import numpy as np

from ..dem_buildings import BuildingSpec, add_dem_buildings
from ..dem_los_reference import height_at
from ..unified.schema import Simulation, FMCW
from .io import local_path, read_json

SOURCE = '.runtime/stage2_20261023/configs_v1/foothill_hamming_0_s23.json'


def navigation_scene(*, source=SOURCE, terrain_only=False, spacing=10., seed=42):
    path = local_path(source)
    parent = read_json(path)
    t = parent['scene']['terrain']
    cover = np.asarray(t['land_cover_map'])
    choices = [(j, i) for j in range(cover.shape[0]-3) for i in range(cover.shape[1]-3)
               if (cover[j:j+4, i:i+4] == 'cropland').all()]
    if not choices:
        raise ValueError('no all-dominant-cropland 4x4 classified window')
    j, i = choices[0]
    n, e, h = map(np.asarray, (t['north_m'], t['east_m'], t['elevation_up_m']))
    origin = np.array([n[i+2], e[j+2]])
    origin_h = float(h[j+2, i+2])
    axes = np.arange(-210., 210.+spacing/2, spacing)
    nn, ee = np.meshgrid(axes+origin[0], axes+origin[1])
    hh = height_at(n, e, h, nn, ee)-origin_h
    nc = np.searchsorted(n, (nn[:-1, :-1]+nn[1:, 1:])/2, side='right')-1
    ec = np.searchsorted(e, (ee[:-1, :-1]+ee[1:, 1:])/2, side='right')-1
    local_cover = cover[ec, nc]
    waveform = FMCW(start_hz=9.9e9, slope_hz_s=3.125e12, ramp_s=80e-6,
                    slot_period_s=100e-6, adc_start_s=5e-6,
                    sample_rate_hz=3e6, samples=192, loops=64)
    d = waveform.wavelength_m/2
    tx = [[0., y*d, z*d] for z in [-2., 2.] for y in [-2., 2.]]
    rx = [[0., y*d, z*d] for z in [-1.5, -.5, .5, 1.5] for y in [-1.5, -.5, .5, 1.5]]
    provenance = parent['input_provenance'][:4] + [
        f'Local parent config SHA256={hashlib.sha256(path.read_bytes()).hexdigest()}',
        f'Fixed agricultural center: parent NE={origin.tolist()}, up={origin_h}; '
        f'first all-cropland 4x4 classified window row={j},col={i}.',
        f'420m context, {spacing}m tessellation of measured 30m piecewise-linear DSM; no new measured detail.',
        'Central 120m classified agricultural patch; surrounding actual classes retained in provenance sidecar.',
        'Uniform assumed relative terrain reflectivity 0.002; land class is not a scattering calibration.',
        'Civil navigation: target is 30m above DEM center; radar rotates in yaw, level airframe.',
        'Frozen diffuse phases throughout each episode; receiver noise indexed independently by global frame.',
        'Runtime estimator consumes only numeric radar image magnitudes, acquisition axes and timestamps.']
    cfg = Simulation(name='Real DSM civil closed-loop navigation with yawing planar radar',
        waveform=waveform, array=dict(layout='custom', num_tx=4, num_rx=16,
                                     tx_local_m=tx, rx_local_m=rx),
        platform=dict(position_ned_m=[-24,-18,-30], velocity_ned_mps=[0,3,0],
                      boresight_ned=[0,0,1], right_hint_ned=[0,1,0], beam_half_angle_deg=60),
        scene=dict(seed=seed, max_scatterers=16384, amplitude_reference_range_m=30,
                   terrain=dict(kind='dem', max_grid_vertices=16384, north_m=axes.tolist(),
                                east_m=axes.tolist(), elevation_up_m=hh.tolist(), relative_reflectivity=.002)),
        visibility=dict(occlusion_enabled=True, report_coverage=True, occluders='all_surfaces',
                        accelerator='bvh', bvh_kernel='numba', max_ray_triangle_tests=500_000_000),
        processing=dict(range_limits_m=[8,120], azimuth_deg=list(range(-60,61,6)),
                        elevation_deg=list(range(-60,61,6)), save_adc=True),
        sequence=dict(frame_count=1, frame_interval_s=.25),
        receiver_noise=dict(variance=1e-5, seed=seed,
                            provenance='Fixed assumed complex ADC noise variance, not signal-normalized SNR'),
        save_components=False, input_provenance=provenance)
    placement = []
    if not terrain_only:
        cfg, placement = add_dem_buildings(cfg, [BuildingSpec(id='center_landmark', center_ne_m=(0,0),
            footprint_ne_m=(6,8), height_above_highest_ground_m=5, max_foundation_relief_m=10,
            relative_reflectivity=.02, divisions=2)])
    meta = dict(source=str(path.relative_to(local_path('.'))), parent_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                parent_origin_ne_m=origin.tolist(), parent_origin_up_m=origin_h, cell_row=j, cell_col=i,
                land_cover_map=local_cover.tolist(), cover_counts={c:int((local_cover==c).sum()) for c in np.unique(local_cover)},
                classified_field_bounds_ne_m=[[-60,-60],[60,60]], context_bounds_ne_m=[[-210,-210],[210,210]],
                placement=placement, measured_grid_m=30, numerical_spacing_m=spacing,
                ground_target_neu_m=[0,0,0], hover_target_neu_m=[0,0,30])
    return cfg, meta
