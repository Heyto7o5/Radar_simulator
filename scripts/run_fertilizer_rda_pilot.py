#!/usr/bin/env python3
"""Reproducible fixed-scene, RDA-only ideal fertilizer-position correction pilot."""
import argparse
import hashlib
import itertools
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from radar_platform.dem_buildings import BuildingSpec, add_dem_buildings
from radar_platform.dem_los_reference import height_at
from radar_platform.fertilizer_pilot import (RDACalibration, ideal_position_adjustment,
                                            rda_features, terrain_landing)
from radar_platform.unified.schema import Simulation
from radar_platform.unified.session import SimulationSession


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def make_scene(source, spacing=10.):
    raw = json.loads(source.read_text())
    t = raw['scene']['terrain']
    cover = np.asarray(t['land_cover_map'])
    # First 4x4 all-dominant-cropland window, no outcome-based site selection.
    choices = [(j, i) for j in range(cover.shape[0]-3) for i in range(cover.shape[1]-3)
               if (cover[j:j+4, i:i+4] == 'cropland').all()]
    if not choices:
        raise ValueError('no 120m all-dominant-cropland patch')
    j, i = choices[0]
    n, e, h = map(np.asarray, (t['north_m'], t['east_m'], t['elevation_up_m']))
    origin = np.array([n[i+2], e[j+2]])
    origin_h = h[j+2, i+2]
    axes = np.arange(-60, 60+spacing/2, spacing)
    nn, ee = np.meshgrid(axes+origin[0], axes+origin[1])
    hh = height_at(n, e, h, nn, ee)-origin_h
    config = Simulation.model_validate(dict(
        name='Fixed cropland DSM: ideal no-wind fertilizer correction pilot',
        input_provenance=raw['input_provenance'][:4]+[
            f'Parent config SHA256={hashlib.sha256(source.read_bytes()).hexdigest()}',
            f'First row-major all-cropland 4x4 classified cells: row={j},col={i}; '
            f'local origin NE={origin.tolist()}, parent elevation={origin_h} m.',
            f'{spacing}m numerical tessellation of original 30m piecewise-linear DSM; no new measured detail.',
            'Land-cover classification is not field inspection. Uniform relative ground reflectivity is assumed.',
            'Fixed acquisition, altitude 30m above center, level northward approach; no unknown attitude.',
            'Offline scene calibration encodes DEM-center target. Runtime predictor receives only numeric RDA.',
            'Ideal drop: no wind or drag, zero vertical release velocity, inherited horizontal velocity, delay 0.2s.',
            'Ideal instantaneous horizontal setpoint adjustment; no actuator dynamics or fertilizer radar returns.'],
        waveform=dict(kind='fmcw', start_hz=9.9e9, slope_hz_s=3.125e12,
                      ramp_s=80e-6, slot_period_s=250e-6, adc_start_s=5e-6,
                      sample_rate_hz=2e6, samples=128, loops=32),
        platform=dict(position_ned_m=[-11, 0, -30], velocity_ned_mps=[3, 0, 0],
                      boresight_ned=[0, 0, 1], right_hint_ned=[0, 1, 0], beam_half_angle_deg=65),
        array=dict(layout='ula', num_tx=2, num_rx=8),
        scene=dict(seed=42, max_scatterers=4096, amplitude_reference_range_m=30,
                   terrain=dict(kind='dem', north_m=axes.tolist(), east_m=axes.tolist(),
                                elevation_up_m=hh.tolist(), relative_reflectivity=.002)),
        visibility=dict(occlusion_enabled=True, report_coverage=True, occluders='all_surfaces',accelerator='bvh'),
        processing=dict(range_limits_m=[15, 95], azimuth_deg=list(range(-60, 61, 4)),
                        elevation_deg=[0.], save_adc=True),
        sequence=dict(frame_count=2, frame_interval_s=1)))
    specs = [BuildingSpec(id=f'landmark_{k}', center_ne_m=xy, footprint_ne_m=sz,
                         height_above_highest_ground_m=ht, max_foundation_relief_m=10,
                         relative_reflectivity=.02, divisions=2)
             for k, (xy, sz, ht) in enumerate([
                 ((8, 22), (6, 8), 5), ((-24, -18), (8, 5), 4), ((26, -16), (5, 7), 6)])]
    built, placement = add_dem_buildings(config, specs)
    return config, built, dict(parent_origin_ne_m=origin.tolist(), parent_origin_up_m=float(origin_h),
                               cell_row=j, cell_col=i, placement=placement,
                               target_ne_m=[0, 0], dominant_cropland_cells=16)


def acquire(base, state, folder, save_full=False, seed=42):
    folder.mkdir(parents=True, exist_ok=True)
    data = base.model_dump(mode='json')
    data['platform']['position_ned_m'] = [*state[:2], -30.]
    data['platform']['velocity_ned_mps'] = [*state[2:], 0.]
    data['scene']['seed'] = seed
    config = Simulation.model_validate(data)
    config_text = config.model_dump_json(indent=2)
    digest = hashlib.sha256(config_text.encode()).hexdigest()
    if (folder/'rda.npz').exists():
        meta = json.loads((folder/'summary.json').read_text())
        if meta['config_sha256'] != digest:
            raise ValueError('existing case config mismatch')
        with np.load(folder/'rda.npz') as f:
            return f['magnitude'], meta
    session = SimulationSession(config)
    session.save_inputs(folder)
    magnitudes, energies = [], []
    started = time.perf_counter()
    for index in range(2):
        frame = session.frame(index)
        magnitudes.append(np.abs(frame.products.rad[:, :, 0, :]).astype('float32'))
        energies.append(float(np.sum(np.abs(frame.adc)**2)))
        if save_full:
            session.save_frame(folder, frame)
    rda = np.stack(magnitudes)
    meta = dict(config_sha256=digest, elapsed_s=time.perf_counter()-started,
                applicability=session.estimate['applicability'], adc_energy=energies)
    write(folder/'summary.json', meta)
    np.savez_compressed(folder/'rda.npz', magnitude=rda,
                        range_m=frame.products.range_m, velocity_mps=frame.products.velocity_mps,
                        azimuth_deg=frame.products.azimuth_deg, frame_times_s=[0., 1.])
    return rda, meta


def acquire_job(job):
    name, base, state, folder, save_full = job
    _, meta = acquire(base, state, folder, save_full)
    return dict(condition=name, case=folder.name, seconds=meta['elapsed_s'],
                applicability=meta['applicability']['status'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--workers', type=int, default=8)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    bare, built, provenance = make_scene(args.source)
    write(args.output/'scene_provenance.json', provenance)
    for name, cfg in [('terrain', bare), ('buildings', built)]:
        (args.output/f'{name}_config.json').write_text(cfg.model_dump_json(indent=2)+'\n')
    center = np.array([-3*(1+.2+np.sqrt(60/9.80665)), 0, 3, 0])
    if args.smoke:
        for name, cfg in [('terrain', bare), ('buildings', built)]:
            _, meta = acquire(cfg, center, args.output/name/'smoke', save_full=True)
            print(json.dumps(dict(condition=name, **meta)), flush=True)
        return
    train = np.array([center+np.array(x) for x in itertools.product([-3., 0., 3.], [-3., 0., 3.], [-.6, 0., .6], [-.6, 0., .6])])
    rng = np.random.default_rng(20261028)
    tests = center+rng.uniform([-2.8, -2.8, -.55, -.55], [2.8, 2.8, .55, .55], (12, 4))
    protocol = dict(calibration_states=train.tolist(), heldout_states=tests.tolist(),
                    neighbors=4, target=[0, 0], altitude_m=30, release_delay_s=.2,
                    observation_s=1, phase_seed=42,
                    split='continuous heldout states, same fixed geography; no cross-scene claim')
    write(args.output/'protocol.json', protocol)
    jobs = []
    for name, cfg in [('terrain', bare), ('buildings', built)]:
        for prefix, states, full_index in [('calibration', train, 40), ('test', tests, 0)]:
            jobs.extend((name, cfg, state, args.output/name/f'{prefix}_{k:03d}', k==full_index)
                        for k, state in enumerate(states))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(acquire_job, job) for job in jobs]
        for done, future in enumerate(as_completed(pending), 1):
            print(json.dumps(dict(done=done, total=len(jobs), **future.result())), flush=True)
    results = []
    for name, cfg in [('terrain', bare), ('buildings', built)]:
        features = []
        for k, state in enumerate(train):
            rda, meta = acquire(cfg, state, args.output/name/f'calibration_{k:03d}', save_full=k==40)
            features.append(rda_features(rda))
        features = np.asarray(features)
        np.savez_compressed(args.output/name/'calibration.npz', features=features, states=train)
        model = RDACalibration(features, train, neighbors=4)
        for k, state in enumerate(tests):
            rda, meta = acquire(cfg, state, args.output/name/f'test_{k:03d}', save_full=k==0)
            predicted, diagnostic = model.predict(rda)
            shift = ideal_position_adjustment(predicted, observation_s=1, altitude_m=30, release_delay_s=.2)
            current = state[:2]+state[2:]
            before, _ = terrain_landing(cfg.scene.terrain, current, state[2:], 30, .2)
            after, _ = terrain_landing(cfg.scene.terrain, current+shift, state[2:], 30, .2)
            oracle = ideal_position_adjustment(state, observation_s=1, altitude_m=30, release_delay_s=.2)
            oracle_land, _ = terrain_landing(cfg.scene.terrain, current+oracle, state[2:], 30, .2)
            row = dict(condition=name, test=k, truth_state=state.tolist(), predicted_state=predicted.tolist(),
                       correction_ne_m=shift.tolist(), before_ne_m=before.tolist(), after_ne_m=after.tolist(),
                       before_error_m=float(np.linalg.norm(before)), after_error_m=float(np.linalg.norm(after)),
                       oracle_error_m=float(np.linalg.norm(oracle_land)), **diagnostic)
            results.append(row)
            write(args.output/'results.json', results)
            print(json.dumps(row), flush=True)
    summary = {}
    for name in ['terrain', 'buildings']:
        rows = [r for r in results if r['condition']==name]
        summary[name] = dict(cases=len(rows), before_mean_m=float(np.mean([r['before_error_m'] for r in rows])),
                            after_mean_m=float(np.mean([r['after_error_m'] for r in rows])),
                            after_p90_m=float(np.percentile([r['after_error_m'] for r in rows], 90)),
                            improved=sum(r['after_error_m'] < r['before_error_m'] for r in rows),
                            oracle_max_m=max(r['oracle_error_m'] for r in rows))
    write(args.output/'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
