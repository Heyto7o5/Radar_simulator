#!/usr/bin/env python3
"""Frozen-calibration stress cases; never select a model using these outcomes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import numpy as np

from run_fertilizer_rda_pilot import acquire, write, make_scene
from radar_platform.fertilizer_pilot import RDACalibration, ideal_position_adjustment, terrain_landing
from radar_platform.unified.schema import Simulation


def generate(job):
    condition, base, state, folder, seed = job
    _, meta = acquire(base, state, folder, seed=seed)
    return dict(condition=condition, case=folder.name, elapsed_s=meta['elapsed_s'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--generate-only', action='store_true')
    p.add_argument('--source', type=Path, help='Also check 5m tessellation on first two states')
    args = p.parse_args()
    protocol = json.loads((args.run/'protocol.json').read_text())
    tests = np.asarray(protocol['heldout_states'])[:6]
    root = args.run/'stress'
    root.mkdir(exist_ok=True)
    configs = {name: Simulation.model_validate_json((args.run/f'{name}_config.json').read_text())
               for name in ['terrain', 'buildings']}
    conditions = []
    for name in configs:
        conditions.append((name+'_new_phase', name, configs[name], 31415, 6))
    for scale in [.5, 2.]:
        data = configs['buildings'].model_dump(mode='json')
        for obj in data['scene']['objects']:
            obj['relative_reflectivity'] *= scale
        conditions.append((f'buildings_reflectivity_{scale}', 'buildings', Simulation.model_validate(data), 42, 6))
    if args.source:
        fine_bare, fine_built, _ = make_scene(args.source, spacing=5.)
        conditions.extend([(name+'_grid5', name, cfg, 42, 2)
                           for name, cfg in [('terrain', fine_bare), ('buildings', fine_built)]])
    write(root/'protocol.json', dict(test_indices=list(range(6)), conditions=[x[0] for x in conditions],
                                   calibration='frozen main calibration; no retuning',
                                   new_phase='new fixed diffuse phase seed for both frames',
                                   reflectivity='amplitude coefficient multiplier, not dB',
                                   grid5='first two states only; same geometric surface, different scatter discretization and phases; not pure quadrature convergence'))
    jobs = [(label, cfg, state, root/label/f'test_{k:03d}', seed)
            for label, _, cfg, seed, count in conditions for k, state in enumerate(tests[:count])]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(generate, job) for job in jobs]):
            print(json.dumps(future.result()), flush=True)
    if args.generate_only:
        return
    models = {}
    for name in configs:
        with np.load(args.run/name/'calibration.npz') as f:
            models[name] = RDACalibration(f['features'], f['states'])
    rows = []
    for label, name, cfg, seed, count in conditions:
        for k, state in enumerate(tests[:count]):
            with np.load(root/label/f'test_{k:03d}'/'rda.npz') as f:
                predicted, diagnostic = models[name].predict(f['magnitude'])
            shift = ideal_position_adjustment(predicted, observation_s=1, altitude_m=30, release_delay_s=.2)
            current = state[:2]+state[2:]
            before, _ = terrain_landing(cfg.scene.terrain, current, state[2:], 30, .2)
            after, _ = terrain_landing(cfg.scene.terrain, current+shift, state[2:], 30, .2)
            rows.append(dict(condition=label, test=k, before_error_m=float(np.linalg.norm(before)),
                             after_error_m=float(np.linalg.norm(after)), predicted_state=predicted.tolist(), **diagnostic))
    write(root/'results.json', rows)
    summary = {label: dict(mean_after_m=float(np.mean([r['after_error_m'] for r in rows if r['condition']==label])),
                          improved=sum(r['after_error_m'] < r['before_error_m'] for r in rows if r['condition']==label), cases=count)
               for label, _, _, _, count in conditions}
    write(root/'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
