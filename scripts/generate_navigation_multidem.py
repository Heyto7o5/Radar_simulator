#!/usr/bin/env python3
"""Generate cross-region RDA observations with split labels fixed before echoes."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
import hashlib
import json
import os
import time

from radar_platform.navigation.io import local_path, read_json, write_json
from radar_platform.navigation_multidem.data import records_for_plan, generate_multidem_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geography', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--train-per-region', type=int, default=512)
    parser.add_argument('--val-per-region', type=int, default=128)
    parser.add_argument('--test-per-region', type=int, default=128)
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train', 'val'])
    args = parser.parse_args()
    geography, output = local_path(args.geography), local_path(args.output)
    if read_json(geography/'status.json')['status'] != 'complete':
        raise ValueError('all declared geographic sources must finish before radar generation')
    plan = read_json(geography/'plan.json')
    records = records_for_plan(plan, train_per_region=args.train_per_region, val_per_region=args.val_per_region,
                               test_per_region=args.test_per_region)
    # Freeze test recipes, but do not generate/read test echoes while training.
    protocol = dict(plan=plan, records=records, input='numeric RDA only; no scene/commands/pose',
                    labels='offline body-relative ground XYZ and ego velocity at latest CPI end',
                    geography=str(geography.relative_to(local_path('.'))),
                    counts={s: sum(r['split'] == s for r in records) for s in ['train', 'val', 'test']})
    if (output/'protocol.json').exists() and read_json(output/'protocol.json') != protocol:
        raise ValueError('frozen dataset protocol differs; use a new output')
    write_json(output/'protocol.json', protocol)
    files = sorted(local_path('backend/radar_platform').rglob('*.py'))+[local_path(__file__)]
    hashes = {str(f.relative_to(local_path('.'))): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    # Only hash physical generation dependencies and the new pipeline; old model changes do not matter.
    excluded = {'model', 'model_v2', 'model_v3', 'learning', 'learning_v2', 'inference', 'controller', 'state_filter', 'episode'}
    hashes = {k: v for k, v in hashes.items() if not ('/navigation/' in k and k.rsplit('/', 1)[-1][:-3] in excluded)}
    if (output/'source_sha256.json').exists() and read_json(output/'source_sha256.json') != hashes:
        raise ValueError('generation source changed; use a versioned dataset')
    write_json(output/'source_sha256.json', hashes)
    selected = [r for r in records if r['split'] in args.splits]
    started = time.monotonic(); results = []; sampling_hashes = set()
    write_json(output/'status.json', dict(status='generating', pid=os.getpid(), splits=args.splits, total=len(selected)))
    try:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as executor:
            jobs = [(str(geography), r, str(output/'cases'/r['id'])) for r in selected]
            for future in as_completed([executor.submit(generate_multidem_case, job) for job in jobs]):
                result = future.result(); results.append(result); sampling_hashes.add(result['sampling_sha256'])
                if len(sampling_hashes) != 1:
                    raise ValueError('sampling changed between scenes; tensor-only model requires fixed sampling')
                print(json.dumps(dict(done=len(results), total=len(selected), **result)), flush=True)
                write_json(output/'status.json', dict(status='generating', pid=os.getpid(), done=len(results), total=len(selected),
                           elapsed_s=time.monotonic()-started, splits=args.splits))
        write_json(output/'sampling.json', read_json(output/'cases'/selected[0]['id']/'sampling.json'))
        write_json(output/'status.json', dict(status='complete', cases=len(results), splits=args.splits,
                   counts={s: sum(r['split'] == s for r in selected) for s in args.splits},
                   elapsed_s=time.monotonic()-started, test_generated='test' in args.splits))
    except BaseException as exc:
        write_json(output/'status.json', dict(status='failed', error=str(exc), completed=len(results), splits=args.splits))
        raise


if __name__ == '__main__':
    main()
