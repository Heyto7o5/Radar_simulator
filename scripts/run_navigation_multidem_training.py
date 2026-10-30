#!/usr/bin/env python3
"""Run only the new cross-DEM generation and training, without old finishers."""
import argparse
import hashlib
import os
import subprocess
import sys
import time
import zipfile

from radar_platform.navigation.io import local_path, read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--geography', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--train-per-region', type=int, default=512)
    parser.add_argument('--val-per-region', type=int, default=128)
    parser.add_argument('--test-per-region', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
    args = parser.parse_args()
    root = local_path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    files = sorted(local_path('backend/radar_platform').rglob('*.py'))
    files += [local_path(f'scripts/{name}') for name in ['generate_navigation_multidem.py', 'train_navigation_multidem.py', 'run_navigation_multidem_training.py']]
    hashes = {str(p.relative_to(local_path('.'))): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    write_json(root/'source_sha256.json', hashes)
    write_json(root/'run_protocol.json', dict(**vars(args), device='cpu', model_seed=11, test_evaluation_enabled=False,
        old_pipeline_used=False, acceptance='Training completion is not cross-DEM closed-loop success'))
    with zipfile.ZipFile(root/'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, str(path.relative_to(local_path('.'))))
    stages = [
        ('generation', ['scripts/generate_navigation_multidem.py', '--geography', args.geography,
            '--output', str(root/'data'), '--workers', str(args.workers),
            '--train-per-region', str(args.train_per_region), '--val-per-region', str(args.val_per_region),
            '--test-per-region', str(args.test_per_region), '--splits', 'train', 'val']),
        ('training', ['scripts/train_navigation_multidem.py', '--data', str(root/'data'),
            '--output', str(root/'model'), '--threads', str(args.threads), '--epochs', str(args.epochs), '--device', 'cpu'])]
    started = time.monotonic()
    try:
        for stage, command in stages:
            for name, expected in hashes.items():
                if hashlib.sha256(local_path(name).read_bytes()).hexdigest() != expected:
                    raise ValueError(f'source changed before {stage}: {name}; frozen pipeline stopped')
            write_json(root/'status.json', dict(status=stage, pid=os.getpid(), elapsed_s=time.monotonic()-started))
            with (root/f'{stage}.log').open('x') as log:
                subprocess.run([sys.executable, *command], cwd=local_path('.'), stdout=log, stderr=subprocess.STDOUT, check=True)
        write_json(root/'status.json', dict(status='training_complete', elapsed_s=time.monotonic()-started,
            model=read_json(root/'model/status.json'), test_evaluated=False, closed_loop_evaluated=False,
            next_step='Freeze model, then evaluate new independent test-region RDA and RDA-only closed loops; no old filter/roof prior.'))
    except BaseException as exc:
        write_json(root/'status.json', dict(status='failed', error=str(exc), elapsed_s=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
