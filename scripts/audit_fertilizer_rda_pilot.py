#!/usr/bin/env python3
"""Audit completed artifacts against the declared fixed-scene experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from radar_platform.unified.schema import Simulation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    root = args.run
    protocol = json.loads((root/'protocol.json').read_text())
    train, test = map(np.asarray, (protocol['calibration_states'], protocol['heldout_states']))
    assert not np.any(np.all(train[:, None] == test[None, :], axis=-1)), 'test overlaps calibration'
    checked, statuses, product_hashes = 0, {}, {}
    for condition in ['terrain', 'buildings']:
        for prefix, states in [('calibration', train), ('test', test)]:
            for k, state in enumerate(states):
                folder = root/condition/f'{prefix}_{k:03d}'
                cfg = Simulation.model_validate_json((folder/'request.json').read_text())
                np.testing.assert_allclose(cfg.platform.position_ned_m[:2], state[:2], atol=0, rtol=0)
                np.testing.assert_allclose(cfg.platform.velocity_ned_mps[:2], state[2:], atol=0, rtol=0)
                meta = json.loads((folder/'summary.json').read_text())
                digest = hashlib.sha256(cfg.model_dump_json(indent=2).encode()).hexdigest()
                assert digest == meta['config_sha256']
                with np.load(folder/'rda.npz', allow_pickle=False) as f:
                    x = f['magnitude']
                    assert x.shape == (2, len(f['range_m']), len(f['velocity_mps']), len(f['azimuth_deg']))
                    assert np.isfinite(x).all() and np.all(x >= 0) and np.all(x.sum(axis=(1, 2, 3)) > 0)
                    assert set(f.files) == {'magnitude', 'range_m', 'velocity_mps', 'azimuth_deg', 'frame_times_s'}
                product_hashes[str((folder/'rda.npz').relative_to(root))] = hashlib.sha256((folder/'rda.npz').read_bytes()).hexdigest()
                status = meta['applicability']['status']
                statuses[status] = statuses.get(status, 0)+1
                checked += 1
    results = json.loads((root/'results.json').read_text())
    assert len(results) == 2*len(test)
    assert len({(r['condition'],r['test']) for r in results}) == len(results)
    metrics = {}
    for name in ['terrain', 'buildings']:
        rows = [r for r in results if r['condition']==name]
        truth = np.array([r['truth_state'] for r in rows])
        pred = np.array([r['predicted_state'] for r in rows])
        np.testing.assert_array_equal(truth, test)
        metrics[name] = dict(position_rmse_m=float(np.sqrt(np.mean(np.sum((truth[:, :2]-pred[:, :2])**2,axis=1)))),
                             velocity_rmse_mps=float(np.sqrt(np.mean(np.sum((truth[:, 2:]-pred[:, 2:])**2,axis=1)))),
                             opposite_correction_count=int(sum(np.dot(r['correction_ne_m'], -np.array(r['before_ne_m'])) < 0 for r in rows)))
        for r in rows:
            assert abs(np.linalg.norm(r['after_ne_m'])-r['after_error_m']) < 1e-10
            assert r['oracle_error_m'] < 1e-8
    # Confirm nominal buildings genuinely contribute sampled echo paths.
    visibility = {}
    for k in range(2):
        folder = root/'buildings'/'calibration_040'/f'frame_{k:04d}'
        cov = json.loads((folder/'coverage.json').read_text())
        indices, counts = np.asarray(cov['object_index']), np.asarray(cov['sampled_echo_channel_slots'])
        visibility[str(k)] = {str(i):int(counts[indices==i].sum()) for i in np.unique(indices)}
        assert all(count > 0 for count in visibility[str(k)].values())
    audit = dict(checked_main_cases=checked, checked_main_frames=2*checked,
                 applicability_counts=statuses, test_calibration_disjoint=True,
                 rda_files_contain_no_pose_or_landmark_truth=True, oracle_landing_check=True,
                 metrics=metrics, nominal_sampled_echo_counts_by_object=visibility,
                 scope='artifact and ideal-model audit; not real radar or flight-control validation')
    (root/'audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    (root/'rda_sha256.json').write_text(json.dumps(product_hashes, indent=2)+'\n')
    print(json.dumps(audit))


if __name__ == '__main__':
    main()
