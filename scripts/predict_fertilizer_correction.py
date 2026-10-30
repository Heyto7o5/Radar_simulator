#!/usr/bin/env python3
"""Export or use fixed-scene calibration; prediction reads no DEM or pose truth."""
import argparse
import json
from pathlib import Path
import numpy as np

from radar_platform.fertilizer_pilot import RDACalibration, ideal_position_adjustment


AXES = ('range_m', 'velocity_mps', 'azimuth_deg', 'frame_times_s')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    export = sub.add_parser('export')
    export.add_argument('--run', type=Path, required=True)
    export.add_argument('--condition', choices=['terrain', 'buildings'], default='buildings')
    export.add_argument('--output', type=Path, required=True)
    infer = sub.add_parser('predict')
    infer.add_argument('--model', type=Path, required=True)
    infer.add_argument('--rda', type=Path, required=True)
    infer.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError('existing output preserved')
    if args.command == 'export':
        protocol = json.loads((args.run/'protocol.json').read_text())
        with np.load(args.run/args.condition/'calibration.npz', allow_pickle=False) as model:
            values = {k: model[k] for k in ('features', 'states')}
        with np.load(args.run/args.condition/'calibration_000'/'rda.npz', allow_pickle=False) as sample:
            values.update({k: sample[k] for k in AXES})
            values['rda_shape'] = np.array(sample['magnitude'].shape)
        values['contract_json'] = np.array(json.dumps(dict(
            version=1, neighbors=protocol['neighbors'], altitude_m=protocol['altitude_m'],
            observation_s=protocol['observation_s'], release_delay_s=protocol['release_delay_s'],
            target_ne_m=[0, 0], condition=args.condition,
            scope='fixed calibrated scene, fixed altitude/attitude, no wind or drag; ideal position adjustment')))
        with args.output.open('xb') as handle:
            np.savez_compressed(handle, **values)
        print(json.dumps(dict(model=str(args.output))))
        return
    with np.load(args.model, allow_pickle=False) as artifact, np.load(args.rda, allow_pickle=False) as observation:
        for key in AXES:
            if not np.array_equal(artifact[key], observation[key]):
                raise ValueError(f'observation {key} differs from calibration')
        rda = observation['magnitude']
        if not np.array_equal(artifact['rda_shape'], rda.shape):
            raise ValueError('observation RDA shape differs from calibration')
        contract = json.loads(str(artifact['contract_json']))
        model = RDACalibration(artifact['features'], artifact['states'], contract['neighbors'])
        state, diagnostic = model.predict(rda)
    correction = ideal_position_adjustment(state, observation_s=contract['observation_s'],
                                          altitude_m=contract['altitude_m'], release_delay_s=contract['release_delay_s'])
    result = dict(estimated_start_position_ne_m=state[:2].tolist(),
                  estimated_velocity_ne_mps=state[2:].tolist(), correction_ne_m=correction.tolist(),
                  predicted_flat_target_plane_landing_ne_m=(-correction).tolist(),
                  diagnostic=diagnostic, scope=contract['scope'],
                  confidence='not calibrated; nearest distance is not a probability',
                  input='RDA magnitudes and radar axes only; scene relation encoded in frozen calibration')
    with args.output.open('x') as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
