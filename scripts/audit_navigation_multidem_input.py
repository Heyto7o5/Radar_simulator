#!/usr/bin/env python3
"""Replay one new-model prediction while denying all file access in predict()."""
import argparse
import inspect
import sys

import numpy as np
import torch

from radar_platform.navigation.io import local_path, write_json
from radar_platform.navigation_multidem.model import MultiDEMRadarNet, RDAEstimator, ARCHITECTURE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    checkpoint = torch.load(local_path(args.model)/'model.pt', map_location='cpu', weights_only=True)
    if checkpoint['architecture'] != ARCHITECTURE:
        raise ValueError('unexpected model architecture')
    model = MultiDEMRadarNet(); model.load_state_dict(checkpoint['state_dict']); model.eval()
    case = local_path(args.case)
    with np.load(case/'rda.npz', allow_pickle=False) as data:
        magnitude = data['magnitude'].copy()
    features = torch.from_numpy(np.load(case/'rda_features.npy', allow_pickle=False).astype(np.float32))[None]
    with torch.inference_mode():
        expected_state, expected_scale, expected_logit = model(features)
    predictor = RDAEstimator(model)
    guard = {'active': False, 'attempts': []}

    def audit(event, event_args):
        if guard['active'] and event in {'open', 'os.listdir', 'os.scandir'}:
            guard['attempts'].append(event)
            raise PermissionError('all file access forbidden during RDA-only prediction')

    sys.addaudithook(audit)
    guard['active'] = True
    try:
        prediction = predictor.predict(magnitude)
    finally:
        guard['active'] = False
    np.testing.assert_allclose(prediction['state_body'], expected_state[0].numpy(), atol=1e-5, rtol=1e-6)
    np.testing.assert_allclose(prediction['predicted_scale'], expected_scale[0].numpy(), atol=1e-5, rtol=1e-6)
    result = dict(passed=True, architecture=ARCHITECTURE,
        predict_arguments=list(inspect.signature(predictor.predict).parameters),
        model_buffers=list(dict(model.named_buffers())), forbidden_file_access_attempts=guard['attempts'],
        max_state_difference=float(np.max(np.abs(prediction['state_body']-expected_state[0].numpy()))),
        scope='Prediction uses only supplied RDA tensor and frozen learned weights/statistics; not an accuracy or cross-DEM success claim',
        smoke_checkpoint=checkpoint['smoke_only'])
    write_json(args.output, result)
    print(result)


if __name__ == '__main__':
    main()
