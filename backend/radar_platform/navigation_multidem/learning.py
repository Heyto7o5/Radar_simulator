"""Offline supervision is kept separate from the sole model argument."""
import numpy as np
import torch
from torch.utils.data import Dataset

from ..navigation.io import local_path, read_json
from .protocol import validate_records


class GeographicRadarDataset(Dataset):
    def __init__(self, root, split):
        self.root = local_path(root)
        protocol = read_json(self.root/'protocol.json')
        validate_records(protocol['records'], protocol['plan'])
        self.records = [r for r in protocol['records'] if r['split'] == split]
        if not self.records:
            raise ValueError('requested split has no records')
        labels, valid = [], []
        for row in self.records:
            folder = self.root/'cases'/row['id']
            if not (folder/'rda_features.npy').exists():
                raise ValueError(f'missing generated RDA input: {row["id"]}')
            truth = read_json(folder/'truth.json')
            labels.append(truth['state_body'])
            valid.append(truth['valid_landmark_observation'])
        self.labels = np.asarray(labels, np.float32)
        self.valid = np.asarray(valid, np.float32)
        if not np.isfinite(self.labels).all():
            raise ValueError('nonfinite training labels')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        features = np.load(self.root/'cases'/row['id']/'rda_features.npy', allow_pickle=False).astype(np.float32)
        # Region, geometry, labels and command histories are never appended to x.
        return torch.from_numpy(features), torch.from_numpy(self.labels[index]), torch.tensor(self.valid[index])


@torch.inference_mode()
def evaluate(model, dataset, loader, device):
    model.eval()
    predictions, scales, probabilities = [], [], []
    for features, _, _ in loader:
        state, scale, logit = model(features.to(device))
        predictions.append(state.cpu().numpy()); scales.append(scale.cpu().numpy())
        probabilities.append(logit.sigmoid().cpu().numpy())
    prediction, scale, probability = map(np.concatenate, (predictions, scales, probabilities))
    position = np.linalg.norm(prediction[:, :3]-dataset.labels[:, :3], axis=1)
    velocity = np.linalg.norm(prediction[:, 3:]-dataset.labels[:, 3:], axis=1)

    def summary(mask):
        n = int(np.count_nonzero(mask))
        if not n:
            return dict(count=0, position_mean_m=None, position_p90_m=None, velocity_mean_mps=None)
        return dict(count=n, position_mean_m=float(position[mask].mean()),
                    position_p90_m=float(np.percentile(position[mask], 90)), velocity_mean_mps=float(velocity[mask].mean()))

    regions = np.asarray([r['region_id'] for r in dataset.records])
    result = dict(all_states=summary(np.ones(len(dataset), bool)), visible_states=summary(dataset.valid > 0),
                  by_region={r: summary(regions == r) for r in sorted(set(regions))},
                  visibility_accuracy=float(np.mean((probability >= .5) == (dataset.valid > 0))))
    return result, dict(prediction=prediction, predicted_scale=scale, visibility_probability=probability,
                        truth=dataset.labels, visible=dataset.valid, region_id=regions)
