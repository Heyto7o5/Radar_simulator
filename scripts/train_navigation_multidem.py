#!/usr/bin/env python3
"""Train a fresh RDA-only CNN; validation geography selects the checkpoint."""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from radar_platform.navigation.io import local_path, read_json, write_json, digest
from radar_platform.navigation_multidem.learning import GeographicRadarDataset, evaluate
from radar_platform.navigation_multidem.model import MultiDEMRadarNet, ARCHITECTURE, FEATURE_SHAPE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    data, output = local_path(args.data), local_path(args.output)
    status = read_json(data/'status.json')
    if status['status'] != 'complete' or not {'train', 'val'} <= set(status['splits']):
        raise ValueError('complete cross-region train/val generation required')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads); torch.manual_seed(args.seed); np.random.seed(args.seed)
    train, val = GeographicRadarDataset(data, 'train'), GeographicRadarDataset(data, 'val')
    if not np.any(train.valid) or not np.any(val.valid):
        raise ValueError('no physically visible landmarks in train/validation')
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = MultiDEMRadarNet().to(args.device)
    model.target_mean.copy_(torch.from_numpy(train.labels[train.valid > 0].mean(axis=0)).to(args.device))
    model.target_std.copy_(torch.from_numpy(train.labels[train.valid > 0].std(axis=0).clip(.1)).to(args.device))
    dataset_protocol = read_json(data/'protocol.json')
    write_json(output/'training_protocol.json', dict(**vars(args), architecture=ARCHITECTURE,
        initialization='random; no old navigation or RD-Mamba checkpoint loaded',
        dataset_protocol_sha256=digest(dataset_protocol), input_shape=list(FEATURE_SHAPE),
        input='RDA-derived numeric tensor only', forbidden_inputs=['scene geometry', 'roof height', 'DEM', 'truth pose', 'commands', 'motion model'],
        dataset_counts=dataset_protocol['counts'], test_accessed=False,
        selection='macro-average of validation-region all-state position MAE plus 0.5 velocity MAE',
        uncertainty='predicted residual scale, uncalibrated; not an arrival guarantee',
        smoke_scope='pipeline wiring only, not navigation accuracy' if args.smoke_only else None))
    files = sorted(local_path('backend/radar_platform/navigation_multidem').glob('*.py'))+[local_path(__file__)]
    write_json(output/'source_sha256.json', {str(p.relative_to(local_path('.'))): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    write_json(output/'status.json', dict(status='training', pid=os.getpid(), test_accessed=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1e-5)
    history, best, best_epoch = [], float('inf'), 0
    started = time.monotonic()
    for epoch in range(1, args.epochs+1):
        model.train(); loss_sum = 0.
        for features, target, visible in train_loader:
            features, target, visible = features.to(args.device), target.to(args.device), visible.to(args.device)
            prediction, scale, logit = model(features)
            normalized_error = (prediction-target)/model.target_std
            normalized_scale = scale/model.target_std
            # Robust mean estimation plus a scale head trained on detached residuals.
            regression = F.smooth_l1_loss(normalized_error, torch.zeros_like(normalized_error), reduction='none').mean(1)
            scale_loss = (.5*(normalized_error.detach()/normalized_scale)**2+normalized_scale.log()).mean(1)
            loss = ((regression+.05*scale_loss)*visible).sum()/visible.sum().clamp_min(1)
            loss = loss+.1*F.binary_cross_entropy_with_logits(logit, visible)
            if not torch.isfinite(loss):
                raise ValueError('nonfinite loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5); optimizer.step()
            loss_sum += float(loss.detach())*len(features)
        scheduler.step()
        metrics, predictions = evaluate(model, val, val_loader, args.device)
        score = float(np.mean([m['position_mean_m']+.5*m['velocity_mean_mps'] for m in metrics['by_region'].values()]))
        row = dict(epoch=epoch, train_loss=loss_sum/len(train), validation=metrics, score=score, elapsed_s=time.monotonic()-started)
        history.append(row)
        if score < best:
            best, best_epoch = score, epoch
            torch.save(dict(architecture=ARCHITECTURE, state_dict=model.state_dict(), seed=args.seed,
                       epoch=epoch, validation_score=score, inference_input='RDA only', smoke_only=args.smoke_only), output/'model.tmp.pt')
            (output/'model.tmp.pt').replace(output/'model.pt')
            np.savez_compressed(output/'validation_predictions.npz', **predictions)
            write_json(output/'validation_metrics.json', metrics)
        write_json(output/'history.json', history)
        write_json(output/'status.json', dict(status='training', pid=os.getpid(), epoch=epoch, best_epoch=best_epoch, test_accessed=False))
        print(json.dumps(row), flush=True)
        if epoch-best_epoch >= args.patience:
            break
    write_json(output/'sampling.json', read_json(data/'sampling.json'))
    write_json(output/'status.json', dict(status='complete', best_epoch=best_epoch, validation_score=best,
        elapsed_s=time.monotonic()-started, test_accessed=False, smoke_only=args.smoke_only, closed_loop_evaluated=False))


if __name__ == '__main__':
    main()
