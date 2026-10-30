#!/usr/bin/env python3
"""Separate background-floor agreement from annotated strong-echo agreement.

This is a post-hoc diagnostic, not a new held-out model selection experiment.
All model parameters and noise seeds are reused unchanged from the earlier run.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

from radar_platform.alignment_metrics import add_calibrated_noise, apply_channel_phase_response
from radar_platform.array_fmcw import process_array_adc
from radar_platform.comparison import compare_power
from radar_platform.ramp_alignment import aligned_array, aligned_config, read_ramp_adc, simulate_track_adc
from radar_platform.repeat_metrics import distribution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-run', type=Path, required=True)
    parser.add_argument('--noise-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path('/nas/datasets/hyt/datasets/radar_open/ramp_adc_77ghz/alignment_subset_20261005'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = json.loads((args.base_run / 'simulation_inputs.json').read_text())
    power = json.loads((args.base_run / 'power_alignment.json').read_text())
    east = json.loads((args.base_run / 'registration.json').read_text())['east_sign']
    frozen = json.loads((args.noise_run / 'frozen_calibration.json').read_text())
    variance = np.array(frozen['noise_variance_ADC_units_tx_major'])
    phase = np.array(frozen['phase_rad'])
    gain = frozen['signal_power_gain_kept_fixed']
    config, array = aligned_config(), aligned_array(east)
    window_gain = np.sum(np.hamming(128)**2) / np.sum(np.hamming(128))**2
    window_gain *= np.sum(np.hamming(255)**2) / np.sum(np.hamming(255))**2
    expected_noise_rd = float(variance.mean() * window_gain)
    thresholds = [6., 10., 20.]
    protocol = {
        'post_hoc_diagnostic': True, 'no_parameter_refit': True,
        'expected_mean_noise_RD_power': expected_noise_rd,
        'strong_measured_return_thresholds_above_noise_db': thresholds,
        'primary_display_threshold_db': 10.,
        'mask': 'same measured-power mask for all models, inside annotation range ROI; all Doppler',
        'not_CFAR_or_detection_ground_truth': True,
        'seed': 'same noise realization for noise-only and signal+noise; seed=42+evaluation index',
        'normalization': 'same frozen calibration reference; -60 dB lower floor, no upper clipping',
        'noise_calibration_sha256': hashlib.sha256((args.noise_run / 'frozen_calibration.json').read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (args.output / 'frozen_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')

    def process(adc):
        return process_array_adc(adc, config, array, azimuth_deg=[0.],
            closing_doppler_sign=-1, spatial_phase_sign=-1,
            range_window='hamming', doppler_window='hamming', spatial_window='hamming')

    old_rows = {r['frame_id']: r for r in csv.DictReader((args.noise_run / 'frame_metrics.csv').open(encoding='utf-8-sig'))}
    rows = []
    for i, record in enumerate(records):
        measured = process(read_ramp_adc(args.root / record['frame_id']))
        real_power = np.mean(abs(measured.rd_channels).astype(float)**2, axis=(2, 3))
        signal = apply_channel_phase_response(simulate_track_adc(record['smoothed_labels'], east_sign=east), phase) * np.sqrt(gain)
        noise = add_calibrated_noise(np.zeros_like(signal), variance, 42 + i)
        valid = np.broadcast_to(((measured.range_m >= 1) & (measured.range_m <= 24))[:, None], real_power.shape)
        target_range = np.zeros(len(measured.range_m), dtype=bool)
        for label in record['raw_labels']:
            r = label['range_m']
            extent = .5 * (abs(label['x_m'] / r) * label['width_m'] + abs(label['y_m'] / r) * label['length_m'])
            target_range |= abs(measured.range_m - r) <= 1.5 + extent
        roi = valid & target_range[:, None]
        for model, adc in [('signal_only', signal), ('noise_only', noise), ('signal_plus_noise', signal + noise)]:
            prediction = process(adc)
            predicted_power = np.mean(abs(prediction.rd_channels).astype(float)**2, axis=(2, 3))
            comparison = compare_power(real_power, predicted_power, reference_power=power['reference_RD_power'], valid_mask=valid, target_mask=roi)
            if model == 'signal_plus_noise':
                if abs(comparison.metrics['mae_db'] - float(old_rows[record['frame_id']]['rd_mae_db'])) > 1e-5:
                    raise AssertionError('control does not reproduce frozen signal+noise experiment')
            row = {'frame_id': record['frame_id'], 'role': record['role'], 'model': model,
                   'rd_mae_db': comparison.metrics['mae_db'], 'roi_mae_db': comparison.metrics['target_mae_db']}
            for threshold in thresholds:
                mask = roi & (real_power >= expected_noise_rd * 10**(threshold / 10))
                key = f'strong_{threshold:g}db'
                row[key + '_cells'] = int(mask.sum())
                row[key + '_mae_db'] = float(np.mean(abs(comparison.difference_db[mask]))) if mask.any() else None
            rows.append(row)
        if i % 50 == 0:
            print('same-noise controls', i, '/', len(records), flush=True)
    with (args.output / 'frame_metrics.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for role in ['development', 'heldout_record']:
        for model in ['signal_only', 'noise_only', 'signal_plus_noise']:
            group = [r for r in rows if r['role'] == role and r['model'] == model]
            entry = {'role': role, 'model': model, 'frames': len(group)}
            for metric in ['rd_mae_db', 'roi_mae_db', *[f'strong_{t:g}db_mae_db' for t in thresholds]]:
                values = [r[metric] for r in group if r[metric] is not None]
                entry[metric] = distribution(values) if values else None
            entry['strong_10db_cells'] = sum(r['strong_10db_cells'] for r in group)
            summaries.append(entry)
    (args.output / 'summary.json').write_text(json.dumps({'protocol': protocol, 'results': summaries}, indent=2) + '\n')
    font = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font_manager.fontManager.addfont(font)
    plt.rcParams.update({'font.family': font_manager.FontProperties(fname=font).get_name(), 'axes.unicode_minus': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, role, title in zip(axes, ['development', 'heldout_record'], ['开发记录：120 帧', '跨记录评估：180 帧']):
        group = [r for r in summaries if r['role'] == role]
        x = np.arange(3)
        for shift, metric, name in [(-.2, 'rd_mae_db', '全图 MAE'), (.2, 'strong_10db_mae_db', '标注区强回波 MAE')]:
            values = [r[metric]['mean'] for r in group]
            bars = axis.bar(x + shift, values, .38, label=name)
            axis.bar_label(bars, fmt='%.2f', fontsize=9)
        axis.set_xticks(x, ['只有目标', '只有噪声', '目标 + 噪声'])
        axis.set_ylabel('平均绝对差 / dB（越小越好）')
        axis.set_title(title)
        axis.set_ylim(0, max(r['strong_10db_mae_db']['mean'] for r in group) * 1.28)
        axis.legend(fontsize=8)
    fig.suptitle('RAMP：总分降低不等于汽车散射已复现\n强回波 = 实测功率高于校准噪声均值 10 dB；不是独立目标检测真值', fontsize=12)
    fig.savefig(args.output / 'noise_control.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
