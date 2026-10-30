#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from radar_platform.tsms_validation import (
    compare_rd,
    load_processed_rd,
    processed_axes,
    rd_metrics,
    simulate_gaussian_point_rd,
)


def stats(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "p90": float(np.quantile(data, 0.9)),
    }


def fit_linear(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if np.unique(x).size < 2:
        return 0.0, float(np.median(y))
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def predict(model: tuple[float, float], x: float) -> float:
    return model[0] * x + model[1]


def load_distance(directory: Path, distance: float, calibration_count: int) -> dict:
    files = sorted(directory.rglob("*.mat"))
    if len(files) <= calibration_count:
        raise ValueError(f"{directory} has {len(files)} files; need > {calibration_count}")
    records = []
    calibration_amplitudes = []
    representative = None
    for file_index, path in enumerate(files):
        rd = load_processed_rd(path)
        range_axis, velocity_axis = processed_axes(distance, rd.shape)
        metrics = rd_metrics(
            rd, range_axis, velocity_axis, 0.6, 0.19,
            expected_range_m=distance,
            expected_velocity_mps=0.0,
        )
        records.append({"file": path.name, "metrics": metrics})
        if file_index < calibration_count:
            amplitude = np.abs(rd).astype(np.float32)
            calibration_amplitudes.append(amplitude / max(float(np.max(amplitude)), 1.0e-30))
        if representative is None:
            representative = (rd, range_axis, velocity_axis)
    return {
        "distance_m": distance,
        "calibration": records[:calibration_count],
        "test": records[calibration_count:],
        "representative": representative,
        "empirical_psf": np.median(np.stack(calibration_amplitudes, axis=0), axis=0).astype(np.complex64),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-range TSMS corner-reflector validation")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/nas/datasets/hyt/datasets/radar_open/tsms_drone_sample/extracted"),
    )
    parser.add_argument("--calibration-count", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    distance_dirs = []
    for directory in sorted(args.root.glob("corner_reflector_*m")):
        try:
            distance = float(directory.name.removeprefix("corner_reflector_").removesuffix("m"))
        except ValueError:
            continue
        distance_dirs.append((distance, directory))
    distance_dirs.sort(key=lambda item: item[0])
    if len(distance_dirs) < 2:
        raise SystemExit("At least two extracted corner-reflector distances are required")

    datasets = [
        load_distance(directory, distance, args.calibration_count)
        for distance, directory in distance_dirs
    ]

    calibration_x = []
    range_bias_y = []
    velocity_bias_y = []
    range_width_y = []
    velocity_width_y = []
    noise_floor_y = []
    peak_amplitude_y = []
    for dataset in datasets:
        distance = dataset["distance_m"]
        for record in dataset["calibration"]:
            metrics = record["metrics"]
            calibration_x.append(distance)
            range_bias_y.append(metrics["peak_range_m"] - distance)
            velocity_bias_y.append(metrics["peak_velocity_mps"])
            range_width_y.append(metrics["range_width_3db_m"])
            velocity_width_y.append(metrics["velocity_width_3db_mps"])
            noise_floor_y.append(metrics["noise_floor_median_db"])
            peak_amplitude_y.append(metrics["peak_amplitude_db_raw"])

    x = np.asarray(calibration_x)
    models = {
        "range_bias": fit_linear(x, np.asarray(range_bias_y)),
        "velocity_bias": fit_linear(x, np.asarray(velocity_bias_y)),
        "range_width": fit_linear(x, np.asarray(range_width_y)),
        "velocity_width": fit_linear(x, np.asarray(velocity_width_y)),
        "noise_floor": fit_linear(x, np.asarray(noise_floor_y)),
        "peak_amplitude_db": fit_linear(np.log10(x), np.asarray(peak_amplitude_y)),
    }

    rows = []
    representatives = {}
    for dataset in datasets:
        distance = dataset["distance_m"]
        ideal_comparisons = []
        calibrated_comparisons = []
        empirical_comparisons = []
        measured_metrics = []
        for index, record in enumerate(dataset["test"]):
            measured = record["metrics"]
            rd = load_processed_rd(
                next(path for path in distance_dirs if path[0] == distance)[1] / "Raw File" / record["file"]
            )
            range_axis, velocity_axis = processed_axes(distance, rd.shape)
            ideal_rd = simulate_gaussian_point_rd(
                range_axis, velocity_axis, distance, 0.0,
                0.6, 0.19, predict(models["noise_floor"], distance),
                seed=1000 + index,
            )
            calibrated_rd = simulate_gaussian_point_rd(
                range_axis,
                velocity_axis,
                distance + predict(models["range_bias"], distance),
                predict(models["velocity_bias"], distance),
                max(0.05, predict(models["range_width"], distance)),
                max(0.01, predict(models["velocity_width"], distance)),
                predict(models["noise_floor"], distance),
                seed=2000 + index,
            )
            ideal = rd_metrics(
                ideal_rd, range_axis, velocity_axis, 0.6, 0.19,
                expected_range_m=distance, expected_velocity_mps=0.0,
            )
            calibrated = rd_metrics(
                calibrated_rd, range_axis, velocity_axis, 0.6, 0.19,
                expected_range_m=distance, expected_velocity_mps=0.0,
            )
            empirical_rd = dataset["empirical_psf"]
            empirical = rd_metrics(
                empirical_rd, range_axis, velocity_axis, 0.6, 0.19,
                expected_range_m=distance, expected_velocity_mps=0.0,
            )
            ideal_comparisons.append(compare_rd(measured, ideal, range_axis, velocity_axis))
            calibrated_comparisons.append(compare_rd(measured, calibrated, range_axis, velocity_axis))
            empirical_comparisons.append(compare_rd(measured, empirical, range_axis, velocity_axis))
            measured_metrics.append(measured)
            if distance not in representatives:
                representatives[distance] = {
                    "measured": rd,
                    "calibrated": calibrated_rd,
                    "empirical": empirical_rd,
                    "range_axis": range_axis,
                    "velocity_axis": velocity_axis,
                }

        def mean_metric(items, key):
            return float(np.mean([item[key] for item in items]))

        rows.append({
            "distance_m": distance,
            "n_test": len(measured_metrics),
            "measured_peak_range_mean_m": mean_metric(measured_metrics, "peak_range_m"),
            "measured_peak_range_std_m": float(np.std([m["peak_range_m"] for m in measured_metrics])),
            "measured_range_mae_m": float(np.mean([abs(m["peak_range_m"] - distance) for m in measured_metrics])),
            "measured_velocity_mae_mps": float(np.mean([abs(m["peak_velocity_mps"]) for m in measured_metrics])),
            "measured_snr_db": mean_metric(measured_metrics, "peak_to_noise_db"),
            "measured_target_to_global_db": mean_metric(measured_metrics, "target_to_global_db"),
            "measured_peak_amplitude_db": mean_metric(measured_metrics, "peak_amplitude_db_raw"),
            "measured_range_width_3db_m": mean_metric(measured_metrics, "range_width_3db_m"),
            "measured_velocity_width_3db_mps": mean_metric(measured_metrics, "velocity_width_3db_mps"),
            "measured_range_pslr_db": mean_metric(measured_metrics, "range_pslr_db"),
            "ideal_peak_range_error_m": mean_metric(ideal_comparisons, "peak_range_error_m"),
            "calibrated_peak_range_error_m": mean_metric(calibrated_comparisons, "peak_range_error_m"),
            "ideal_range_wasserstein_m": mean_metric(ideal_comparisons, "range_wasserstein_m"),
            "calibrated_range_wasserstein_m": mean_metric(calibrated_comparisons, "range_wasserstein_m"),
            "empirical_range_wasserstein_m": mean_metric(empirical_comparisons, "range_wasserstein_m"),
            "ideal_range_pslr_error_db": mean_metric(ideal_comparisons, "range_pslr_error_db"),
            "calibrated_range_pslr_error_db": mean_metric(calibrated_comparisons, "range_pslr_error_db"),
            "empirical_range_pslr_error_db": mean_metric(empirical_comparisons, "range_pslr_error_db"),
        })

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("/nas/datasets/hyt/datasets/radar_open/tsms_drone_sample/processed") / f"multirange_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "https://doi.org/10.25452/figshare.plus.30027313.v1",
        "calibration_count_per_distance": args.calibration_count,
        "models": {key: {"slope": value[0], "intercept": value[1]} for key, value in models.items()},
        "rows": rows,
    }
    (output / "comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    distances = np.asarray([row["distance_m"] for row in rows])
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
    middle_distance = float(distances[len(distances) // 2])
    representative = representatives[middle_distance]
    measured_db = 20 * np.log10(np.abs(representative["measured"]) / np.max(np.abs(representative["measured"])) + 1e-12)
    simulated_db = 20 * np.log10(np.abs(representative["empirical"]) / np.max(np.abs(representative["empirical"])) + 1e-12)
    extent = [representative["velocity_axis"][0], representative["velocity_axis"][-1], representative["range_axis"][0], representative["range_axis"][-1]]
    axes[0, 0].imshow(measured_db, origin="lower", aspect="auto", extent=extent, vmin=-45, vmax=0, cmap="turbo")
    axes[0, 0].set_title(f"Measured RD · {middle_distance:g} m")
    axes[0, 1].imshow(simulated_db, origin="lower", aspect="auto", extent=extent, vmin=-45, vmax=0, cmap="turbo")
    axes[0, 1].set_title(f"Calibration-only empirical PSF · {middle_distance:g} m")
    for axis in axes[0, :2]:
        axis.set_xlabel("Radial velocity (m/s)"); axis.set_ylabel("Range (m)")

    axes[0, 2].errorbar(
        distances,
        [row["measured_peak_range_mean_m"] for row in rows],
        yerr=[row["measured_peak_range_std_m"] for row in rows],
        fmt="o", label="Measured peak",
    )
    axes[0, 2].plot(distances, distances, "k--", label="Truth")
    axes[0, 2].set_title("Range accuracy"); axes[0, 2].set_xlabel("True range (m)"); axes[0, 2].set_ylabel("Detected range (m)"); axes[0, 2].legend()

    axes[1, 0].plot(distances, [row["measured_snr_db"] for row in rows], "o-")
    axes[1, 0].set_title("Measured peak-to-noise"); axes[1, 0].set_xlabel("Range (m)"); axes[1, 0].set_ylabel("SNR-like ratio (dB)")

    axes[1, 1].plot(distances, [row["measured_range_width_3db_m"] for row in rows], "o-", label="Range width")
    axes[1, 1].plot(distances, [row["measured_velocity_width_3db_mps"] for row in rows], "s-", label="Velocity width")
    axes[1, 1].set_title("Measured -3 dB widths"); axes[1, 1].set_xlabel("Range (m)"); axes[1, 1].legend()

    width = 0.25
    positions = np.arange(len(rows))
    axes[1, 2].bar(positions - width, [row["ideal_range_wasserstein_m"] for row in rows], width, label="Ideal")
    axes[1, 2].bar(positions, [row["calibrated_range_wasserstein_m"] for row in rows], width, label="Gaussian PSF")
    axes[1, 2].bar(positions + width, [row["empirical_range_wasserstein_m"] for row in rows], width, label="Empirical PSF")
    axes[1, 2].set_xticks(positions, [f"{distance:g}m" for distance in distances])
    axes[1, 2].set_title("Range-profile Wasserstein"); axes[1, 2].set_ylabel("Distance (m)"); axes[1, 2].legend()
    figure.suptitle("TSMS Corner Reflector: Multi-range RD Validation", fontsize=16)
    figure.savefig(output / "multirange_validation.png", dpi=180)
    figure.savefig(output / "multirange_validation.pdf")
    plt.close(figure)

    table_lines = [
        "| Distance | Range MAE | Velocity MAE | Target/global | SNR | Ideal W₁ | Gaussian W₁ | Empirical W₁ |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            f"| {row['distance_m']:.0f} m | {row['measured_range_mae_m']:.3f} m | {row['measured_velocity_mae_mps']:.3f} m/s | {row['measured_target_to_global_db']:.2f} dB | {row['measured_snr_db']:.2f} dB | {row['ideal_range_wasserstein_m']:.3f} m | {row['calibrated_range_wasserstein_m']:.3f} m | {row['empirical_range_wasserstein_m']:.3f} m |"
        )
    report = "# TSMS角反射器多距离RD验证\n\n" + "\n".join(table_lines) + f"\n\n综合图：`multirange_validation.png`。校准模型仅使用每个距离前{args.calibration_count}次测量，表格使用其余独立测量。\n"
    (output / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
