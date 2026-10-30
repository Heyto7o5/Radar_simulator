#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wasserstein_distance

from radar_platform.fmcw import FMCWConfig, FMCWPointTarget, process_fmcw_rd, simulate_fmcw_if
from radar_platform.tsms_validation import (
    compare_rd,
    load_processed_rd,
    load_raw_sample,
    normalized_power_db,
    processed_axes,
    rd_metrics,
)


def mean(items: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([float(item[key]) for item in items]))


def fmcw_config_from_sample(sample: Any) -> FMCWConfig:
    source = sample.config
    return FMCWConfig(
        start_hz=source.start_hz,
        stop_hz=source.stop_hz,
        ramp_s=source.ramp_s,
        chirp_period_s=source.chirp_period_s,
        samples=source.samples,
        chirps=source.chirps,
        sample_rate_hz=source.sample_rate_hz,
    )


def range_crop_start(config: FMCWConfig, distance_m: float, fft_size: int) -> int:
    if distance_m <= 10.0:
        start_range = 0.0
    elif distance_m <= 20.0:
        start_range = 10.0
    else:
        start_range = 20.0
    bin_spacing = (
        299_792_458.0
        * config.sample_rate_hz
        / (2.0 * config.slope_hz_s * fft_size)
    )
    return int(round(start_range / bin_spacing))


def process_platform_if(
    cube: np.ndarray,
    config: FMCWConfig,
    distance_m: float,
    range_bins: int,
    range_window: np.ndarray,
    doppler_window: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    result = process_fmcw_rd(
        cube,
        config,
        range_fft_size=4096,
        doppler_fft_size=4096,
        range_window=range_window,
        doppler_window=doppler_window,
        discard_first_sample=True,
        range_bin_start=range_crop_start(config, distance_m, 4096),
        range_bin_count=range_bins,
    )
    return result.rd, result.range_axis_m, result.velocity_axis_mps


def load_distance(
    directory: Path,
    distance_m: float,
    calibration_count: int,
) -> dict[str, Any]:
    paths = sorted(directory.rglob("*.mat"))
    if len(paths) <= calibration_count:
        raise ValueError(f"{directory} contains only {len(paths)} measurements")
    records: list[dict[str, Any]] = []
    for path in paths:
        rd = load_processed_rd(path)
        range_axis, velocity_axis = processed_axes(distance_m, rd.shape)
        metrics = rd_metrics(
            rd,
            range_axis,
            velocity_axis,
            0.6,
            0.19,
            expected_range_m=distance_m,
            expected_velocity_mps=0.0,
        )
        records.append({"path": path, "shape": rd.shape, "metrics": metrics})
    calibration = records[:calibration_count]
    test = records[calibration_count:]
    calibration_metrics = [item["metrics"] for item in calibration]
    return {
        "distance_m": distance_m,
        "calibration": calibration,
        "test": test,
        "calibration_model": {
            "range_bias_m": mean(calibration_metrics, "peak_range_m") - distance_m,
            "velocity_bias_mps": mean(calibration_metrics, "peak_velocity_mps"),
            "noise_floor_db": mean(calibration_metrics, "noise_floor_median_db"),
            "range_width_3db_m": mean(calibration_metrics, "range_width_3db_m"),
            "velocity_width_3db_mps": mean(calibration_metrics, "velocity_width_3db_mps"),
        },
    }


def choose_windows(
    config: FMCWConfig,
    sample: Any,
    datasets: list[dict[str, Any]],
) -> tuple[str, np.ndarray, str, np.ndarray, list[dict[str, float | str]]]:
    range_candidates = {
        "rectangular": np.ones(config.samples - 1),
        "Hamming": np.hamming(config.samples - 1),
        "Hann": np.hanning(config.samples - 1),
        "Kaiser(beta=4)": np.kaiser(config.samples - 1, 4.0),
        "official-raw-sample": np.asarray(sample.range_window[:, 0], dtype=np.float64),
    }
    doppler_candidates = {
        "rectangular": np.ones(config.chirps),
        "Hamming": np.hamming(config.chirps),
        "Hann": np.hanning(config.chirps),
        "official-raw-sample": np.asarray(sample.doppler_window[0, :], dtype=np.float64),
    }
    target_range_width = float(
        np.mean([data["calibration_model"]["range_width_3db_m"] for data in datasets])
    )
    target_velocity_width = float(
        np.mean([data["calibration_model"]["velocity_width_3db_mps"] for data in datasets])
    )
    receiver_positions = np.arange(4, dtype=np.float64) * config.wavelength_m / 2.0
    cube = simulate_fmcw_if(
        config,
        [FMCWPointTarget(range_m=8.0)],
        receiver_positions,
        real_output=True,
    )
    candidates: list[dict[str, float | str]] = []
    best: tuple[float, str, np.ndarray, str, np.ndarray] | None = None
    for range_name, range_window in range_candidates.items():
        for doppler_name, doppler_window in doppler_candidates.items():
            rd_cube, _, _ = process_platform_if(
                cube,
                config,
                8.0,
                161,
                range_window,
                doppler_window,
            )
            range_axis, velocity_axis = processed_axes(8.0, rd_cube.shape[:2])
            metrics = rd_metrics(
                rd_cube[:, :, 0],
                range_axis,
                velocity_axis,
                0.6,
                0.19,
                expected_range_m=8.0,
                expected_velocity_mps=0.0,
            )
            range_error = abs(metrics["range_width_3db_m"] - target_range_width)
            velocity_error = abs(metrics["velocity_width_3db_mps"] - target_velocity_width)
            score = range_error / 0.6 + velocity_error / 0.19
            candidate = {
                "range_window": range_name,
                "doppler_window": doppler_name,
                "range_width_m": metrics["range_width_3db_m"],
                "velocity_width_mps": metrics["velocity_width_3db_mps"],
                "score": score,
            }
            candidates.append(candidate)
            if best is None or score < best[0]:
                best = (score, range_name, range_window, doppler_name, doppler_window)
    assert best is not None
    return best[1], best[2], best[3], best[4], candidates


def add_calibrated_noise(
    signal_rd: np.ndarray,
    noise_rd: np.ndarray,
    range_axis: np.ndarray,
    velocity_axis: np.ndarray,
    target_range_m: float,
    target_velocity_mps: float,
    noise_floor_db: float,
) -> tuple[np.ndarray, float]:
    range_grid = range_axis[:, None]
    velocity_grid = velocity_axis[None, :]
    background = ~(
        (np.abs(range_grid - target_range_m) <= 1.8)
        & (np.abs(velocity_grid - target_velocity_mps) <= 0.6)
    )
    signal_peak = max(float(np.max(np.abs(signal_rd))), 1.0e-30)
    unit_noise_median = max(float(np.median(np.abs(noise_rd)[background])), 1.0e-30)
    desired_ratio = 10.0 ** (noise_floor_db / 20.0)
    scale = desired_ratio * signal_peak / unit_noise_median
    return (signal_rd + scale * noise_rd).astype(np.complex64), float(scale)


def local_comparison(
    measured_rd: np.ndarray,
    simulated_rd: np.ndarray,
    range_axis: np.ndarray,
    velocity_axis: np.ndarray,
    expected_range_m: float,
) -> dict[str, float]:
    range_mask = np.abs(range_axis - expected_range_m) <= 1.5
    velocity_mask = np.abs(velocity_axis) <= 0.6
    measured_power = np.abs(measured_rd[np.ix_(range_mask, velocity_mask)]) ** 2
    simulated_power = np.abs(simulated_rd[np.ix_(range_mask, velocity_mask)]) ** 2
    measured_power /= max(float(np.sum(measured_power)), 1.0e-30)
    simulated_power /= max(float(np.sum(simulated_power)), 1.0e-30)
    measured_range = np.sum(measured_power, axis=1)
    simulated_range = np.sum(simulated_power, axis=1)
    measured_velocity = np.sum(measured_power, axis=0)
    simulated_velocity = np.sum(simulated_power, axis=0)
    measured_flat = measured_power.ravel()
    simulated_flat = simulated_power.ravel()
    if np.std(measured_flat) and np.std(simulated_flat):
        correlation = float(np.corrcoef(measured_flat, simulated_flat)[0, 1])
    else:
        correlation = 0.0
    return {
        "local_range_wasserstein_m": float(
            wasserstein_distance(
                range_axis[range_mask],
                range_axis[range_mask],
                u_weights=measured_range,
                v_weights=simulated_range,
            )
        ),
        "local_velocity_wasserstein_mps": float(
            wasserstein_distance(
                velocity_axis[velocity_mask],
                velocity_axis[velocity_mask],
                u_weights=measured_velocity,
                v_weights=simulated_velocity,
            )
        ),
        "local_power_correlation": correlation,
    }


def simulate_distance(
    config: FMCWConfig,
    dataset: dict[str, Any],
    range_window: np.ndarray,
    doppler_window: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    distance = dataset["distance_m"]
    calibration = dataset["calibration_model"]
    shape = dataset["test"][0]["shape"]
    target_range = distance + calibration["range_bias_m"]
    target_velocity = calibration["velocity_bias_mps"]
    receiver_positions = np.arange(4, dtype=np.float64) * config.wavelength_m / 2.0
    signal_if = simulate_fmcw_if(
        config,
        [
            FMCWPointTarget(
                range_m=target_range,
                radial_velocity_mps=target_velocity,
            )
        ],
        receiver_positions,
        real_output=True,
        seed=seed,
    )
    signal_cube, physical_range_axis, physical_velocity_axis = process_platform_if(
        signal_if,
        config,
        distance,
        shape[0],
        range_window,
        doppler_window,
    )
    rng = np.random.default_rng(seed + 10_000)
    unit_noise_if = rng.normal(0.0, 1.0, size=signal_if.shape).astype(np.float32)
    noise_cube, _, _ = process_platform_if(
        unit_noise_if,
        config,
        distance,
        shape[0],
        range_window,
        doppler_window,
    )
    range_axis, velocity_axis = processed_axes(distance, shape)
    channels = []
    noise_scales = []
    for channel in range(signal_cube.shape[2]):
        rd, scale = add_calibrated_noise(
            signal_cube[:, :, channel],
            noise_cube[:, :, channel],
            range_axis,
            velocity_axis,
            target_range,
            target_velocity,
            calibration["noise_floor_db"],
        )
        channels.append(rd)
        noise_scales.append(scale)
    channel_cube = np.stack(channels, axis=2)
    # The official TSMS reconstruction selects one receiver channel.  Keep
    # all four simulated channels, but use channel 0 for like-for-like scores.
    return {
        "if_cube": signal_if,
        "rd_channels": channel_cube,
        "rd": channel_cube[:, :, 0],
        "range_axis": range_axis,
        "velocity_axis": velocity_axis,
        "physical_range_axis": physical_range_axis,
        "physical_velocity_axis": physical_velocity_axis,
        "noise_scales": noise_scales,
        "target_range_m": target_range,
        "target_velocity_mps": target_velocity,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the platform FMCW signal simulator against held-out TSMS RD"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/nas/datasets/hyt/datasets/radar_open/tsms_drone_sample/extracted"),
    )
    parser.add_argument(
        "--raw-sample",
        type=Path,
        default=Path("/nas/datasets/hyt/datasets/radar_open/tsms_drone_sample/raw/FMCW_RawData.mat"),
    )
    parser.add_argument("--calibration-count", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    distance_dirs: list[tuple[float, Path]] = []
    for directory in args.root.glob("corner_reflector_*m"):
        try:
            distance = float(directory.name.removeprefix("corner_reflector_").removesuffix("m"))
        except ValueError:
            continue
        distance_dirs.append((distance, directory))
    distance_dirs.sort()
    if len(distance_dirs) < 2:
        raise SystemExit("at least two TSMS distances are required")

    raw_sample = load_raw_sample(args.raw_sample)
    config = fmcw_config_from_sample(raw_sample)
    datasets = [
        load_distance(directory, distance, args.calibration_count)
        for distance, directory in distance_dirs
    ]
    range_window_name, range_window, doppler_window_name, doppler_window, window_candidates = choose_windows(
        config,
        raw_sample,
        datasets,
    )

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (
        Path("/nas/datasets/hyt/datasets/radar_open/tsms_drone_sample/processed")
        / f"platform_fmcw_{timestamp}"
    )
    output.mkdir(parents=True, exist_ok=False)

    rows: list[dict[str, Any]] = []
    simulations: dict[float, dict[str, Any]] = {}
    representatives: dict[float, np.ndarray] = {}
    for dataset_index, dataset in enumerate(datasets):
        distance = dataset["distance_m"]
        simulation = simulate_distance(
            config,
            dataset,
            range_window,
            doppler_window,
            seed=42 + dataset_index,
        )
        simulations[distance] = simulation
        simulated_metrics = rd_metrics(
            simulation["rd"],
            simulation["range_axis"],
            simulation["velocity_axis"],
            0.6,
            0.19,
            expected_range_m=distance,
            expected_velocity_mps=0.0,
        )
        measured_metrics: list[dict[str, Any]] = []
        comparisons: list[dict[str, float]] = []
        local_scores: list[dict[str, float]] = []
        for record_index, record in enumerate(dataset["test"]):
            measured_rd = load_processed_rd(record["path"])
            if record_index == 0:
                representatives[distance] = measured_rd
            measured_metrics.append(record["metrics"])
            comparisons.append(
                compare_rd(
                    record["metrics"],
                    simulated_metrics,
                    simulation["range_axis"],
                    simulation["velocity_axis"],
                )
            )
            local_scores.append(
                local_comparison(
                    measured_rd,
                    simulation["rd"],
                    simulation["range_axis"],
                    simulation["velocity_axis"],
                    distance,
                )
            )
        measured_peak_mean = mean(measured_metrics, "peak_range_m")
        measured_velocity_mean = mean(measured_metrics, "peak_velocity_mps")
        row = {
            "distance_m": distance,
            "n_calibration": len(dataset["calibration"]),
            "n_test": len(dataset["test"]),
            "calibration_range_bias_m": dataset["calibration_model"]["range_bias_m"],
            "calibration_velocity_bias_mps": dataset["calibration_model"]["velocity_bias_mps"],
            "measured_peak_range_mean_m": measured_peak_mean,
            "platform_peak_range_m": simulated_metrics["peak_range_m"],
            "platform_to_measured_peak_error_m": abs(
                simulated_metrics["peak_range_m"] - measured_peak_mean
            ),
            "measured_peak_velocity_mean_mps": measured_velocity_mean,
            "platform_peak_velocity_mps": simulated_metrics["peak_velocity_mps"],
            "platform_to_measured_velocity_error_mps": abs(
                simulated_metrics["peak_velocity_mps"] - measured_velocity_mean
            ),
            "measured_range_width_3db_m": mean(measured_metrics, "range_width_3db_m"),
            "platform_range_width_3db_m": simulated_metrics["range_width_3db_m"],
            "range_width_error_m": abs(
                simulated_metrics["range_width_3db_m"]
                - mean(measured_metrics, "range_width_3db_m")
            ),
            "measured_velocity_width_3db_mps": mean(
                measured_metrics, "velocity_width_3db_mps"
            ),
            "platform_velocity_width_3db_mps": simulated_metrics[
                "velocity_width_3db_mps"
            ],
            "velocity_width_error_mps": abs(
                simulated_metrics["velocity_width_3db_mps"]
                - mean(measured_metrics, "velocity_width_3db_mps")
            ),
            "range_pslr_error_db": mean(comparisons, "range_pslr_error_db"),
            "noise_floor_error_db": mean(comparisons, "noise_floor_error_db"),
            "full_range_wasserstein_m": mean(comparisons, "range_wasserstein_m"),
            "full_velocity_wasserstein_mps": mean(comparisons, "velocity_wasserstein_mps"),
            "local_range_wasserstein_m": mean(local_scores, "local_range_wasserstein_m"),
            "local_velocity_wasserstein_mps": mean(
                local_scores, "local_velocity_wasserstein_mps"
            ),
            "local_power_correlation": mean(local_scores, "local_power_correlation"),
            "range_axis_max_error_m": float(
                np.max(np.abs(simulation["physical_range_axis"] - simulation["range_axis"]))
            ),
            "velocity_axis_max_error_mps": float(
                np.max(
                    np.abs(
                        simulation["physical_velocity_axis"] - simulation["velocity_axis"]
                    )
                )
            ),
        }
        rows.append(row)

    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    representative_distance = 8.0 if 8.0 in simulations else rows[len(rows) // 2]["distance_m"]
    representative_simulation = simulations[representative_distance]
    representative_measured = representatives[representative_distance]
    np.save(output / "simulated_if_8m.npy", representative_simulation["if_cube"], allow_pickle=False)
    np.save(output / "simulated_rd_8m.npy", representative_simulation["rd_channels"], allow_pickle=False)
    np.save(output / "measured_rd_8m.npy", representative_measured, allow_pickle=False)

    measured_db = normalized_power_db(representative_measured, floor_db=-60.0)
    simulated_db = normalized_power_db(representative_simulation["rd"], floor_db=-60.0)
    difference_db = np.abs(measured_db - simulated_db)
    range_axis = representative_simulation["range_axis"]
    velocity_axis = representative_simulation["velocity_axis"]
    extent = [velocity_axis[0], velocity_axis[-1], range_axis[0], range_axis[-1]]
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    first = axes[0, 0].imshow(
        measured_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-45,
        vmax=0,
        cmap="turbo",
    )
    axes[0, 0].set_title("Held-out measured RD · 8 m")
    second = axes[0, 1].imshow(
        simulated_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=-45,
        vmax=0,
        cmap="turbo",
    )
    axes[0, 1].set_title("Platform FMCW simulation · 8 m")
    third = axes[0, 2].imshow(
        difference_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        vmin=0,
        vmax=45,
        cmap="magma",
    )
    axes[0, 2].set_title("Absolute normalized-dB difference")
    for axis in axes[0]:
        axis.set_xlabel("Radial velocity (m/s)")
        axis.set_ylabel("Range (m)")
    figure.colorbar(first, ax=axes[0, :2], shrink=0.82, label="Normalized power (dB)")
    figure.colorbar(third, ax=axes[0, 2], shrink=0.82, label="Absolute difference (dB)")

    measured_range_profile = np.max(measured_db, axis=1)
    simulated_range_profile = np.max(simulated_db, axis=1)
    axes[1, 0].plot(range_axis, measured_range_profile, label="Measured test", lw=1.5)
    axes[1, 0].plot(range_axis, simulated_range_profile, label="Platform simulation", lw=1.5)
    axes[1, 0].axvline(representative_distance, color="black", ls="--", lw=1, label="Truth")
    axes[1, 0].set_xlim(representative_distance - 2.0, representative_distance + 2.0)
    axes[1, 0].set_ylim(-45, 1)
    axes[1, 0].set_title("8 m range profile")
    axes[1, 0].set_xlabel("Range (m)")
    axes[1, 0].set_ylabel("Normalized power (dB)")
    axes[1, 0].legend(fontsize=8)

    distances = np.asarray([row["distance_m"] for row in rows])
    axes[1, 1].plot(
        distances,
        [abs(row["measured_peak_range_mean_m"] - row["distance_m"]) for row in rows],
        "o-",
        label="Measured-to-truth",
    )
    axes[1, 1].plot(
        distances,
        [row["platform_to_measured_peak_error_m"] for row in rows],
        "s-",
        label="Platform-to-measured",
    )
    axes[1, 1].set_title("Peak-range errors")
    axes[1, 1].set_xlabel("True range (m)")
    axes[1, 1].set_ylabel("Absolute error (m)")
    axes[1, 1].legend(fontsize=8)

    width = 0.32
    positions = np.arange(len(rows))
    axes[1, 2].bar(
        positions - width / 2,
        [row["local_range_wasserstein_m"] for row in rows],
        width,
        label="Target ROI",
    )
    axes[1, 2].bar(
        positions + width / 2,
        [row["full_range_wasserstein_m"] for row in rows],
        width,
        label="Full range profile",
    )
    axes[1, 2].set_xticks(positions, [f"{distance:g}m" for distance in distances])
    axes[1, 2].set_title("Measured vs platform Range W₁")
    axes[1, 2].set_ylabel("Wasserstein distance (m)")
    axes[1, 2].legend(fontsize=8)
    figure.suptitle(
        "TSMS held-out validation of the independent platform FMCW simulator",
        fontsize=16,
    )
    figure.savefig(output / "platform_fmcw_validation.png", dpi=180)
    figure.savefig(output / "platform_fmcw_validation.pdf")
    plt.close(figure)

    summary = {
        "platform_to_measured_peak_mae_m": float(
            np.mean([row["platform_to_measured_peak_error_m"] for row in rows])
        ),
        "platform_to_measured_velocity_mae_mps": float(
            np.mean([row["platform_to_measured_velocity_error_mps"] for row in rows])
        ),
        "range_width_mae_m": float(np.mean([row["range_width_error_m"] for row in rows])),
        "velocity_width_mae_mps": float(
            np.mean([row["velocity_width_error_mps"] for row in rows])
        ),
        "local_range_wasserstein_m": float(
            np.mean([row["local_range_wasserstein_m"] for row in rows])
        ),
        "full_range_wasserstein_m": float(
            np.mean([row["full_range_wasserstein_m"] for row in rows])
        ),
        "local_power_correlation": float(
            np.mean([row["local_power_correlation"] for row in rows])
        ),
        "range_pslr_error_db": float(
            np.mean([row["range_pslr_error_db"] for row in rows])
        ),
        "noise_floor_error_db": float(
            np.mean([row["noise_floor_error_db"] for row in rows])
        ),
    }
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "method": "independent FMCW IF simulation -> Range FFT -> Doppler FFT",
        "calibration_count_per_distance": args.calibration_count,
        "test_count_per_distance": len(datasets[0]["test"]),
        "fmcw_config": config.__dict__,
        "selected_windows": {
            "range": range_window_name,
            "doppler": doppler_window_name,
            "selection_scope": "calibration widths only",
        },
        "window_candidates": window_candidates,
        "summary": summary,
        "rows": rows,
        "boundaries": [
            "The platform RD is generated from independent FMCW IF samples, not a measured RD template.",
            "Calibration uses only the first 30 repetitions at each distance.",
            "Per-distance bias and noise-floor calibration are not an absolute power/RCS model.",
            "Stable leakage and multipath are intentionally absent from the point-target simulator and remain visible in the difference map.",
        ],
    }
    (output / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    table = [
        "| 距离 | 实测峰值 | 平台峰值 | 峰值差 | Range宽度差 | ROI W₁ | 全剖面 W₁ |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table.append(
            "| "
            f"{row['distance_m']:.0f} m | {row['measured_peak_range_mean_m']:.3f} m | "
            f"{row['platform_peak_range_m']:.3f} m | {row['platform_to_measured_peak_error_m']:.3f} m | "
            f"{row['range_width_error_m']:.3f} m | {row['local_range_wasserstein_m']:.3f} m | "
            f"{row['full_range_wasserstein_m']:.3f} m |"
        )
    report = f"""# TSMS：平台FMCW仿真—实测验证

本结果从目标距离/速度和24GHz波形参数独立生成FMCW中频数据，再执行Range/Doppler FFT。仿真RD未使用实测RD模板。每个距离前{args.calibration_count}次只用于估计距离/速度偏差、窗函数和背景量级，表格为后续独立测试。

{chr(10).join(table)}

## 汇总

- 平台峰值对实测峰值平均误差：{summary['platform_to_measured_peak_mae_m']:.3f} m
- 平台速度峰值对实测平均误差：{summary['platform_to_measured_velocity_mae_mps']:.3f} m/s
- Range −3dB宽度平均误差：{summary['range_width_mae_m']:.3f} m
- Velocity −3dB宽度平均误差：{summary['velocity_width_mae_mps']:.3f} m/s
- 目标ROI Range Wasserstein：{summary['local_range_wasserstein_m']:.3f} m
- 全距离剖面 Range Wasserstein：{summary['full_range_wasserstein_m']:.3f} m
- 目标ROI二维功率相关：{summary['local_power_correlation']:.3f}

## 结论边界

平台仿真已经能独立复现FMCW简单目标的距离、Doppler和主瓣量级。全剖面差异明显大于目标ROI差异，差值图中的零距离泄漏、固定条纹和多径尚未由简单点目标模型复现。当前不验证绝对RCS或完整接收机。
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
