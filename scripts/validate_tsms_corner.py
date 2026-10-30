#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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


def summarize(values: list[float]) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "p90": float(np.quantile(data, 0.9)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a simple point target against repeated TSMS corner-reflector RD measurements")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/nas/datasets/hyt/datasets/radar_open/tsms_drone_sample/extracted/corner_reflector_2m"),
    )
    parser.add_argument("--truth-range", type=float, default=2.0)
    parser.add_argument("--truth-velocity", type=float, default=0.0)
    parser.add_argument("--calibration-count", type=int, default=30)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    files = sorted(args.input_dir.rglob("*.mat"))
    if len(files) <= args.calibration_count:
        raise SystemExit("Need more extracted measurements than calibration-count")
    calibration_files = files[: args.calibration_count]
    test_files = files[args.calibration_count :]

    calibration_metrics = []
    for path in calibration_files:
        rd = load_processed_rd(path)
        range_axis, velocity_axis = processed_axes(args.truth_range, rd.shape)
        calibration_metrics.append(rd_metrics(rd, range_axis, velocity_axis, 0.6, 0.19))
    range_bias = float(np.median([item["peak_range_m"] - args.truth_range for item in calibration_metrics]))
    velocity_bias = float(np.median([item["peak_velocity_mps"] - args.truth_velocity for item in calibration_metrics]))
    range_width = float(np.median([item["range_width_3db_m"] for item in calibration_metrics]))
    velocity_width = float(np.median([item["velocity_width_3db_mps"] for item in calibration_metrics]))
    noise_floor = float(np.median([item["noise_floor_median_db"] for item in calibration_metrics]))

    test_records = []
    representative = None
    for index, path in enumerate(test_files):
        measured_rd = load_processed_rd(path)
        range_axis, velocity_axis = processed_axes(args.truth_range, measured_rd.shape)
        measured = rd_metrics(measured_rd, range_axis, velocity_axis, 0.6, 0.19)
        ideal_rd = simulate_gaussian_point_rd(
            range_axis,
            velocity_axis,
            args.truth_range,
            args.truth_velocity,
            0.6,
            0.19,
            noise_floor,
            seed=1000 + index,
        )
        calibrated_rd = simulate_gaussian_point_rd(
            range_axis,
            velocity_axis,
            args.truth_range + range_bias,
            args.truth_velocity + velocity_bias,
            range_width,
            velocity_width,
            noise_floor,
            seed=2000 + index,
        )
        ideal = rd_metrics(ideal_rd, range_axis, velocity_axis, 0.6, 0.19)
        calibrated = rd_metrics(calibrated_rd, range_axis, velocity_axis, 0.6, 0.19)
        record = {
            "file": path.name,
            "measured_peak_range_m": measured["peak_range_m"],
            "measured_peak_velocity_mps": measured["peak_velocity_mps"],
            "truth_range_error_m": abs(measured["peak_range_m"] - args.truth_range),
            "truth_velocity_error_mps": abs(measured["peak_velocity_mps"] - args.truth_velocity),
            "ideal": compare_rd(measured, ideal, range_axis, velocity_axis),
            "calibrated": compare_rd(measured, calibrated, range_axis, velocity_axis),
        }
        test_records.append(record)
        if representative is None:
            representative = {
                "measured_rd": measured_rd,
                "ideal_rd": ideal_rd,
                "calibrated_rd": calibrated_rd,
                "range_axis_m": range_axis,
                "velocity_axis_mps": velocity_axis,
            }

    keys = list(test_records[0]["ideal"])
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "https://doi.org/10.25452/figshare.plus.30027313.v1",
        "truth": {"range_m": args.truth_range, "velocity_mps": args.truth_velocity},
        "split": {
            "calibration_files": [path.name for path in calibration_files],
            "test_files": [path.name for path in test_files],
        },
        "calibrated_psf": {
            "range_bias_m": range_bias,
            "velocity_bias_mps": velocity_bias,
            "range_width_3db_m": range_width,
            "velocity_width_3db_mps": velocity_width,
            "noise_floor_median_db": noise_floor,
        },
        "measured_repeatability": {
            "range_error_m": summarize([item["truth_range_error_m"] for item in test_records]),
            "velocity_error_mps": summarize([item["truth_velocity_error_mps"] for item in test_records]),
            "peak_range_m": summarize([item["measured_peak_range_m"] for item in test_records]),
            "peak_velocity_mps": summarize([item["measured_peak_velocity_mps"] for item in test_records]),
        },
        "ideal_summary": {key: summarize([item["ideal"][key] for item in test_records]) for key in keys},
        "calibrated_summary": {key: summarize([item["calibrated"][key] for item in test_records]) for key in keys},
        "records": test_records,
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("/nas/datasets/hyt/datasets/radar_open/tsms_drone_sample/processed") / f"corner_2m_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    (output / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if representative:
        np.savez_compressed(output / "representative_rd.npz", **representative)
        measured_abs = np.abs(representative["measured_rd"])
        calibrated_abs = np.abs(representative["calibrated_rd"])
        measured_db = 20.0 * np.log10(measured_abs / np.max(measured_abs) + 1.0e-12)
        calibrated_db = 20.0 * np.log10(calibrated_abs / np.max(calibrated_abs) + 1.0e-12)
        range_axis = representative["range_axis_m"]
        velocity_axis = representative["velocity_axis_mps"]
        range_mask = (range_axis >= max(0.0, args.truth_range - 1.0)) & (range_axis <= args.truth_range + 1.0)
        velocity_mask = (velocity_axis >= -1.0) & (velocity_axis <= 1.0)
        extent = [
            float(velocity_axis[velocity_mask][0]),
            float(velocity_axis[velocity_mask][-1]),
            float(range_axis[range_mask][0]),
            float(range_axis[range_mask][-1]),
        ]
        figure, axes = plt.subplots(2, 2, figsize=(11.5, 8.0), constrained_layout=True)
        axes[0, 0].imshow(
            measured_db[np.ix_(range_mask, velocity_mask)],
            origin="lower", aspect="auto", extent=extent, vmin=-45, vmax=0, cmap="turbo",
        )
        axes[0, 0].set_title("Measured corner-reflector RD")
        axes[0, 1].imshow(
            calibrated_db[np.ix_(range_mask, velocity_mask)],
            origin="lower", aspect="auto", extent=extent, vmin=-45, vmax=0, cmap="turbo",
        )
        axes[0, 1].set_title("Calibrated simple-target RD")
        for axis in axes[0]:
            axis.set_xlabel("Radial velocity (m/s)")
            axis.set_ylabel("Range (m)")

        axes[1, 0].plot(range_axis, np.max(measured_db, axis=1), label="Measured")
        axes[1, 0].plot(range_axis, np.max(calibrated_db, axis=1), label="Calibrated")
        axes[1, 0].axvline(args.truth_range, color="black", linestyle="--", label="2 m truth")
        axes[1, 0].set_xlim(max(0.0, args.truth_range - 1.0), args.truth_range + 1.0)
        axes[1, 0].set_ylim(-45, 2)
        axes[1, 0].set_title("Range profiles")
        axes[1, 0].set_xlabel("Range (m)")
        axes[1, 0].set_ylabel("Normalized magnitude (dB)")
        axes[1, 0].legend()

        measured_ranges = [item["measured_peak_range_m"] for item in test_records]
        measured_velocities = [item["measured_peak_velocity_mps"] for item in test_records]
        axes[1, 1].scatter(measured_ranges, measured_velocities, alpha=0.75, label="30 test captures")
        axes[1, 1].axvline(args.truth_range, color="black", linestyle="--")
        axes[1, 1].axhline(args.truth_velocity, color="black", linestyle="--")
        axes[1, 1].set_title("Measured peak repeatability")
        axes[1, 1].set_xlabel("Peak range (m)")
        axes[1, 1].set_ylabel("Peak velocity (m/s)")
        axes[1, 1].legend()
        figure.suptitle("TSMS 24 GHz · 2 m Corner Reflector Validation", fontsize=15)
        figure.savefig(output / "corner_2m_validation.png", dpi=180)
        figure.savefig(output / "corner_2m_validation.pdf")
        plt.close(figure)

    ideal = summary["ideal_summary"]
    calibrated = summary["calibrated_summary"]
    repeat = summary["measured_repeatability"]
    report = f"""# TSMS 2m角反射器简单目标验证

- 校准/测试：{len(calibration_files)} / {len(test_files)} 次独立重复测量
- 实测峰值距离：{repeat['peak_range_m']['mean']:.4f} ± {repeat['peak_range_m']['std']:.4f} m
- 实测对2m真值的距离MAE：{repeat['range_error_m']['mean']:.4f} m
- 实测峰值速度：{repeat['peak_velocity_mps']['mean']:.4f} ± {repeat['peak_velocity_mps']['std']:.4f} m/s
- 实测对0m/s真值的速度MAE：{repeat['velocity_error_mps']['mean']:.4f} m/s
- 校准得到固定range bias：{range_bias:.4f} m
- 校准Range/Velocity -3dB宽度：{range_width:.4f} m / {velocity_width:.4f} m/s

| 指标 | 理想点目标 | 校准PSF |
|---|---:|---:|
| 距离峰值误差 / m | {ideal['peak_range_error_m']['mean']:.4f} | {calibrated['peak_range_error_m']['mean']:.4f} |
| 速度峰值误差 / m/s | {ideal['peak_velocity_error_mps']['mean']:.4f} | {calibrated['peak_velocity_error_mps']['mean']:.4f} |
| Range -3dB宽度误差 / m | {ideal['range_width_3db_error_m']['mean']:.4f} | {calibrated['range_width_3db_error_m']['mean']:.4f} |
| Range PSLR误差 / dB | {ideal['range_pslr_error_db']['mean']:.4f} | {calibrated['range_pslr_error_db']['mean']:.4f} |
| Range ISLR误差 / dB | {ideal['range_islr_error_db']['mean']:.4f} | {calibrated['range_islr_error_db']['mean']:.4f} |
| 噪声底误差 / dB | {ideal['noise_floor_error_db']['mean']:.4f} | {calibrated['noise_floor_error_db']['mean']:.4f} |
| Range Wasserstein / m | {ideal['range_wasserstein_m']['mean']:.4f} | {calibrated['range_wasserstein_m']['mean']:.4f} |
| Velocity Wasserstein / m/s | {ideal['velocity_wasserstein_mps']['mean']:.4f} | {calibrated['velocity_wasserstein_mps']['mean']:.4f} |

理想点目标用于验证物理分辨率；校准PSF只使用前{len(calibration_files)}次测量估计固定偏置、主瓣宽度和噪声底，后{len(test_files)}次测量不参与校准。残余PSLR/ISLR和Wasserstein差异主要反映真实窗函数、硬件滤波、多径和非白背景尚未进入模型。
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Output: {output}")


if __name__ == "__main__":
    main()
