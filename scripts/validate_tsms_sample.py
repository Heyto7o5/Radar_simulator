#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from radar_platform.tsms_validation import (
    compare_rd,
    load_raw_sample,
    rd_metrics,
    reconstruct_rd,
    simulate_real_if_point_target,
)


def serializable_metrics(metrics: dict) -> dict:
    return {
        key: value
        for key, value in metrics.items()
        if not isinstance(value, np.ndarray)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a simple point-target model against the TSMS FMCW raw sample")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("/nas/datasets/hyt/datasets/radar_open/tsms_drone_sample/raw/FMCW_RawData.mat"),
    )
    parser.add_argument("--expected-range", type=float, default=None)
    parser.add_argument("--expected-velocity", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    sample = load_raw_sample(args.input)
    measured_rd = reconstruct_rd(sample)
    measured = rd_metrics(
        measured_rd,
        sample.range_axis_m,
        sample.velocity_axis_mps,
        sample.config.native_range_resolution_m,
        sample.config.native_velocity_resolution_mps,
    )
    conditioning_range = args.expected_range or measured["peak_range_m"]

    candidates = np.geomspace(0.002, 1.0, 16)
    calibrations = []
    for noise_std in candidates:
        simulated_data = simulate_real_if_point_target(
            sample,
            conditioning_range,
            args.expected_velocity,
            float(noise_std),
            seed=42,
        )
        simulated_rd = reconstruct_rd(sample, simulated_data)
        simulated = rd_metrics(
            simulated_rd,
            sample.range_axis_m,
            sample.velocity_axis_mps,
            sample.config.native_range_resolution_m,
            sample.config.native_velocity_resolution_mps,
        )
        comparison = compare_rd(
            measured,
            simulated,
            sample.range_axis_m,
            sample.velocity_axis_mps,
        )
        objective = comparison["noise_floor_error_db"] + comparison["range_wasserstein_m"] + comparison["velocity_wasserstein_mps"]
        calibrations.append((objective, float(noise_std), simulated, comparison, simulated_rd))
    _, selected_noise, simulated, comparison, simulated_rd = min(calibrations, key=lambda item: item[0])

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("/nas/datasets/hyt/datasets/radar_open/tsms_drone_sample/processed") / f"point_target_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "https://doi.org/10.25452/figshare.plus.30027313.v1",
        "target_identity_known": args.expected_range is not None,
        "conditioning_range_m": conditioning_range,
        "expected_velocity_mps": args.expected_velocity,
        "selected_noise_std": selected_noise,
        "config": sample.config.__dict__,
        "measured": serializable_metrics(measured),
        "simulated": serializable_metrics(simulated),
        "comparison": comparison,
        "limitations": [
            "The standalone raw sample does not identify target type or range in its MAT metadata.",
            "When --expected-range is omitted, the measured peak range conditions the simulation and is not an independent range validation.",
            "A single sample cannot provide calibration/test separation or repeatability statistics.",
        ],
    }
    (output / "comparison.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    np.savez_compressed(
        output / "rd_data.npz",
        measured_rd=measured_rd,
        simulated_rd=simulated_rd,
        range_axis_m=sample.range_axis_m,
        velocity_axis_mps=sample.velocity_axis_mps,
    )
    report = f"""# TSMS FMCW 简单点目标初步对比

- 样本目标身份/距离写入MAT：{'是' if args.expected_range is not None else '否'}
- 条件距离：{conditioning_range:.4f} m
- 实测峰值：{measured['peak_range_m']:.4f} m, {measured['peak_velocity_mps']:.4f} m/s
- 仿真峰值：{simulated['peak_range_m']:.4f} m, {simulated['peak_velocity_mps']:.4f} m/s
- 距离峰值误差：{comparison['peak_range_error_m']:.4f} m
- 速度峰值误差：{comparison['peak_velocity_error_mps']:.4f} m/s
- Range -3dB宽度误差：{comparison['range_width_3db_error_m']:.4f} m
- Range PSLR误差：{comparison['range_pslr_error_db']:.4f} dB
- Range ISLR误差：{comparison['range_islr_error_db']:.4f} dB
- 噪声底误差：{comparison['noise_floor_error_db']:.4f} dB
- Range Wasserstein：{comparison['range_wasserstein_m']:.4f} m
- Velocity Wasserstein：{comparison['velocity_wasserstein_mps']:.4f} m/s

该独立样本可验证官方RD重建和简单点目标主瓣/背景量级。由于MAT未标目标身份和距离，未传入 `--expected-range` 时，距离由实测峰值反推，只属于处理链烟雾测试。完成角反射器子包解析后，应使用文件名中的已知距离和多次重复测量重新计算独立指标。
"""
    (output / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Output: {output}")


if __name__ == "__main__":
    main()

