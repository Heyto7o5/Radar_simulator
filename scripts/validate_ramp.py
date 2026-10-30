#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from radar_platform.ramp_comparison import (
    SimulationTuning,
    aggregate_frame_metrics,
    compare_frame,
    default_tuning_candidates,
    diagnose_alignment,
    official_range_axis_m,
    official_rv_slices,
    official_velocity_axis_mps,
)
from radar_platform.ramp_dataset import list_frames, load_labels, load_real_frame


def evaluate(
    frame_ids: list[str],
    tuning: SimulationTuning,
    seed_offset: int,
    keep_representative: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray] | None]:
    frame_metrics: list[dict[str, Any]] = []
    representative = None
    for index, frame_id in enumerate(frame_ids):
        adc, path = load_real_frame(frame_id)
        labels = load_labels(path)
        metrics, real_slices, simulated_slices = compare_frame(
            adc,
            labels,
            tuning,
            seed=seed_offset + index,
        )
        metrics["frame_id"] = frame_id
        metrics["label_count"] = len(labels)
        frame_metrics.append(metrics)
        if keep_representative and representative is None:
            representative = {
                "real_rv": real_slices,
                "simulated_rv": simulated_slices,
                "range_axis_m": official_range_axis_m(),
                "velocity_axis_mps": official_velocity_axis_mps(),
            }
    return aggregate_frame_metrics(frame_metrics), frame_metrics, representative


def markdown_report(payload: dict[str, Any]) -> str:
    best = payload["selected_tuning"]
    baseline = payload["baseline_test_summary"]
    tuned = payload["tuned_test_summary"]
    real_real = payload["real_real_variability"]
    def metric(summary: dict[str, Any], key: str) -> float:
        return float(summary[key]["mean"])
    lines = [
        "# RAMP 77GHz 仿真—实测 RD 对比报告",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "## 实验约束",
        "",
        f"- 校准帧：{len(payload['calibration_frames'])}，保留测试帧：{len(payload['test_frames'])}。",
        "- 处理链严格复现官方 RV slice：Range Hamming + 128点FFT；两个重叠128-chirp Doppler块；8通道幅度平均。",
        "- 仿真场景由公开标签条件化；背景几何、材质、真实散射中心和接收机标定未知。",
        "- 参数只在校准帧选择，测试帧不参与选择。",
        "",
        "## 选定参数",
        "",
        "```json",
        json.dumps(best, ensure_ascii=False, indent=2),
        "```",
        "",
        "## 保留测试集结果",
        "",
        "| 指标 | 点目标基线 | 标定后 | Real–Real变化参考 |",
        "|---|---:|---:|---:|",
        f"| Range Wasserstein（归一化） | {metric(baseline, 'range_wasserstein_normalized'):.4f} | {metric(tuned, 'range_wasserstein_normalized'):.4f} | {metric(real_real, 'range_wasserstein_normalized'):.4f} |",
        f"| Velocity Wasserstein（归一化） | {metric(baseline, 'velocity_wasserstein_normalized'):.4f} | {metric(tuned, 'velocity_wasserstein_normalized'):.4f} | {metric(real_real, 'velocity_wasserstein_normalized'):.4f} |",
        f"| 幅度直方图 JS 散度 | {metric(baseline, 'histogram_js_divergence'):.4f} | {metric(tuned, 'histogram_js_divergence'):.4f} | {metric(real_real, 'histogram_js_divergence'):.4f} |",
        f"| Range profile相关 | {metric(baseline, 'range_profile_correlation'):.4f} | {metric(tuned, 'range_profile_correlation'):.4f} | {metric(real_real, 'range_profile_correlation'):.4f} |",
        f"| Velocity profile相关 | {metric(baseline, 'velocity_profile_correlation'):.4f} | {metric(tuned, 'velocity_profile_correlation'):.4f} | {metric(real_real, 'velocity_profile_correlation'):.4f} |",
        f"| 逐像素相关 | {metric(baseline, 'pixel_correlation'):.4f} | {metric(tuned, 'pixel_correlation'):.4f} | {metric(real_real, 'pixel_correlation'):.4f} |",
        f"| 噪声底中值误差 / dB | {metric(baseline, 'noise_floor_abs_error_db'):.3f} | {metric(tuned, 'noise_floor_abs_error_db'):.3f} | {metric(real_real, 'noise_floor_abs_error_db'):.3f} |",
        f"| 综合分布目标（越低越好） | {metric(baseline, 'distribution_objective'):.4f} | {metric(tuned, 'distribution_objective'):.4f} | {metric(real_real, 'distribution_objective'):.4f} |",
        "",
        f"实测标签近邻峰值距离/速度 MAE：{tuned['geometry']['real_range_mae_m']} m / {tuned['geometry']['real_velocity_mae_mps']} m/s。",
        f"仿真标签条件距离/速度 MAE：{tuned['geometry']['simulated_range_mae_m']} m / {tuned['geometry']['simulated_velocity_mae_mps']} m/s。",
        "",
        "## 对齐诊断与修改建议",
        "",
    ]
    for item in payload["diagnostics"]:
        lines.extend([
            f"### {item['issue']}",
            "",
            f"可能原因：{item['likely_causes']}",
            "",
            f"下一修改：{item['next_change']}",
            "",
        ])
    lines.extend([
        "## 结论边界",
        "",
        "本报告验证的是公开77GHz传感器配置下的ADC/RD处理、标签条件几何和统计形态。由于数据集没有完整背景几何、材质、散射中心和绝对接收机标定，逐像素一致不能作为当前阶段的主要验收标准。",
    ])
    return "\n".join(lines)


def real_real_metrics(frame_ids: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from radar_platform.ramp_comparison import compare_rv_slices
    pairs: list[dict[str, Any]] = []
    previous = None
    previous_id = None
    for frame_id in frame_ids:
        adc, path = load_real_frame(frame_id)
        slices = official_rv_slices(adc)
        labels = load_labels(path)
        if previous is not None:
            slice_metrics = [compare_rv_slices(previous[i], slices[i], labels) for i in range(2)]
            numeric_keys = [key for key, value in slice_metrics[0].items() if isinstance(value, (int, float))]
            aggregate = {key: float(np.mean([item[key] for item in slice_metrics])) for key in numeric_keys}
            aggregate["geometry"] = slice_metrics[0]["geometry"]
            aggregate["pair"] = [previous_id, frame_id]
            pairs.append(aggregate)
        previous = slices
        previous_id = frame_id
    return aggregate_frame_metrics(pairs), pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate and evaluate label-conditioned RAMP RD simulation")
    parser.add_argument("--max-frames", type=int, default=30)
    parser.add_argument("--calibration-frames", type=int, default=8)
    parser.add_argument("--test-frames", type=int, default=10)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    frames = [item["id"] for item in list_frames(args.max_frames)]
    if len(frames) < 6:
        raise SystemExit("At least six extracted real frames are required. Run prepare-demo --count 30 first.")
    calibration = frames[: min(args.calibration_frames, len(frames) // 2)]
    remaining = frames[len(calibration):]
    test = remaining[-min(args.test_frames, len(remaining)):]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.output is None:
        data_root = Path(os.environ.get("RADAR_DATA_ROOT", "/nas/datasets/hyt/datasets/radar_open"))
        output = data_root / "ramp_adc_77ghz" / "processed" / f"comparison_{timestamp}"
    else:
        output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    baseline = default_tuning_candidates()[0]
    calibration_runs = []
    for tuning in default_tuning_candidates():
        summary, _, _ = evaluate(calibration, tuning, seed_offset=1000)
        calibration_runs.append({"tuning": tuning.to_dict(), "summary": summary})
        print(f"{tuning.name}: calibration objective={summary['distribution_objective']['mean']:.4f}")
    selected_record = min(
        calibration_runs,
        key=lambda item: item["summary"]["distribution_objective"]["mean"],
    )
    selected = SimulationTuning(**selected_record["tuning"])

    baseline_summary, baseline_frames, _ = evaluate(test, baseline, seed_offset=2000)
    tuned_summary, tuned_frames, representative = evaluate(
        test, selected, seed_offset=3000, keep_representative=True
    )
    real_real_summary, real_real_pairs = real_real_metrics(test)
    diagnostics = diagnose_alignment(tuned_summary, real_real_summary)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "official_processing_source": "https://github.com/Xiangyu-Gao/Radar-multiple-perspective-object-detection/blob/main/slice3d.py",
        "calibration_frames": calibration,
        "test_frames": test,
        "candidate_calibration_runs": calibration_runs,
        "selected_tuning": selected.to_dict(),
        "baseline_test_summary": baseline_summary,
        "tuned_test_summary": tuned_summary,
        "real_real_variability": real_real_summary,
        "baseline_frame_metrics": baseline_frames,
        "tuned_frame_metrics": tuned_frames,
        "real_real_pairs": real_real_pairs,
        "diagnostics": diagnostics,
    }
    (output / "comparison.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output / "report.md").write_text(markdown_report(payload), encoding="utf-8")
    if representative is not None:
        np.savez_compressed(output / "representative_rv.npz", **representative)
    print(f"Selected: {selected.name}")
    print(f"Report: {output / 'report.md'}")


if __name__ == "__main__":
    main()
