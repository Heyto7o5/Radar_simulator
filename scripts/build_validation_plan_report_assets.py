#!/usr/bin/env python3
"""Build the static figures used by the RD validation extension plan.

The figures intentionally combine locally held measurements with schematics.
They do not download or invent remote dataset imagery.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle, Wedge
from PIL import Image

from radar_platform.models import clean_demo_request
from radar_platform.ramp_dataset import RAMP_CONFIG, load_real_frame, process_adc
from radar_platform.simulator import simulate_frame
from radar_platform.tsms_validation import load_processed_rd, normalized_power_db, processed_axes


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "docs" / "report_assets"
DATA_ROOT = Path("/nas/datasets/hyt/datasets/radar_open")

INK = "#14213d"
MUTED = "#52647a"
BLUE = "#2374ab"
CYAN = "#2a9d8f"
ORANGE = "#f4a261"
RED = "#d1495b"
LIGHT = "#eef4f8"
GRID = "#d7e0e8"


def configure_style() -> None:
    noto_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(noto_path).exists():
        font_manager.fontManager.addfont(noto_path)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def clean_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.2, color=GRID)


def add_panel_label(axis: plt.Axes, label: str, title: str) -> None:
    axis.text(
        0.0,
        1.04,
        label,
        transform=axis.transAxes,
        color="white",
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": BLUE, "edgecolor": "none"},
    )
    axis.text(
        0.075,
        1.04,
        title,
        transform=axis.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def build_dataset_examples() -> Path:
    figure, axes = plt.subplots(2, 2, figsize=(15.6, 9.0), constrained_layout=True)
    figure.suptitle("已落地的公开数据：场景证据与真实 RD", fontsize=19, fontweight="bold")

    scene = axes[0, 0]
    scene.set_xlim(-1, 31)
    scene.set_ylim(-0.2, 6.0)
    scene.axhspan(-0.2, 0.0, color="#8f9d7a")
    scene.plot([0, 30], [0, 0], color="#60704d", lw=2)
    scene.plot([0, 0], [0, 1.5], color=INK, lw=2)
    scene.scatter([0], [1.5], s=180, marker="^", color=BLUE, zorder=4)
    scene.text(1.0, 2.15, "24 GHz TinyRad\n地面三脚架", ha="left", fontsize=10)
    scene.plot([8, 8], [0, 1.5], color=INK, lw=2)
    corner = Polygon([[7.65, 1.25], [8.0, 1.75], [8.35, 1.25]], closed=True, color=ORANGE)
    scene.add_patch(corner)
    scene.text(8, 2.05, "角反射器\n静止参考靶", ha="center", fontsize=10)
    scene.add_patch(FancyArrowPatch((0.4, 1.5), (7.5, 1.5), arrowstyle="<->", mutation_scale=12, color=RED, lw=1.8))
    scene.text(4, 1.72, "示例 8 m（数据覆盖 2–30 m）", ha="center", color=RED, fontsize=10)
    scene.add_patch(Wedge((0, 1.5), 11.5, -9, 9, color=BLUE, alpha=0.09))
    scene.text(15.5, 4.45, "室外空地 18×38 m\n目标与传感器约 1.5 m 高\n近水平、受控重复测量", ha="center", va="center", fontsize=11, color=MUTED)
    scene.set_xlabel("水平距离 / m")
    scene.set_ylabel("离地高度 / m")
    clean_axis(scene)
    add_panel_label(scene, "A", "TSMS-Drone 对应场景（示意）")

    tsms_path = (
        DATA_ROOT
        / "tsms_drone_sample/extracted/corner_reflector_8m/Raw File/Corner Reflector_8m_031.mat"
    )
    tsms_rd = load_processed_rd(tsms_path)
    tsms_range, tsms_velocity = processed_axes(8.0, tsms_rd.shape)
    tsms_db = normalized_power_db(tsms_rd)
    heat = axes[0, 1].imshow(
        tsms_db.T,
        origin="lower",
        aspect="auto",
        extent=[tsms_range[0], tsms_range[-1], tsms_velocity[0], tsms_velocity[-1]],
        cmap="turbo",
        vmin=-55,
        vmax=0,
        rasterized=True,
    )
    axes[0, 1].axvline(8.0, color="white", ls="--", lw=1.2, alpha=0.9)
    axes[0, 1].text(8.15, 4.1, "已知真值 8 m", color="white", fontsize=10, va="top")
    axes[0, 1].set_xlabel("距离 / m")
    axes[0, 1].set_ylabel("径向速度 / (m/s)")
    add_panel_label(axes[0, 1], "B", "TSMS 角反射器实测复 RD")
    colorbar = figure.colorbar(heat, ax=axes[0, 1], fraction=0.035, pad=0.02)
    colorbar.set_label("归一化功率 / dB")

    camera_path = (
        DATA_ROOT
        / "ramp_adc_77ghz/extracted/Automotive/2019_04_09_bms1000/images_0/0000000003.jpg"
    )
    camera = Image.open(camera_path)
    axes[1, 0].imshow(camera)
    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.02,
        0.05,
        "同步标签：cyclist · 约 9.91 m\n校园停车场 · 低高度水平前视",
        transform=axes[1, 0].transAxes,
        color="white",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": (0.05, 0.08, 0.14, 0.78), "edgecolor": "none"},
    )
    add_panel_label(axes[1, 0], "C", "RAMP 77 GHz 原始 ADC 对应相机场景")

    frame_id = "Automotive/2019_04_09_bms1000/radar_raw_frame/000003.mat"
    adc, _ = load_real_frame(frame_id)
    ramp_db, _, ramp_range, ramp_velocity = process_adc(adc)
    ramp_heat = axes[1, 1].imshow(
        ramp_db,
        origin="lower",
        aspect="auto",
        extent=[ramp_range[0], ramp_range[-1], ramp_velocity[0], ramp_velocity[-1]],
        cmap="turbo",
        vmin=-52,
        vmax=0,
        rasterized=True,
    )
    label_range = math.hypot(-0.72219021, 9.87937847)
    axes[1, 1].axvline(label_range, color="white", ls="--", lw=1.2, alpha=0.9)
    axes[1, 1].text(label_range + 0.35, ramp_velocity[-1] * 0.83, "相机标签距离", color="white", fontsize=10)
    axes[1, 1].set_xlabel("距离 / m")
    axes[1, 1].set_ylabel("径向速度 / (m/s)")
    add_panel_label(axes[1, 1], "D", "同一帧 ADC→RD（8 通道非相干平均）")
    colorbar = figure.colorbar(ramp_heat, ax=axes[1, 1], fraction=0.035, pad=0.02)
    colorbar.set_label("归一化功率 / dB")

    path = OUTPUT_DIR / "open-dataset-scene-rd-examples.png"
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def build_current_platform() -> Path:
    request = clean_demo_request()
    result = simulate_frame(request, 0)
    figure, axes = plt.subplots(1, 2, figsize=(15.6, 6.4), constrained_layout=True)
    figure.suptitle("现有平台：10 GHz 高空近垂直脉冲 LFM 主链路", fontsize=19, fontweight="bold")

    geometry = axes[0]
    height = -request.platform.position_ned_m[2]
    ground_x = np.linspace(-500, 1900, 100)
    geometry.fill_between(ground_x, -0.12, 0.0, color="#9bad83", alpha=0.8)
    geometry.plot(ground_x, np.zeros_like(ground_x), color="#60704d", lw=2)
    platform_x = 0.0
    geometry.scatter([platform_x], [height], s=240, marker="v", color=BLUE, zorder=6)
    geometry.text(platform_x + 70, height + 80, "平台 5 km\n90 m/s · 俯冲 80°", fontsize=10, va="bottom")
    dive = math.radians(request.platform.dive_angle_deg)
    center_x = height / math.tan(dive)
    geometry.plot([platform_x, center_x], [height, 0], color=BLUE, lw=2)
    beam_angle = math.radians(request.radar.beam_half_angle_deg)
    for sign in (-1, 1):
        x = height / math.tan(dive + sign * beam_angle)
        geometry.plot([platform_x, x], [height, 0], color=BLUE, lw=1, alpha=0.5)
    geometry.fill(
        [platform_x, height / math.tan(dive - beam_angle), height / math.tan(dive + beam_angle)],
        [height, 0, 0],
        color=BLUE,
        alpha=0.08,
    )
    for target in request.scene.targets:
        x = target.position_ned_m[0]
        geometry.scatter([x], [0], s=90, color=ORANGE, edgecolor=INK, zorder=5)
        geometry.text(x, -180, target.id, ha="center", fontsize=10)
    geometry.annotate(
        "主波束落点 ≈ 882 m",
        xy=(center_x, 0),
        xytext=(1350, 1800),
        arrowprops={"arrowstyle": "->", "color": RED},
        color=RED,
        fontsize=10,
    )
    geometry.set_xlim(-350, 2000)
    geometry.set_ylim(-300, 5600)
    geometry.set_xlabel("地面沿航向距离 / m")
    geometry.set_ylabel("高度 / m")
    clean_axis(geometry)
    add_panel_label(geometry, "A", "当前理想场景与波束几何")

    metadata = result.metadata
    range_axis = metadata.range_start_m + np.arange(metadata.display_cols) * metadata.range_step_m
    velocity_axis = metadata.velocity_start_mps + np.arange(metadata.display_rows) * metadata.velocity_step_mps
    rd_heat = axes[1].imshow(
        result.display_db,
        origin="lower",
        aspect="auto",
        extent=[range_axis[0], range_axis[-1], velocity_axis[0], velocity_axis[-1]],
        cmap="turbo",
        vmin=-55,
        vmax=0,
        rasterized=True,
    )
    for validation in metadata.validations:
        axes[1].scatter(
            [validation.true_range_m],
            [validation.apparent_velocity_mps],
            marker="o",
            s=50,
            facecolor="none",
            edgecolor="white",
            linewidth=1.3,
        )
        axes[1].text(
            validation.true_range_m + 12,
            validation.apparent_velocity_mps + 2.0,
            validation.target_id,
            color="white",
            fontsize=9,
        )
    axes[1].set_xlabel("斜距 / m")
    axes[1].set_ylabel("表观径向速度 / (m/s)")
    add_panel_label(axes[1], "B", "平台实际生成的 RD 帧")
    colorbar = figure.colorbar(rd_heat, ax=axes[1], fraction=0.035, pad=0.02)
    colorbar.set_label("归一化功率 / dB")

    path = OUTPUT_DIR / "current-platform-scene-rd.png"
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def draw_box(axis: plt.Axes, x: float, y: float, width: float, height: float, title: str, lines: list[str], color: str) -> None:
    axis.add_patch(
        Rectangle((x, y), width, height, facecolor=color, alpha=0.10, edgecolor=color, linewidth=1.6)
    )
    axis.text(x + 0.04 * width, y + height - 0.13 * height, title, fontsize=12, fontweight="bold", color=color, va="top")
    axis.text(
        x + 0.05 * width,
        y + height - 0.30 * height,
        "\n".join(f"• {line}" for line in lines),
        fontsize=10,
        color=INK,
        va="top",
        linespacing=1.45,
    )


def build_scene_bridge() -> Path:
    figure, axis = plt.subplots(figsize=(15.6, 7.0), constrained_layout=True)
    figure.suptitle("缺失的不是一张图，而是可追溯的“场景—雷达—数据”映射", fontsize=19, fontweight="bold")
    axis.set_xlim(0, 15.6)
    axis.set_ylim(0, 7.0)
    axis.axis("off")

    draw_box(axis, 0.35, 3.55, 4.05, 2.55, "公开数据证据", ["真实 ADC / RD", "相机、标签或已知距离", "雷达配置与时间戳"], BLUE)
    draw_box(axis, 5.75, 3.55, 4.05, 2.55, "统一 Scene Manifest", ["雷达位姿 / 外参 / 轨迹", "目标几何、姿态与速度", "地形、材料、天气与不确定度"], CYAN)
    draw_box(axis, 11.15, 3.55, 4.05, 2.55, "同条件仿真与对比", ["同波形、同采样、同坐标轴", "ADC → RD → 检测", "峰值、PSF、分布与时序指标"], ORANGE)
    for x0, x1 in ((4.45, 5.68), (9.85, 11.08)):
        axis.add_patch(FancyArrowPatch((x0, 4.83), (x1, 4.83), arrowstyle="-|>", mutation_scale=18, lw=2, color=MUTED))

    axis.text(0.35, 2.75, "三种重建等级", fontsize=12, fontweight="bold")
    levels = [
        ("A 受控真值", "角反射器 / 标定球 / 已知位置", "可做强验证", CYAN),
        ("B 标签条件", "相机 / LiDAR / 3D 框 / UID", "验证几何与量级", BLUE),
        ("C 统计背景", "未知建筑、地面、多径模板", "只做分布验证", ORANGE),
    ]
    for index, (title, evidence, claim, color) in enumerate(levels):
        x = 0.35 + index * 5.05
        axis.add_patch(Rectangle((x, 0.55), 4.55, 1.65, facecolor=LIGHT, edgecolor=GRID, linewidth=1.1))
        axis.text(x + 0.18, 1.82, title, fontsize=11, fontweight="bold", color=color)
        axis.text(x + 0.18, 1.36, evidence, fontsize=10, color=INK)
        axis.text(x + 0.18, 0.89, claim, fontsize=10, color=MUTED)
    axis.text(15.2, 0.16, "原则：缺失信息必须标注为未知，不生成虚假真值", ha="right", fontsize=10, color=RED)

    path = OUTPUT_DIR / "scene-reconstruction-bridge.png"
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def build_roadmap() -> Path:
    figure, axis = plt.subplots(figsize=(15.6, 5.6), constrained_layout=True)
    figure.suptitle("验证扩展路线：先形成可比数据，再增加场景复杂度", fontsize=19, fontweight="bold")
    axis.set_xlim(0, 16)
    axis.set_ylim(0, 5.7)
    axis.axis("off")
    axis.plot([1.0, 15.0], [3.2, 3.2], color=GRID, lw=5, solid_capstyle="round")
    phases = [
        (1.3, "P0 已完成", "参考靶闭环", ["TSMS 6 距离", "校准/测试隔离", "RD 指标表"], CYAN),
        (5.2, "P1 优先", "波形与数据统一", ["FMCW 接入主内核", "阵元级复数据", "Scene Manifest"], BLUE),
        (9.2, "P2 随后", "目标与背景补齐", ["分层目标库", "泄漏/多径模板", "连续距离插值"], ORANGE),
        (13.2, "P3 回归主场景", "高空近垂直验证", ["真实 DEM", "10 GHz 地表模型", "外场标定方案"], RED),
    ]
    for x, phase, title, lines, color in phases:
        axis.scatter([x], [3.2], s=230, color=color, edgecolor="white", linewidth=2, zorder=5)
        axis.text(x, 4.55, phase, ha="center", fontsize=11, fontweight="bold", color=color)
        axis.text(x, 4.05, title, ha="center", fontsize=12, fontweight="bold")
        axis.text(x, 2.55, "\n".join(lines), ha="center", va="top", fontsize=10, color=MUTED, linespacing=1.45)
    axis.text(3.25, 3.55, "方法可复用", ha="center", fontsize=9, color=MUTED)
    axis.text(7.2, 3.55, "能力新增", ha="center", fontsize=9, color=MUTED)
    axis.text(11.2, 3.55, "保真增强", ha="center", fontsize=9, color=MUTED)
    axis.text(8.0, 0.35, "本路线专注 RD / ADC / 阵列与场景验证，不展开 SAR", ha="center", fontsize=11, color=INK)

    path = OUTPUT_DIR / "validation-extension-roadmap.png"
    figure.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(figure)
    return path


def main() -> None:
    configure_style()
    os.environ.setdefault("RADAR_DATA_ROOT", str(DATA_ROOT))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        build_dataset_examples(),
        build_current_platform(),
        build_scene_bridge(),
        build_roadmap(),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
