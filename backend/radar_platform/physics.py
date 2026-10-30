from __future__ import annotations

import math

import numpy as np

from .models import C0, DerivedMetrics, PlatformConfig, RadarConfig, SimulationRequest
from .kinematics import relative_geometry


def platform_velocity_ned(platform: PlatformConfig) -> np.ndarray:
    dive = math.radians(platform.dive_angle_deg)
    heading = math.radians(platform.heading_deg)
    horizontal = platform.speed_mps * math.cos(dive)
    return np.array(
        [
            horizontal * math.cos(heading),
            horizontal * math.sin(heading),
            platform.speed_mps * math.sin(dive),
        ],
        dtype=np.float64,
    )


def platform_boresight_ned(platform: PlatformConfig) -> np.ndarray:
    if platform.boresight_ned is not None:
        direction=np.asarray(platform.boresight_ned,dtype=float)
        return direction/np.linalg.norm(direction)
    # Backwards compatible orientation, independent of whether speed is zero.
    dive,heading=map(math.radians,(platform.dive_angle_deg,platform.heading_deg))
    return np.array([math.cos(dive)*math.cos(heading),math.cos(dive)*math.sin(heading),math.sin(dive)])


def derived_metrics(radar: RadarConfig) -> DerivedMetrics:
    wavelength = C0 / radar.carrier_hz
    pri = 1.0 / radar.prf_hz
    warnings: list[str] = []
    if radar.sample_rate_hz < 1.25 * radar.bandwidth_hz:
        warnings.append("采样率接近信号带宽，脉冲压缩旁瓣对参数误差更敏感。")
    return DerivedMetrics(
        wavelength_m=wavelength,
        range_resolution_m=C0 / (2.0 * radar.bandwidth_hz),
        range_sample_spacing_m=C0 / (2.0 * radar.sample_rate_hz),
        max_unambiguous_range_m=C0 * pri / 2.0,
        full_echo_max_range_m=C0 * (pri - radar.pulse_width_s) / 2.0,
        max_unambiguous_velocity_mps=wavelength * radar.prf_hz / 4.0,
        velocity_bin_mps=wavelength * radar.prf_hz / (2.0 * radar.pulses_per_cpi),
        pri_s=pri,
        cpi_s=pri * radar.pulses_per_cpi,
        num_fast_time_samples=int(math.ceil(radar.sample_rate_hz * pri)),
        warnings=warnings,
    )


def alias_doppler(doppler_hz: float, prf_hz: float) -> tuple[float, int]:
    aliased = (doppler_hz + prf_hz / 2.0) % prf_hz - prf_hz / 2.0
    order = int(round((doppler_hz - aliased) / prf_hz))
    return float(aliased), order


def target_geometry(
    platform_position: np.ndarray,
    platform_velocity: np.ndarray,
    target_position: np.ndarray,
    target_velocity: np.ndarray | tuple = (0.,0.,0.),
) -> tuple[float, np.ndarray, float]:
    distance,los,closing=relative_geometry(platform_position,platform_velocity,target_position,target_velocity)
    return float(distance),los,float(closing)


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a_unit = a / np.linalg.norm(a)
    b_unit = b / np.linalg.norm(b)
    cosine = float(np.clip(np.dot(a_unit, b_unit), -1.0, 1.0))
    return math.acos(cosine)


def request_warnings(request: SimulationRequest) -> list[str]:
    radar = request.radar
    metrics = derived_metrics(radar)
    warnings = list(metrics.warnings)
    velocity = platform_velocity_ned(request.platform)
    boresight = platform_boresight_ned(request.platform)
    position = np.asarray(request.platform.position_ned_m, dtype=np.float64)
    ambiguous = False
    outside_range = False
    outside_beam = False
    incomplete_echo = False
    for target in request.scene.targets:
        target_position = np.asarray(target.position_ned_m, dtype=np.float64)
        distance, los, radial_velocity = target_geometry(position, velocity, target_position,target.velocity_ned_mps)
        doppler = 2.0 * radial_velocity / metrics.wavelength_m
        aliased, order = alias_doppler(doppler, radar.prf_hz)
        ambiguous |= order != 0 or not math.isclose(doppler, aliased)
        outside_range |= distance >= metrics.max_unambiguous_range_m
        incomplete_echo |= distance >= metrics.full_echo_max_range_m
        outside_beam |= angle_between(boresight, los) > math.radians(radar.beam_half_angle_deg)
    if ambiguous:
        warnings.append("高速场景存在单 PRF 多普勒模糊；界面显示折叠频率，不执行解模糊。")
    if outside_range:
        warnings.append("至少一个目标超过最大无模糊距离。")
    if incomplete_echo:
        warnings.append("至少一个目标的完整脉冲回波超出当前 PRI 接收窗。")
    if outside_beam:
        warnings.append("至少一个目标位于配置波束之外，将不会产生理想目标回波。")
    if request.scene.simplified_clutter_enabled:
        warnings.append("简化杂波仅用于演示，尚未进行频段、材料或绝对功率标定。")
    return warnings
