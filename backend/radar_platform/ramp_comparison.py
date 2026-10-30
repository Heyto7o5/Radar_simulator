"""Historical label-conditioned IF experiments and their frozen evaluation.

Not imported by current simulation/API paths. Retained so past reports can be
reproduced, not as a second supported forward platform. New label scenes use
ramp_alignment.track_configuration and the common run_scene entry point.
"""
from __future__ import annotations

import math
import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy, wasserstein_distance

from .models import C0
from .ramp_dataset import RAMP_CONFIG, normalize_adc_axes
from .ramp_label_priors import relative_class_amplitude as _class_rcs_scale


@dataclass(frozen=True)
class SimulationTuning:
    name: str
    scatterers_per_target: int = 1
    target_scale: float = 1.0
    clutter_count: int = 0
    clutter_amplitude: float = 0.0
    noise_std: float = 1.0e-5
    dc_amplitude: float = 0.0
    micro_doppler_std_mps: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


OFFICIAL_SWEEP_SLOPE_HZ_S = 21.0017e12
OFFICIAL_DOPPLER_FFT = 128
OFFICIAL_RANGE_FFT = 128
COMPARISON_DB_FLOOR = -60.0


def official_range_axis_m() -> np.ndarray:
    frequency = np.arange(OFFICIAL_RANGE_FFT) * RAMP_CONFIG.sample_rate_hz / OFFICIAL_RANGE_FFT
    return frequency * C0 / (2.0 * OFFICIAL_SWEEP_SLOPE_HZ_S)


def official_velocity_axis_mps() -> np.ndarray:
    """Legacy official-slice raw FFT sign, not the closing-positive aligned axis.

    Retained for historical baselines. Use ramp_alignment.process_aligned_adc
    for the matched physical RD/RAD comparison.
    """
    doppler = np.fft.fftshift(
        np.fft.fftfreq(OFFICIAL_DOPPLER_FFT, d=RAMP_CONFIG.same_tx_chirp_period_s)
    )
    return doppler * RAMP_CONFIG.wavelength_m / 2.0


def virtual_channels(adc: np.ndarray) -> np.ndarray:
    """Convert [sample, chirp, rx, tx] to official [sample, virtual_rx, chirp]."""
    normalized = normalize_adc_axes(adc)
    return np.transpose(normalized, (0, 3, 2, 1)).reshape(
        RAMP_CONFIG.num_samples,
        RAMP_CONFIG.num_tx * RAMP_CONFIG.num_rx,
        RAMP_CONFIG.num_chirps,
    )


def official_rv_slices(adc: np.ndarray) -> np.ndarray:
    """Reproduce the official RAMP-CNN RV slice processing.

    Returns amplitude with shape [two overlapping windows, range, velocity].
    """
    data = virtual_channels(adc)
    range_window = np.hamming(OFFICIAL_RANGE_FFT).astype(np.float64)[:, None, None]
    range_fft = np.fft.fft(data * range_window, OFFICIAL_RANGE_FFT, axis=0)
    doppler_window = np.hamming(OFFICIAL_DOPPLER_FFT).astype(np.float64)[None, None, :]
    slices: list[np.ndarray] = []
    for start in (0, 127):
        block = range_fft[:, :, start : start + OFFICIAL_DOPPLER_FFT]
        doppler_fft = np.fft.fftshift(
            np.fft.fft(block * doppler_window, OFFICIAL_DOPPLER_FFT, axis=2),
            axes=2,
        )
        slices.append(np.sum(np.abs(doppler_fft), axis=1) / data.shape[1])
    return np.stack(slices, axis=0).astype(np.float32)


def _scatterer_count(class_id: int, maximum: int) -> int:
    preferred = {0: 5, 80: 8, 2: 12, 3: 8, 5: 18, 7: 18}.get(class_id, 8)
    return max(1, min(maximum, preferred))


def simulate_label_conditioned_adc_legacy(
    labels: list[dict[str, Any]],
    tuning: SimulationTuning,
    seed: int,
) -> np.ndarray:
    """Frozen pre-alignment baseline, including per-frame random target layout."""
    return _simulate_label_adc(labels,tuning,seed,stable_layout=False,scene_seed=42)


def label_scatterers(labels: list[dict[str, Any]],tuning: SimulationTuning,scene_seed: int=42):
    """Deterministic UID-local scatterers; one point is exactly the box center.

    Normalized offsets remain fixed for each UID. If annotations resize the
    bounding box, the layout scales with it; this is not a rigid CAD model.
    The public seed controls geometry, whereas measurement noise has its own
    seed in simulate_label_conditioned_adc. Missing velocities are marked.
    """
    if len({str(x['uid']) for x in labels})!=len(labels):raise ValueError('duplicate UID in one frame')
    result=[]
    for label in sorted(labels,key=lambda x:str(x['uid'])):
        uid=str(label['uid']);class_id=int(label['class_id'])
        key=hashlib.sha256(f'{scene_seed}:{uid}:{class_id}'.encode()).digest()
        rng=np.random.default_rng(int.from_bytes(key[:8],'little'))
        count=_scatterer_count(class_id,tuning.scatterers_per_target)
        offsets=np.zeros((1,2)) if count==1 else rng.uniform(-.5,.5,size=(count,2))
        weights=rng.lognormal(mean=0,sigma=.35,size=count);weights/=np.linalg.norm(weights)
        initial_phases=rng.uniform(-math.pi,math.pi,size=count)
        micro=rng.normal(0,tuning.micro_doppler_std_mps,size=count)
        width=max(.1,float(label['width_m']));length=max(.1,float(label['length_m']))
        for index,(offset,weight,phase,extra_v) in enumerate(zip(offsets,weights,initial_phases,micro)):
            x=float(label['x_m'])+width*offset[0];y=float(label['y_m'])+length*offset[1]
            if y<=0:raise ValueError('target scatterer is behind the supported radar half-space')
            distance=math.hypot(x,y)
            result.append({'uid':uid,'scatterer_index':index,'x_m':x,'y_m':y,'range_m':distance,
                'angle_rad':math.atan2(x,y),'velocity_mps':float(label.get('radial_velocity_mps') or 0.)+extra_v,
                'velocity_available':label.get('radial_velocity_mps') is not None,
                'normalized_offset':offset.tolist(),
                'amplitude':tuning.target_scale*_class_rcs_scale(class_id)*float(weight)*(10/max(distance,1))**2,
                'phase_rad':float(phase)})
    return result


def simulate_label_conditioned_adc(labels: list[dict[str,Any]],tuning: SimulationTuning,seed: int,
                                   *,scene_seed: int=42) -> np.ndarray:
    """Stable label-conditioned legacy IF convention; not a physical CAD twin.

    Corrected center placement and UID-stable geometry. ``seed`` now controls
    nuisance/noise realizations only; ``scene_seed`` controls target layout.
    New 3D motion/physical dechirp remains a separate explicit adapter.
    """
    return _simulate_label_adc(labels,tuning,seed,stable_layout=True,scene_seed=scene_seed)


def _simulate_label_adc(labels,tuning,seed,*,stable_layout,scene_seed):
    """Generate an ideal TDM-MIMO FMCW frame conditioned on public labels.

    The model is intentionally explicit about its unresolved nuisance terms:
    distributed static clutter, white receiver noise, DC leakage and simple
    extended-target scatterers. These parameters are calibrated on separate
    frames and are never treated as measured material/RCS truth.
    """
    cfg = RAMP_CONFIG
    rng = np.random.default_rng(seed)
    fast_time = np.arange(cfg.num_samples, dtype=np.float64) / cfg.sample_rate_hz
    chirp_index = np.arange(cfg.num_chirps, dtype=np.float64)
    adc = np.zeros(
        (cfg.num_samples, cfg.num_chirps, cfg.num_rx, cfg.num_tx),
        dtype=np.complex64,
    )
    element_positions = np.arange(cfg.num_tx * cfg.num_rx, dtype=np.float64) * cfg.wavelength_m / 2.0

    scatterers: list[tuple[float, float, float, float, float]] = []
    for label in ([] if stable_layout else labels):
        class_id = int(label["class_id"])
        count = _scatterer_count(class_id, tuning.scatterers_per_target)
        x0, y0 = float(label["x_m"]), float(label["y_m"])
        width = max(0.1, float(label["width_m"]))
        length = max(0.1, float(label["length_m"]))
        velocity = float(label.get("radial_velocity_mps") or 0.0)
        weights = rng.lognormal(mean=0.0, sigma=0.35, size=count)
        weights /= max(float(np.linalg.norm(weights)), 1.0e-12)
        for index in range(count):
            x = x0 + rng.uniform(-width / 2.0, width / 2.0)
            y = max(0.5, y0 + rng.uniform(-length / 2.0, length / 2.0))
            distance = math.hypot(x, y)
            angle = math.atan2(x, y)
            micro = rng.normal(0.0, tuning.micro_doppler_std_mps)
            amplitude = (
                tuning.target_scale
                * _class_rcs_scale(class_id)
                * float(weights[index])
                * (10.0 / max(distance, 1.0)) ** 2
            )
            scatterers.append((distance, angle, velocity + micro, amplitude, rng.uniform(-math.pi, math.pi)))

    if stable_layout:
        for point in label_scatterers(labels,tuning,scene_seed):
            # Positions describe the current frame. Geometric carrier phase
            # supplies inter-frame range phase; do not also add fd*absolute_time.
            phase=point['phase_rad']-4*math.pi*cfg.carrier_hz*point['range_m']/C0
            scatterers.append((point['range_m'],point['angle_rad'],point['velocity_mps'],point['amplitude'],phase))

    if tuning.clutter_count > 0 and tuning.clutter_amplitude > 0.0:
        clutter_ranges = rng.uniform(1.0, 25.0, tuning.clutter_count)
        clutter_angles = rng.uniform(-math.radians(70.0), math.radians(70.0), tuning.clutter_count)
        clutter_weights = rng.lognormal(mean=-0.4, sigma=0.8, size=tuning.clutter_count)
        for distance, angle, weight in zip(clutter_ranges, clutter_angles, clutter_weights):
            amplitude = tuning.clutter_amplitude * float(weight) * (10.0 / distance) ** 2
            scatterers.append((float(distance), float(angle), 0.0, amplitude, rng.uniform(-math.pi, math.pi)))

    for distance, angle, velocity, amplitude, initial_phase in scatterers:
        doppler_hz = 2.0 * velocity / cfg.wavelength_m
        beat_hz = 2.0 * OFFICIAL_SWEEP_SLOPE_HZ_S * distance / C0 + doppler_hz
        fast_phase = 2.0 * math.pi * beat_hz * fast_time[:, None]
        for tx in range(cfg.num_tx):
            slow_time = chirp_index * cfg.same_tx_chirp_period_s + tx * cfg.chirp_duration_s
            slow_phase = 2.0 * math.pi * doppler_hz * slow_time[None, :]
            base = amplitude * np.exp(1j * (fast_phase + slow_phase + initial_phase))
            for rx in range(cfg.num_rx):
                element = tx * cfg.num_rx + rx
                spatial_phase = 2.0 * math.pi * element_positions[element] * math.sin(angle) / cfg.wavelength_m
                adc[:, :, rx, tx] += (base * np.exp(1j * spatial_phase)).astype(np.complex64)

    if tuning.dc_amplitude > 0.0:
        channel_phase = rng.uniform(-math.pi, math.pi, size=(1, 1, cfg.num_rx, cfg.num_tx))
        adc += (tuning.dc_amplitude * np.exp(1j * channel_phase)).astype(np.complex64)
    if tuning.noise_std > 0.0:
        noise = rng.normal(size=adc.shape) + 1j * rng.normal(size=adc.shape)
        adc += (tuning.noise_std / math.sqrt(2.0) * noise).astype(np.complex64)
    return adc


def normalized_db(amplitude: np.ndarray, floor_db: float = COMPARISON_DB_FLOOR) -> np.ndarray:
    peak = max(float(np.max(amplitude)), 1.0e-30)
    result = 20.0 * np.log10(np.maximum(amplitude, 1.0e-30) / peak)
    return np.clip(result, floor_db, 0.0)


def _probability_profile(db: np.ndarray, axis: int) -> np.ndarray:
    power = 10.0 ** (db / 10.0)
    profile = np.sum(power, axis=axis)
    return profile / max(float(np.sum(profile)), 1.0e-30)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if np.std(a) < 1.0e-12 or np.std(b) < 1.0e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _geometry_errors(
    real_db: np.ndarray,
    simulated_db: np.ndarray,
    labels: list[dict[str, Any]],
    range_axis: np.ndarray,
    velocity_axis: np.ndarray,
) -> dict[str, Any]:
    real_power = 10.0 ** (real_db / 10.0)
    simulated_power = 10.0 ** (simulated_db / 10.0)
    details: list[dict[str, Any]] = []
    for label in labels:
        expected_range = float(label["range_m"])
        range_bin = int(np.argmin(np.abs(range_axis - expected_range)))
        r0, r1 = max(0, range_bin - 5), min(range_axis.size, range_bin + 6)
        velocity = label.get("radial_velocity_mps")
        if velocity is None:
            d0, d1 = 0, velocity_axis.size
        else:
            velocity_bin = int(np.argmin(np.abs(velocity_axis - float(velocity))))
            d0, d1 = max(0, velocity_bin - 6), min(velocity_axis.size, velocity_bin + 7)

        real_local = real_power[r0:r1, d0:d1]
        sim_local = simulated_power[r0:r1, d0:d1]
        if real_local.size == 0 or sim_local.size == 0:
            continue
        real_r, real_d = np.unravel_index(int(np.argmax(real_local)), real_local.shape)
        sim_r, sim_d = np.unravel_index(int(np.argmax(sim_local)), sim_local.shape)
        real_range = float(range_axis[r0 + real_r])
        sim_range = float(range_axis[r0 + sim_r])
        real_velocity = float(velocity_axis[d0 + real_d])
        sim_velocity = float(velocity_axis[d0 + sim_d])
        details.append({
            "uid": label["uid"],
            "expected_range_m": expected_range,
            "real_peak_range_m": real_range,
            "simulated_peak_range_m": sim_range,
            "real_range_error_m": abs(real_range - expected_range),
            "simulated_range_error_m": abs(sim_range - expected_range),
            "expected_velocity_mps": velocity,
            "real_peak_velocity_mps": real_velocity,
            "simulated_peak_velocity_mps": sim_velocity,
            "real_velocity_error_mps": None if velocity is None else abs(real_velocity - float(velocity)),
            "simulated_velocity_error_mps": None if velocity is None else abs(sim_velocity - float(velocity)),
        })
    real_range_errors = [item["real_range_error_m"] for item in details]
    simulated_range_errors = [item["simulated_range_error_m"] for item in details]
    real_velocity_errors = [item["real_velocity_error_mps"] for item in details if item["real_velocity_error_mps"] is not None]
    simulated_velocity_errors = [item["simulated_velocity_error_mps"] for item in details if item["simulated_velocity_error_mps"] is not None]
    return {
        "target_count": len(details),
        "real_range_mae_m": float(np.mean(real_range_errors)) if real_range_errors else None,
        "simulated_range_mae_m": float(np.mean(simulated_range_errors)) if simulated_range_errors else None,
        "real_velocity_mae_mps": float(np.mean(real_velocity_errors)) if real_velocity_errors else None,
        "simulated_velocity_mae_mps": float(np.mean(simulated_velocity_errors)) if simulated_velocity_errors else None,
        "details": details,
    }


def compare_rv_slices(
    real_amplitude: np.ndarray,
    simulated_amplitude: np.ndarray,
    labels: list[dict[str, Any]],
) -> dict[str, Any]:
    if real_amplitude.shape != simulated_amplitude.shape:
        raise ValueError(f"RV slice shape mismatch: {real_amplitude.shape} vs {simulated_amplitude.shape}")
    real_db = normalized_db(real_amplitude)
    simulated_db = normalized_db(simulated_amplitude)
    range_axis = official_range_axis_m()
    velocity_axis = official_velocity_axis_mps()
    range_mask = (range_axis >= 1.0) & (range_axis <= 25.0)
    real_db = real_db[range_mask]
    simulated_db = simulated_db[range_mask]
    cropped_range = range_axis[range_mask]

    real_range = _probability_profile(real_db, axis=1)
    simulated_range = _probability_profile(simulated_db, axis=1)
    real_velocity = _probability_profile(real_db, axis=0)
    simulated_velocity = _probability_profile(simulated_db, axis=0)
    bins = np.linspace(COMPARISON_DB_FLOOR, 0.0, 61)
    real_hist, _ = np.histogram(real_db, bins=bins, density=False)
    simulated_hist, _ = np.histogram(simulated_db, bins=bins, density=False)
    real_hist = (real_hist + 1.0e-8) / np.sum(real_hist + 1.0e-8)
    simulated_hist = (simulated_hist + 1.0e-8) / np.sum(simulated_hist + 1.0e-8)
    real_power_prob = (10.0 ** (real_db / 10.0)).ravel()
    real_power_prob /= np.sum(real_power_prob)
    simulated_power_prob = (10.0 ** (simulated_db / 10.0)).ravel()
    simulated_power_prob /= np.sum(simulated_power_prob)

    range_span = max(float(np.ptp(cropped_range)), 1.0e-12)
    velocity_span = max(float(np.ptp(velocity_axis)), 1.0e-12)
    result = {
        "mae_db": float(np.mean(np.abs(real_db - simulated_db))),
        "nrmse_db": float(np.sqrt(np.mean((real_db - simulated_db) ** 2)) / abs(COMPARISON_DB_FLOOR)),
        "pixel_correlation": _correlation(real_db, simulated_db),
        "range_profile_correlation": _correlation(real_range, simulated_range),
        "velocity_profile_correlation": _correlation(real_velocity, simulated_velocity),
        "range_wasserstein_normalized": float(
            wasserstein_distance(cropped_range, cropped_range, u_weights=real_range, v_weights=simulated_range) / range_span
        ),
        "velocity_wasserstein_normalized": float(
            wasserstein_distance(velocity_axis, velocity_axis, u_weights=real_velocity, v_weights=simulated_velocity) / velocity_span
        ),
        "histogram_js_divergence": float(jensenshannon(real_hist, simulated_hist, base=2.0) ** 2),
        "spectral_entropy_real": float(entropy(real_power_prob, base=2.0)),
        "spectral_entropy_simulated": float(entropy(simulated_power_prob, base=2.0)),
        "spectral_entropy_abs_error": float(abs(entropy(real_power_prob, base=2.0) - entropy(simulated_power_prob, base=2.0))),
        "occupancy_above_minus20_real": float(np.mean(real_db > -20.0)),
        "occupancy_above_minus20_simulated": float(np.mean(simulated_db > -20.0)),
        "occupancy_abs_error": float(abs(np.mean(real_db > -20.0) - np.mean(simulated_db > -20.0))),
        "noise_floor_median_real_db": float(np.median(real_db)),
        "noise_floor_median_simulated_db": float(np.median(simulated_db)),
        "noise_floor_abs_error_db": float(abs(np.median(real_db) - np.median(simulated_db))),
        "geometry": _geometry_errors(real_db, simulated_db, labels, cropped_range, velocity_axis),
    }
    result["distribution_objective"] = float(
        result["range_wasserstein_normalized"]
        + result["velocity_wasserstein_normalized"]
        + result["histogram_js_divergence"]
        + 0.25 * result["occupancy_abs_error"]
        + 0.02 * result["noise_floor_abs_error_db"]
        + 0.02 * result["spectral_entropy_abs_error"]
    )
    return result


def compare_frame(
    real_adc: np.ndarray,
    labels: list[dict[str, Any]],
    tuning: SimulationTuning,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    simulated_adc = simulate_label_conditioned_adc(labels, tuning, seed)
    real_slices = official_rv_slices(real_adc)
    simulated_slices = official_rv_slices(simulated_adc)
    slice_metrics = [
        compare_rv_slices(real_slices[index], simulated_slices[index], labels)
        for index in range(real_slices.shape[0])
    ]
    numeric_keys = [
        key for key, value in slice_metrics[0].items()
        if isinstance(value, (int, float)) and key != "distribution_objective"
    ]
    aggregate = {key: float(np.mean([item[key] for item in slice_metrics])) for key in numeric_keys}
    aggregate["distribution_objective"] = float(np.mean([item["distribution_objective"] for item in slice_metrics]))
    aggregate["geometry"] = slice_metrics[0]["geometry"]
    aggregate["slice_metrics"] = slice_metrics
    return aggregate, real_slices, simulated_slices


def aggregate_frame_metrics(metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not metrics:
        raise ValueError("at least one frame metric is required")
    numeric_keys = [key for key, value in metrics[0].items() if isinstance(value, (int, float))]
    result: dict[str, Any] = {}
    for key in numeric_keys:
        values = np.asarray([item[key] for item in metrics], dtype=np.float64)
        result[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "p90": float(np.quantile(values, 0.9)),
        }
    real_range_errors = [
        item["geometry"]["real_range_mae_m"]
        for item in metrics
        if item["geometry"]["real_range_mae_m"] is not None
    ]
    simulated_range_errors = [
        item["geometry"]["simulated_range_mae_m"]
        for item in metrics
        if item["geometry"]["simulated_range_mae_m"] is not None
    ]
    real_velocity_errors = [
        item["geometry"]["real_velocity_mae_mps"]
        for item in metrics
        if item["geometry"]["real_velocity_mae_mps"] is not None
    ]
    simulated_velocity_errors = [
        item["geometry"]["simulated_velocity_mae_mps"]
        for item in metrics
        if item["geometry"]["simulated_velocity_mae_mps"] is not None
    ]
    result["geometry"] = {
        "frames_with_targets": len(simulated_range_errors),
        "real_range_mae_m": float(np.mean(real_range_errors)) if real_range_errors else None,
        "simulated_range_mae_m": float(np.mean(simulated_range_errors)) if simulated_range_errors else None,
        "real_velocity_mae_mps": float(np.mean(real_velocity_errors)) if real_velocity_errors else None,
        "simulated_velocity_mae_mps": float(np.mean(simulated_velocity_errors)) if simulated_velocity_errors else None,
    }
    return result


def diagnose_alignment(
    summary: dict[str, Any],
    real_real_summary: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    diagnostics: list[dict[str, str]] = []
    range_w = summary["range_wasserstein_normalized"]["mean"]
    velocity_w = summary["velocity_wasserstein_normalized"]["mean"]
    histogram_js = summary["histogram_js_divergence"]["mean"]
    noise_error = summary["noise_floor_abs_error_db"]["mean"]
    occupancy_error = summary["occupancy_abs_error"]["mean"]
    pixel_corr = summary["pixel_correlation"]["mean"]
    geometry = summary["geometry"]

    if geometry["simulated_range_mae_m"] is not None and geometry["simulated_range_mae_m"] > 0.75:
        diagnostics.append({
            "issue": "标签条件距离峰值未对齐",
            "likely_causes": "FMCW slope、ADC起始时间、Range FFT方向、相机/雷达外参或标签中心并非主要散射中心。",
            "next_change": "先用角反射器/强单目标确定固定range bias，再锁定range轴；不要通过杂波参数修补几何误差。",
        })
    if geometry["simulated_velocity_mae_mps"] is not None and geometry["simulated_velocity_mae_mps"] > 0.5:
        diagnostics.append({
            "issue": "标签条件速度峰值未对齐",
            "likely_causes": "TDM同发射通道周期、Doppler符号、标签差分噪声、目标内部微动或真实峰值关联错误。",
            "next_change": "核对120μs同Tx间隔和速度轴符号；使用更长UID轨迹做稳健速度拟合。",
        })
    range_reference = real_real_summary["range_wasserstein_normalized"]["mean"] if real_real_summary else None
    velocity_reference = real_real_summary["velocity_wasserstein_normalized"]["mean"] if real_real_summary else None
    histogram_reference = real_real_summary["histogram_js_divergence"]["mean"] if real_real_summary else None
    if range_w > 0.12 or (range_reference and range_w > 2.0 * range_reference):
        diagnostics.append({
            "issue": "距离边缘分布差异较大",
            "likely_causes": "场景中建筑、道路和未标注目标缺失，或距离衰减/RCS模型不正确。",
            "next_change": "加入基于真实空帧估计的距离相关杂波功率曲线，而不是均匀随机杂波。",
        })
    if velocity_w > 0.12 or (velocity_reference and velocity_w > 2.0 * velocity_reference):
        diagnostics.append({
            "issue": "Doppler边缘分布差异较大",
            "likely_causes": "静态杂波脊、车辆/人体扩展速度和微多普勒没有复现。",
            "next_change": "从真实RD估计零Doppler杂波宽度，并按类别加入扩展速度散射中心。",
        })
    if (
        histogram_js > 0.15
        or noise_error > 6.0
        or occupancy_error > 0.08
        or (histogram_reference and histogram_js > 2.0 * histogram_reference)
    ):
        diagnostics.append({
            "issue": "功率动态范围与背景统计未对齐",
            "likely_causes": "热噪声、DC泄漏、AGC/量化、通道增益和密集环境杂波尚未标定。",
            "next_change": "以无目标区域拟合噪声与静态杂波，再用保留测试集验证；不要在测试帧直接调参。",
        })
    if pixel_corr < 0.2:
        diagnostics.append({
            "issue": "逐像素相关性低",
            "likely_causes": "公开标签只给目标框，不给真实散射中心、背景几何、材质和相位，因此像素级场景不可辨识。",
            "next_change": "将逐像素相关降为诊断指标；主要验收采用几何误差、边缘分布和real-real基线。",
        })
    return diagnostics


def default_tuning_candidates() -> list[SimulationTuning]:
    return [
        SimulationTuning(name="point_only", scatterers_per_target=1, noise_std=1.0e-5),
        SimulationTuning(name="extended_only", scatterers_per_target=12, micro_doppler_std_mps=0.15, noise_std=1.0e-5),
        SimulationTuning(name="noise_001", scatterers_per_target=12, micro_doppler_std_mps=0.15, noise_std=0.01),
        SimulationTuning(name="noise_010", scatterers_per_target=12, micro_doppler_std_mps=0.15, noise_std=0.10),
        SimulationTuning(name="noise_020", scatterers_per_target=12, micro_doppler_std_mps=0.15, noise_std=0.20),
        SimulationTuning(name="noise_030", scatterers_per_target=12, micro_doppler_std_mps=0.15, noise_std=0.30),
        SimulationTuning(name="noise_050", scatterers_per_target=12, micro_doppler_std_mps=0.15, noise_std=0.50),
        SimulationTuning(name="clutter64_low", scatterers_per_target=12, clutter_count=64, clutter_amplitude=0.01, noise_std=0.01, micro_doppler_std_mps=0.15),
        SimulationTuning(name="clutter64_noise030", scatterers_per_target=12, clutter_count=64, clutter_amplitude=0.01, noise_std=0.30, micro_doppler_std_mps=0.15),
        SimulationTuning(name="clutter128_mid", scatterers_per_target=12, clutter_count=128, clutter_amplitude=0.025, noise_std=0.01, micro_doppler_std_mps=0.2),
        SimulationTuning(name="clutter128_noise", scatterers_per_target=12, clutter_count=128, clutter_amplitude=0.025, noise_std=0.03, micro_doppler_std_mps=0.2),
        SimulationTuning(name="clutter256_mid", scatterers_per_target=12, clutter_count=256, clutter_amplitude=0.02, noise_std=0.02, micro_doppler_std_mps=0.2),
        SimulationTuning(name="clutter_dc", scatterers_per_target=12, clutter_count=128, clutter_amplitude=0.025, noise_std=0.02, dc_amplitude=0.02, micro_doppler_std_mps=0.2),
        SimulationTuning(name="noise_dc030", scatterers_per_target=12, noise_std=0.30, dc_amplitude=0.05, micro_doppler_std_mps=0.15),
    ]
