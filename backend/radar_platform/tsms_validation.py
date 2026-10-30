from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import h5py
from scipy.io import loadmat
from scipy.stats import wasserstein_distance

from .models import C0


@dataclass(frozen=True)
class TSMSConfig:
    start_hz: float
    stop_hz: float
    ramp_s: float
    chirp_period_s: float
    samples: int
    chirps: int

    @property
    def bandwidth_hz(self) -> float:
        return self.stop_hz - self.start_hz

    @property
    def center_hz(self) -> float:
        return (self.start_hz + self.stop_hz) / 2.0

    @property
    def slope_hz_s(self) -> float:
        return self.bandwidth_hz / self.ramp_s

    @property
    def sample_rate_hz(self) -> float:
        return self.samples / self.ramp_s

    @property
    def wavelength_m(self) -> float:
        return C0 / self.center_hz

    @property
    def native_range_resolution_m(self) -> float:
        return C0 / (2.0 * self.bandwidth_hz)

    @property
    def native_velocity_resolution_mps(self) -> float:
        return self.wavelength_m / (2.0 * self.chirps * self.chirp_period_s)


@dataclass
class TSMSRawSample:
    config: TSMSConfig
    data: np.ndarray
    range_window: np.ndarray
    range_scale: float
    doppler_window: np.ndarray
    doppler_scale: float
    range_start_index: int
    range_stop_index: int
    range_axis_m: np.ndarray
    velocity_axis_mps: np.ndarray


def load_raw_sample(path: str | Path) -> TSMSRawSample:
    source = loadmat(path, squeeze_me=True, struct_as_record=False)
    cfg = source["Cfg"]
    config = TSMSConfig(
        start_hz=float(cfg.fStrt),
        stop_hz=float(cfg.fStop),
        ramp_s=float(cfg.TRampUp),
        chirp_period_s=float(cfg.Perd),
        samples=int(cfg.N),
        chirps=int(cfg.FrmMeasSiz),
    )
    data = np.asarray(source["Data"])
    expected_rows = config.samples * config.chirps
    if data.ndim != 2 or data.shape[0] < expected_rows:
        raise ValueError(f"unexpected TSMS Data shape: {data.shape}")
    return TSMSRawSample(
        config=config,
        data=data[:expected_rows],
        range_window=np.asarray(source["Win2D"], dtype=np.float64),
        range_scale=float(source["ScaWin"]),
        doppler_window=np.asarray(source["WinVel2D"], dtype=np.float64),
        doppler_scale=float(source["ScaWinVel"]),
        range_start_index=int(source["RMinIdx"]) - 1,
        range_stop_index=int(source["RMaxIdx"]),
        range_axis_m=np.asarray(source["vRangeExt"], dtype=np.float64),
        velocity_axis_mps=np.asarray(source["vVel"], dtype=np.float64),
    )


def reconstruct_rd(
    sample: TSMSRawSample,
    data: np.ndarray | None = None,
    channel: int = 0,
    fft_size: int = 4096,
) -> np.ndarray:
    raw = sample.data if data is None else np.asarray(data)
    if channel < 0 or channel >= raw.shape[1]:
        raise ValueError(f"channel {channel} out of range")
    cfg = sample.config
    measurement = np.reshape(
        raw[: cfg.samples * cfg.chirps, channel],
        (cfg.samples, cfg.chirps),
        order="F",
    )
    range_fft = np.fft.fft(
        measurement[1:, :] * sample.range_window,
        fft_size,
        axis=0,
    ) / sample.range_scale
    cropped = range_fft[sample.range_start_index : sample.range_stop_index, :]
    rd = np.fft.fftshift(
        np.fft.fft(cropped * sample.doppler_window, fft_size, axis=1)
        / sample.doppler_scale,
        axes=1,
    )
    return rd.astype(np.complex64)


def simulate_real_if_point_target(
    sample: TSMSRawSample,
    range_m: float,
    radial_velocity_mps: float,
    noise_std: float,
    seed: int = 42,
) -> np.ndarray:
    cfg = sample.config
    rng = np.random.default_rng(seed)
    fast_time = np.arange(cfg.samples, dtype=np.float64) / cfg.sample_rate_hz
    slow_time = np.arange(cfg.chirps, dtype=np.float64) * cfg.chirp_period_s
    doppler_hz = 2.0 * radial_velocity_mps / cfg.wavelength_m
    beat_hz = 2.0 * cfg.slope_hz_s * range_m / C0 + doppler_hz
    phase = 2.0 * math.pi * (
        beat_hz * fast_time[:, None]
        + doppler_hz * slow_time[None, :]
    )
    channels: list[np.ndarray] = []
    for channel in range(sample.data.shape[1]):
        channel_phase = channel * 0.17
        signal = np.cos(phase + channel_phase)
        noise = rng.normal(0.0, noise_std, size=signal.shape)
        channels.append(np.ravel(signal + noise, order="F"))
    return np.stack(channels, axis=1).astype(np.float32)


def normalized_power_db(rd: np.ndarray, floor_db: float = -120.0) -> np.ndarray:
    power = np.abs(rd) ** 2
    peak = max(float(np.max(power)), 1.0e-30)
    return np.clip(10.0 * np.log10(power / peak + 1.0e-14), floor_db, 0.0)


def _width_at_level(axis: np.ndarray, profile: np.ndarray, peak_index: int, level_db: float) -> float:
    peak = float(profile[peak_index])
    threshold = peak + level_db
    left = peak_index
    right = peak_index
    while left > 0 and profile[left - 1] >= threshold:
        left -= 1
    while right + 1 < profile.size and profile[right + 1] >= threshold:
        right += 1
    return float(axis[right] - axis[left])


def _profile_quality(
    axis: np.ndarray,
    profile_db: np.ndarray,
    peak_index: int,
    native_resolution: float,
) -> dict[str, float]:
    axis_step = float(abs(axis[1] - axis[0]))
    guard = max(2, int(math.ceil(native_resolution / max(axis_step, 1.0e-12))))
    main_start = max(0, peak_index - guard)
    main_stop = min(profile_db.size, peak_index + guard + 1)
    sidelobes = np.concatenate((profile_db[:main_start], profile_db[main_stop:]))
    pslr = float(np.max(sidelobes) - profile_db[peak_index]) if sidelobes.size else float("nan")
    linear = 10.0 ** (profile_db / 10.0)
    main_energy = float(np.sum(linear[main_start:main_stop]))
    side_energy = float(np.sum(linear) - main_energy)
    islr = 10.0 * math.log10(max(side_energy, 1.0e-30) / max(main_energy, 1.0e-30))
    return {
        "width_3db": _width_at_level(axis, profile_db, peak_index, -3.0),
        "pslr_db": pslr,
        "islr_db": islr,
    }


def rd_metrics(
    rd: np.ndarray,
    range_axis_m: np.ndarray,
    velocity_axis_mps: np.ndarray,
    native_range_resolution_m: float,
    native_velocity_resolution_mps: float,
    expected_range_m: float | None = None,
    expected_velocity_mps: float | None = None,
    search_range_half_width_m: float = 1.0,
    search_velocity_half_width_mps: float = 1.0,
) -> dict[str, Any]:
    amplitude = np.abs(rd)
    peak_amplitude_db_raw = 20.0 * math.log10(max(float(np.max(amplitude)), 1.0e-30))
    db = normalized_power_db(rd)
    global_r, global_v = np.unravel_index(int(np.argmax(db)), db.shape)
    if expected_range_m is None:
        peak_r, peak_v = global_r, global_v
    else:
        range_indices = np.flatnonzero(
            np.abs(range_axis_m - expected_range_m) <= search_range_half_width_m
        )
        if expected_velocity_mps is None:
            velocity_indices = np.arange(velocity_axis_mps.size)
        else:
            velocity_indices = np.flatnonzero(
                np.abs(velocity_axis_mps - expected_velocity_mps)
                <= search_velocity_half_width_mps
            )
        if not range_indices.size or not velocity_indices.size:
            raise ValueError("target search window does not intersect RD axes")
        local = db[np.ix_(range_indices, velocity_indices)]
        local_r, local_v = np.unravel_index(int(np.argmax(local)), local.shape)
        peak_r = int(range_indices[local_r])
        peak_v = int(velocity_indices[local_v])
    range_profile = np.max(db, axis=1)
    velocity_profile = np.max(db, axis=0)
    range_quality = _profile_quality(
        range_axis_m, range_profile, peak_r, native_range_resolution_m
    )
    velocity_quality = _profile_quality(
        velocity_axis_mps,
        velocity_profile,
        peak_v,
        native_velocity_resolution_mps,
    )
    mask = np.ones(db.shape, dtype=bool)
    r_guard = max(4, int(math.ceil(3.0 * native_range_resolution_m / abs(range_axis_m[1] - range_axis_m[0]))))
    v_guard = max(4, int(math.ceil(3.0 * native_velocity_resolution_mps / abs(velocity_axis_mps[1] - velocity_axis_mps[0]))))
    mask[max(0, peak_r-r_guard):peak_r+r_guard+1, max(0, peak_v-v_guard):peak_v+v_guard+1] = False
    noise_floor = float(np.median(db[mask]))
    noise_amplitude_db_raw = 20.0 * math.log10(
        max(float(np.median(amplitude[mask])), 1.0e-30)
    )
    target_to_global_db = float(db[peak_r, peak_v] - db[global_r, global_v])
    return {
        "peak_range_m": float(range_axis_m[peak_r]),
        "peak_velocity_mps": float(velocity_axis_mps[peak_v]),
        "peak_range_bin": int(peak_r),
        "peak_velocity_bin": int(peak_v),
        "range_width_3db_m": range_quality["width_3db"],
        "velocity_width_3db_mps": velocity_quality["width_3db"],
        "range_pslr_db": range_quality["pslr_db"],
        "range_islr_db": range_quality["islr_db"],
        "velocity_pslr_db": velocity_quality["pslr_db"],
        "velocity_islr_db": velocity_quality["islr_db"],
        "noise_floor_median_db": noise_floor,
        "peak_to_noise_db": -noise_floor,
        "global_peak_range_m": float(range_axis_m[global_r]),
        "global_peak_velocity_mps": float(velocity_axis_mps[global_v]),
        "target_to_global_db": target_to_global_db,
        "peak_amplitude_db_raw": peak_amplitude_db_raw,
        "noise_amplitude_db_raw": noise_amplitude_db_raw,
        "range_profile_db": range_profile,
        "velocity_profile_db": velocity_profile,
        "rd_db": db,
    }


def compare_rd(
    measured: dict[str, Any],
    simulated: dict[str, Any],
    range_axis_m: np.ndarray,
    velocity_axis_mps: np.ndarray,
) -> dict[str, float]:
    measured_range_power = 10.0 ** (measured["range_profile_db"] / 10.0)
    simulated_range_power = 10.0 ** (simulated["range_profile_db"] / 10.0)
    measured_range_power /= np.sum(measured_range_power)
    simulated_range_power /= np.sum(simulated_range_power)
    measured_velocity_power = 10.0 ** (measured["velocity_profile_db"] / 10.0)
    simulated_velocity_power = 10.0 ** (simulated["velocity_profile_db"] / 10.0)
    measured_velocity_power /= np.sum(measured_velocity_power)
    simulated_velocity_power /= np.sum(simulated_velocity_power)
    return {
        "peak_range_error_m": abs(measured["peak_range_m"] - simulated["peak_range_m"]),
        "peak_velocity_error_mps": abs(measured["peak_velocity_mps"] - simulated["peak_velocity_mps"]),
        "range_width_3db_error_m": abs(measured["range_width_3db_m"] - simulated["range_width_3db_m"]),
        "velocity_width_3db_error_mps": abs(measured["velocity_width_3db_mps"] - simulated["velocity_width_3db_mps"]),
        "range_pslr_error_db": abs(measured["range_pslr_db"] - simulated["range_pslr_db"]),
        "range_islr_error_db": abs(measured["range_islr_db"] - simulated["range_islr_db"]),
        "noise_floor_error_db": abs(measured["noise_floor_median_db"] - simulated["noise_floor_median_db"]),
        "range_wasserstein_m": float(wasserstein_distance(range_axis_m, range_axis_m, u_weights=measured_range_power, v_weights=simulated_range_power)),
        "velocity_wasserstein_mps": float(wasserstein_distance(velocity_axis_mps, velocity_axis_mps, u_weights=measured_velocity_power, v_weights=simulated_velocity_power)),
        "rd_mae_db": float(np.mean(np.abs(measured["rd_db"] - simulated["rd_db"]))),
    }


def load_processed_rd(path: str | Path) -> np.ndarray:
    """Load TSMS processed complex RD saved as MATLAB v5 or v7.3."""
    source = Path(path)
    try:
        with h5py.File(source, "r") as handle:
            raw = handle["RD"][()]
            if raw.dtype.fields and {"real", "imag"}.issubset(raw.dtype.fields):
                rd = raw["real"] + 1j * raw["imag"]
            else:
                rd = np.asarray(raw)
            # MATLAB v7.3/HDF5 exposes dimensions in reverse order.
            rd = rd.T
    except OSError:
        rd = np.asarray(loadmat(source, squeeze_me=True)["RD"])
    if rd.ndim != 2:
        raise ValueError(f"unexpected processed RD shape: {rd.shape}")
    return rd.astype(np.complex64)


def processed_axes(distance_m: float, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    range_bins, velocity_bins = shape
    if distance_m <= 10.0:
        range_limits = (0.0, 12.0)
    elif distance_m <= 20.0:
        range_limits = (10.0, 22.0)
    else:
        range_limits = (20.0, 32.0)
    return (
        np.linspace(*range_limits, range_bins),
        np.linspace(-5.0, 5.0, velocity_bins),
    )


def simulate_gaussian_point_rd(
    range_axis_m: np.ndarray,
    velocity_axis_mps: np.ndarray,
    range_m: float,
    velocity_mps: float,
    range_fwhm_m: float,
    velocity_fwhm_mps: float,
    noise_floor_db: float,
    seed: int,
) -> np.ndarray:
    """Generate a deliberately simple 2-D point-spread response in RD space."""
    # The requested FWHM is measured on power (the usual -3 dB width), while
    # ``envelope`` is a complex-amplitude response.
    sigma_range = max(range_fwhm_m, 1.0e-9) / (2.0 * math.sqrt(math.log(2.0)))
    sigma_velocity = max(velocity_fwhm_mps, 1.0e-9) / (2.0 * math.sqrt(math.log(2.0)))
    envelope = np.exp(
        -0.5 * ((range_axis_m[:, None] - range_m) / sigma_range) ** 2
        -0.5 * ((velocity_axis_mps[None, :] - velocity_mps) / sigma_velocity) ** 2
    )
    rng = np.random.default_rng(seed)
    noise_amplitude = 10.0 ** (noise_floor_db / 20.0)
    noise = noise_amplitude / math.sqrt(2.0) * (
        rng.normal(size=envelope.shape) + 1j * rng.normal(size=envelope.shape)
    )
    phase = rng.uniform(-math.pi, math.pi)
    return (envelope * np.exp(1j * phase) + noise).astype(np.complex64)
