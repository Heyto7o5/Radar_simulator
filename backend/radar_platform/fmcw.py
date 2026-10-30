from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .models import C0


@dataclass(frozen=True)
class FMCWConfig:
    """Signal-level FMCW configuration in SI units.

    The sign convention is positive radial velocity for a closing target.
    Each chirp is sampled inside its ramp, starting at adc_start_s. The output IF cube
    uses the public platform convention ``[sample, chirp, channel]``.
    """

    start_hz: float
    stop_hz: float
    ramp_s: float
    chirp_period_s: float
    samples: int
    chirps: int
    sample_rate_hz: float
    adc_start_s: float = 0.0

    def __post_init__(self) -> None:
        if self.start_hz <= 0.0 or self.stop_hz <= self.start_hz:
            raise ValueError("FMCW stop_hz must exceed a positive start_hz")
        if self.ramp_s <= 0.0 or self.chirp_period_s < self.ramp_s:
            raise ValueError("FMCW chirp period must be at least the ramp duration")
        if self.samples < 2 or self.chirps < 2 or self.sample_rate_hz <= 0.0:
            raise ValueError("FMCW sampling dimensions and rate must be positive")
        values=(self.start_hz,self.stop_hz,self.ramp_s,self.chirp_period_s,self.sample_rate_hz,self.adc_start_s)
        if not all(math.isfinite(x) for x in values):raise ValueError("finite FMCW timing and frequency required")
        if self.adc_start_s < 0 or self.adc_start_s+self.adc_duration_s > self.ramp_s*(1+1e-9):
            raise ValueError("ADC sampling window must fit inside ramp_s")

    @property
    def adc_duration_s(self) -> float:
        return self.samples/self.sample_rate_hz

    @property
    def observed_bandwidth_hz(self) -> float:
        return self.slope_hz_s*self.adc_duration_s

    @property
    def idle_s(self) -> float:
        return self.chirp_period_s-self.ramp_s

    @property
    def bandwidth_hz(self) -> float:
        return self.stop_hz - self.start_hz

    @property
    def center_hz(self) -> float:
        return 0.5 * (self.start_hz + self.stop_hz)

    @property
    def slope_hz_s(self) -> float:
        return self.bandwidth_hz / self.ramp_s

    @property
    def wavelength_m(self) -> float:
        return C0 / self.center_hz

    @property
    def native_range_resolution_m(self) -> float:
        return C0 / (2.0 * self.observed_bandwidth_hz)

    @property
    def native_velocity_resolution_mps(self) -> float:
        return self.wavelength_m / (2.0 * self.chirps * self.chirp_period_s)


@dataclass(frozen=True)
class FMCWPointTarget:
    range_m: float
    radial_velocity_mps: float = 0.0
    amplitude: float = 1.0
    phase_rad: float = 0.0
    azimuth_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.range_m <= 0.0:
            raise ValueError("target range must be positive")
        if self.amplitude <= 0.0:
            raise ValueError("target amplitude must be positive")


@dataclass(frozen=True)
class FMCWRDResult:
    rd: np.ndarray
    range_axis_m: np.ndarray
    velocity_axis_mps: np.ndarray
    range_bin_indices: np.ndarray


def simulate_fmcw_if(
    config: FMCWConfig,
    targets: Sequence[FMCWPointTarget],
    receiver_positions_m: Sequence[float] = (0.0,),
    *,
    real_output: bool = True,
    noise_std: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate an independent dechirped IF cube for point targets.

    The simulator starts from target geometry and waveform parameters; it
    does not use a measured RD template.  Range attenuation is relative and
    normalized to the nearest configured target because this path is not an
    absolute power/RCS model.

    Legacy separable positive-beat / closing-positive phase convention. Range
    is fixed within this IF cube; use array_fmcw.simulate_motion_adc for physical
    tx*conj(rx) mixing, 3D trajectories, TDM offsets and cross-frame continuity.
    """

    from .unified.separable_if import SeparableIFRequest, SeparableIFSession
    request=SeparableIFRequest(waveform=config,targets=list(targets),
        receiver_positions_m=list(receiver_positions_m),real_output=real_output,
        noise_std=noise_std,seed=seed)
    return SeparableIFSession(request).synthesize()


def process_fmcw_rd(
    if_cube: np.ndarray,
    config: FMCWConfig,
    *,
    range_fft_size: int,
    doppler_fft_size: int,
    range_window: np.ndarray | None = None,
    doppler_window: np.ndarray | None = None,
    discard_first_sample: bool = False,
    range_bin_start: int = 0,
    range_bin_count: int | None = None,
) -> FMCWRDResult:
    """Process ``[sample, chirp, channel]`` IF data into complex RD data."""

    cube = np.asarray(if_cube)
    if cube.ndim == 2:
        cube = cube[:, :, None]
    expected = (config.samples, config.chirps)
    if cube.ndim != 3 or cube.shape[:2] != expected:
        raise ValueError(f"IF cube must start with dimensions {expected}, got {cube.shape}")
    sample_start = 1 if discard_first_sample else 0
    used_samples = config.samples - sample_start
    if range_fft_size < used_samples or doppler_fft_size < config.chirps:
        raise ValueError("FFT sizes must not be shorter than the processed dimensions")
    if range_window is None:
        range_weights = np.ones(used_samples, dtype=np.float64)
    else:
        range_weights = np.asarray(range_window, dtype=np.float64).reshape(-1)
        if range_weights.size != used_samples:
            raise ValueError("range window length does not match processed samples")
    if doppler_window is None:
        doppler_weights = np.ones(config.chirps, dtype=np.float64)
    else:
        doppler_weights = np.asarray(doppler_window, dtype=np.float64).reshape(-1)
        if doppler_weights.size != config.chirps:
            raise ValueError("Doppler window length does not match chirps")

    weighted = cube[sample_start:] * range_weights[:, None, None]
    range_fft = np.fft.fft(weighted, range_fft_size, axis=0) / max(
        float(np.sum(range_weights)), 1.0e-30
    )
    if range_bin_start < 0 or range_bin_start >= range_fft_size:
        raise ValueError("range_bin_start is outside the Range FFT")
    count = range_bin_count if range_bin_count is not None else range_fft_size - range_bin_start
    if count <= 0 or range_bin_start + count > range_fft_size:
        raise ValueError("requested Range FFT crop is outside the FFT")
    cropped = range_fft[range_bin_start : range_bin_start + count]
    doppler_fft = np.fft.fftshift(
        np.fft.fft(
            cropped * doppler_weights[None, :, None],
            doppler_fft_size,
            axis=1,
        )
        / max(float(np.sum(doppler_weights)), 1.0e-30),
        axes=1,
    )
    range_indices = np.arange(range_bin_start, range_bin_start + count)
    range_axis = (
        range_indices
        * config.sample_rate_hz
        / range_fft_size
        * C0
        / (2.0 * config.slope_hz_s)
    )
    doppler_axis = np.fft.fftshift(
        np.fft.fftfreq(doppler_fft_size, d=config.chirp_period_s)
    )
    velocity_axis = doppler_axis * config.wavelength_m / 2.0
    return FMCWRDResult(
        rd=np.ascontiguousarray(doppler_fft, dtype=np.complex64),
        range_axis_m=range_axis.astype(np.float64),
        velocity_axis_mps=velocity_axis.astype(np.float64),
        range_bin_indices=range_indices.astype(np.int64),
    )
