"""Historical separable IF approximation. Not physical moving-path dechirping."""
from __future__ import annotations
import math
from typing import Sequence
import numpy as np
from ..models import C0
from ..fmcw import FMCWConfig, FMCWPointTarget

def synthesize_separable_if(
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

    if not targets:
        raise ValueError("at least one FMCW target is required")
    positions = np.asarray(receiver_positions_m, dtype=np.float64)
    if positions.ndim != 1 or positions.size == 0:
        raise ValueError("receiver_positions_m must be a non-empty vector")
    if noise_std < 0.0:
        raise ValueError("noise_std must be non-negative")

    fast_time = config.adc_start_s + np.arange(config.samples, dtype=np.float64) / config.sample_rate_hz
    slow_time = np.arange(config.chirps, dtype=np.float64) * config.chirp_period_s
    dtype = np.float64 if real_output else np.complex128
    cube = np.zeros((config.samples, config.chirps, positions.size), dtype=dtype)
    reference_range = min(target.range_m for target in targets)

    for target in targets:
        doppler_hz = 2.0 * target.radial_velocity_mps / config.wavelength_m
        beat_hz = 2.0 * config.slope_hz_s * target.range_m / C0 + doppler_hz
        temporal_phase = 2.0 * math.pi * (
            beat_hz * fast_time[:, None] + doppler_hz * slow_time[None, :]
        )
        relative_amplitude = target.amplitude * (reference_range / target.range_m) ** 2
        sin_azimuth = math.sin(math.radians(target.azimuth_deg))
        for channel, position_m in enumerate(positions):
            spatial_phase = 2.0 * math.pi * position_m * sin_azimuth / config.wavelength_m
            phase = temporal_phase + target.phase_rad + spatial_phase
            if real_output:
                cube[:, :, channel] += relative_amplitude * np.cos(phase)
            else:
                cube[:, :, channel] += relative_amplitude * np.exp(1j * phase)

    if noise_std:
        rng = np.random.default_rng(seed)
        if real_output:
            cube += rng.normal(0.0, noise_std, size=cube.shape)
        else:
            sigma = noise_std / math.sqrt(2.0)
            cube += sigma * (
                rng.normal(size=cube.shape) + 1j * rng.normal(size=cube.shape)
            )
    output_dtype = np.float32 if real_output else np.complex64
    return np.ascontiguousarray(cube, dtype=output_dtype)
