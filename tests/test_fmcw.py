from __future__ import annotations

import numpy as np
import pytest

from radar_platform.fmcw import FMCWConfig, FMCWPointTarget, process_fmcw_rd, simulate_fmcw_if


def tsms_config() -> FMCWConfig:
    return FMCWConfig(
        start_hz=24.0e9,
        stop_hz=24.25e9,
        ramp_s=512.0e-6,
        chirp_period_s=600.0e-6,
        samples=512,
        chirps=64,
        sample_rate_hz=1.0e6,
    )


def test_fmcw_point_target_range_and_velocity() -> None:
    config = tsms_config()
    target = FMCWPointTarget(range_m=8.0, radial_velocity_mps=0.65)
    data = simulate_fmcw_if(config, [target], seed=9)
    result = process_fmcw_rd(
        data,
        config,
        range_fft_size=4096,
        doppler_fft_size=4096,
        range_window=np.hamming(config.samples - 1),
        doppler_window=np.hanning(config.chirps),
        discard_first_sample=True,
        range_bin_count=300,
    )
    power = np.abs(result.rd[:, :, 0]) ** 2
    range_index, velocity_index = np.unravel_index(int(np.argmax(power)), power.shape)
    range_error = abs(result.range_axis_m[range_index] - target.range_m)
    velocity_error = abs(result.velocity_axis_mps[velocity_index] - target.radial_velocity_mps)
    assert range_error <= result.range_axis_m[1] - result.range_axis_m[0]
    assert velocity_error <= result.velocity_axis_mps[1] - result.velocity_axis_mps[0]


def test_fmcw_if_is_seed_deterministic() -> None:
    config = tsms_config()
    target = FMCWPointTarget(range_m=4.0)
    first = simulate_fmcw_if(config, [target], noise_std=0.02, seed=42)
    second = simulate_fmcw_if(config, [target], noise_std=0.02, seed=42)
    different = simulate_fmcw_if(config, [target], noise_std=0.02, seed=43)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


def test_fmcw_allows_partial_ramp_sampling_and_rejects_overrun() -> None:
    config=FMCWConfig(24e9,24.25e9,512e-6,600e-6,512,64,2e6)
    assert config.adc_duration_s==pytest.approx(256e-6)
    assert config.observed_bandwidth_hz==pytest.approx(125e6)
    with pytest.raises(ValueError, match="sampling window"):
        FMCWConfig(
            start_hz=24.0e9,
            stop_hz=24.25e9,
            ramp_s=512.0e-6,
            chirp_period_s=600.0e-6,
            samples=512,
            chirps=64,
            sample_rate_hz=0.5e6,
        )
