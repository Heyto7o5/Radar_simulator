import numpy as np
import pytest

from radar_platform.tsms_validation import (
    normalized_power_db,
    processed_axes,
    rd_metrics,
    simulate_gaussian_point_rd,
)


def test_normalized_power_db_has_zero_peak() -> None:
    rd = np.array([[1.0 + 0j, 2.0 + 0j]])
    db = normalized_power_db(rd)
    assert np.max(db) == pytest.approx(0.0)
    assert db[0, 0] == pytest.approx(-6.0205999)


def test_gaussian_point_rd_peaks_at_requested_cell() -> None:
    ranges, velocities = processed_axes(2.0, (161, 4096))
    rd = simulate_gaussian_point_rd(
        ranges,
        velocities,
        range_m=2.25,
        velocity_mps=0.0,
        range_fwhm_m=0.675,
        velocity_fwhm_mps=0.217,
        noise_floor_db=-100.0,
        seed=3,
    )
    metrics = rd_metrics(rd, ranges, velocities, 0.6, 0.19)
    assert metrics["peak_range_m"] == pytest.approx(2.25, abs=0.075)
    assert metrics["peak_velocity_mps"] == pytest.approx(0.0, abs=0.003)
    assert metrics["range_width_3db_m"] == pytest.approx(0.675, abs=0.15)


def test_expected_range_window_ignores_stronger_unrelated_peak() -> None:
    ranges = np.linspace(0.0, 12.0, 161)
    velocities = np.linspace(-5.0, 5.0, 4096)
    rd = np.zeros((161, 4096), dtype=np.complex64)
    zero_velocity = int(np.argmin(np.abs(velocities)))
    rd[int(np.argmin(np.abs(ranges - 0.4))), zero_velocity] = 1.0
    target_bin = int(np.argmin(np.abs(ranges - 10.0)))
    rd[target_bin, zero_velocity] = 0.5
    metrics = rd_metrics(
        rd, ranges, velocities, 0.6, 0.19,
        expected_range_m=10.0,
        expected_velocity_mps=0.0,
    )
    assert metrics["peak_range_m"] == pytest.approx(ranges[target_bin])
    assert metrics["global_peak_range_m"] == pytest.approx(ranges[int(np.argmin(np.abs(ranges - 0.4)))])
    assert metrics["target_to_global_db"] == pytest.approx(-6.0205999)
