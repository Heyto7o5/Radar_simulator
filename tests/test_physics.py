import math

import numpy as np
import pytest

from radar_platform.models import RadarConfig, default_request
from radar_platform.physics import (
    alias_doppler,
    angle_between,
    derived_metrics,
    platform_velocity_ned,
    request_warnings,
    target_geometry,
)


def test_default_derived_metrics() -> None:
    metrics = derived_metrics(RadarConfig())
    assert metrics.wavelength_m == pytest.approx(0.0299792458)
    assert metrics.range_resolution_m == pytest.approx(2.4982704833)
    assert metrics.range_sample_spacing_m == pytest.approx(0.9993081933)
    assert metrics.max_unambiguous_range_m == pytest.approx(9993.0819333)
    assert metrics.max_unambiguous_velocity_mps == pytest.approx(112.42217175)
    assert metrics.velocity_bin_mps == pytest.approx(3.5131928672)
    assert metrics.num_fast_time_samples == 10_000


def test_default_velocity_uses_ned_down_positive() -> None:
    request = default_request()
    velocity = platform_velocity_ned(request.platform)
    assert np.linalg.norm(velocity) == pytest.approx(1200.0)
    assert velocity[0] > 0.0
    assert velocity[2] > 0.0
    assert math.degrees(math.atan2(velocity[2], velocity[0])) == pytest.approx(80.0)


def test_alias_doppler_reports_order() -> None:
    aliased, order = alias_doppler(80_000.0, 15_000.0)
    assert aliased == pytest.approx(5_000.0)
    assert order == 5


def test_target_geometry_closing_velocity_positive() -> None:
    distance, los, radial = target_geometry(
        np.array([0.0, 0.0, -5000.0]),
        np.array([0.0, 0.0, 1200.0]),
        np.array([0.0, 0.0, 0.0]),
    )
    assert distance == pytest.approx(5000.0)
    assert los.tolist() == pytest.approx([0.0, 0.0, 1.0])
    assert radial == pytest.approx(1200.0)


def test_angle_between() -> None:
    assert angle_between(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0)
    assert angle_between(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])) == pytest.approx(math.pi / 2)


def test_default_request_warns_about_doppler_ambiguity() -> None:
    warnings = request_warnings(default_request())
    assert any("多普勒模糊" in warning for warning in warnings)


def test_waveform_validation() -> None:
    with pytest.raises(ValueError, match="pulse_width_s"):
        RadarConfig(pulse_width_s=100e-6, prf_hz=15_000)
    with pytest.raises(ValueError, match="sample_rate_hz"):
        RadarConfig(sample_rate_hz=20e6, bandwidth_hz=60e6)

