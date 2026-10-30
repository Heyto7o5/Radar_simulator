import numpy as np
import pytest

from radar_platform.ramp_comparison import (
    SimulationTuning,
    compare_rv_slices,
    official_range_axis_m,
    official_rv_slices,
    official_velocity_axis_mps,
    simulate_label_conditioned_adc,
)
from radar_platform.ramp_dataset import synthesize_adc


def test_official_rv_processing_shape_and_axes() -> None:
    adc, _ = synthesize_adc()
    slices = official_rv_slices(adc)
    assert slices.shape == (2, 128, 128)
    assert official_range_axis_m().shape == (128,)
    assert official_velocity_axis_mps().shape == (128,)


def test_label_conditioned_simulation_is_deterministic() -> None:
    _, labels = synthesize_adc()
    tuning = SimulationTuning(name="test", scatterers_per_target=3, clutter_count=8, clutter_amplitude=0.01, noise_std=0.01)
    first = simulate_label_conditioned_adc(labels, tuning, seed=9)
    second = simulate_label_conditioned_adc(labels, tuning, seed=9)
    assert first.shape == (128, 255, 4, 2)
    assert np.array_equal(first, second)


def test_identical_rv_slices_have_zero_distribution_distance() -> None:
    rng = np.random.default_rng(4)
    amplitude = np.abs(rng.normal(size=(128, 128))) + 0.01
    metrics = compare_rv_slices(amplitude, amplitude, [])
    assert metrics["range_wasserstein_normalized"] == pytest.approx(0.0)
    assert metrics["velocity_wasserstein_normalized"] == pytest.approx(0.0)
    assert metrics["histogram_js_divergence"] == pytest.approx(0.0)
    assert metrics["pixel_correlation"] == pytest.approx(1.0)

