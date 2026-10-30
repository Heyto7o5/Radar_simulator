import numpy as np

from radar_platform.models import SceneConfig, SequenceConfig, SimulationRequest, clean_demo_request, default_request
from radar_platform.simulator import _clutter_scatterers, simulate_frame


def test_default_three_targets_pass_analytic_validation() -> None:
    result = simulate_frame(default_request(), 0)
    assert result.rd_complex.shape == (64, 10_000)
    assert result.display_db.shape[0] == 64
    assert len(result.metadata.validations) == 3
    assert all(validation.passed for validation in result.metadata.validations)
    assert all((validation.range_error_m or 99.0) <= 1.0 for validation in result.metadata.validations)
    assert all(
        validation.doppler_bin_error is not None and validation.doppler_bin_error <= 1.0
        for validation in result.metadata.validations
    )


def test_range_doppler_coupling_is_reported_and_compensated() -> None:
    validation = simulate_frame(default_request(), 0).metadata.validations[1]
    assert validation.range_doppler_bias_m > 3.0
    assert validation.raw_detected_range_m is not None
    assert validation.detected_range_m is not None
    assert abs(validation.detected_range_m - validation.true_range_m) < 1.0


def test_clean_demo_uses_same_core_without_doppler_ambiguity() -> None:
    request = clean_demo_request()
    result = simulate_frame(request, 0)
    assert request.platform.speed_mps == 90.0
    assert all(validation.ambiguity_order == 0 for validation in result.metadata.validations)
    assert all(validation.passed for validation in result.metadata.validations)


def test_sequence_motion_changes_platform_height() -> None:
    request = default_request()
    first = simulate_frame(request, 0)
    second = simulate_frame(request, 1)
    assert second.metadata.platform_position_ned_m[2] > first.metadata.platform_position_ned_m[2]


def test_clutter_generation_is_deterministic() -> None:
    base = default_request()
    request = SimulationRequest(
        radar=base.radar,
        platform=base.platform,
        scene=SceneConfig(
            targets=base.scene.targets,
            simplified_clutter_enabled=True,
            seed=42,
        ),
        sequence=SequenceConfig(frame_count=1, frame_interval_s=0.05),
    )
    position = np.asarray(request.platform.position_ned_m, dtype=np.float64)
    first = _clutter_scatterers(request, position)
    second = _clutter_scatterers(request, position)
    assert len(first) > 0
    assert [item.phase_rad for item in first] == [item.phase_rad for item in second]
