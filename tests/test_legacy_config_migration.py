import numpy as np
import pytest
from radar_platform.models import default_request
from radar_platform.physics import platform_velocity_ned
from radar_platform.unified.legacy_config import convert_legacy
from radar_platform.unified.session import SimulationSession


def test_legacy_parameters_move_to_common_schema_without_hidden_speed_change():
    old = default_request()
    new, notes = convert_legacy(old)
    np.testing.assert_allclose(new.platform.velocity_ned_mps, platform_velocity_ned(old.platform))
    assert np.linalg.norm(new.platform.velocity_ned_mps) == old.platform.speed_mps
    assert new.waveform.carrier_hz == old.radar.carrier_hz
    assert new.waveform.slot_period_s == 1/old.radar.prf_hz
    assert new.waveform.samples/new.waveform.sample_rate_hz <= new.waveform.slot_period_s*(1+1e-10)
    assert [t.id for t in new.scene.objects] == [t.id for t in old.scene.targets]
    assert new.processing.range_window == 'rectangular'
    assert notes
    frame = SimulationSession(new).frame(0)
    assert np.isfinite(frame.adc).all() and np.max(abs(frame.adc)) > 0


def test_legacy_clutter_is_world_fixed_and_seeded():
    old = default_request()
    old.scene.simplified_clutter_enabled = True
    a, _ = convert_legacy(old)
    b, _ = convert_legacy(old)
    assert a.scene.objects[-1].model_dump() == b.scene.objects[-1].model_dump()
    old.scene.seed += 1
    c, _ = convert_legacy(old)
    assert a.scene.objects[-1].points_local_m == c.scene.objects[-1].points_local_m
    assert a.scene.objects[-1].phases_rad != c.scene.objects[-1].phases_rad
    assert a.scene.objects[-1].velocity_ned_mps == (0., 0., 0.)


def test_legacy_public_entry_is_only_an_adapter_to_shared_engine():
    from radar_platform.simulator import simulate_frame
    old = default_request()
    migrated, _ = convert_legacy(old)
    common = SimulationSession(migrated).frame(0)
    legacy = simulate_frame(old, 0)
    scale = int(np.ceil(old.radar.sample_rate_hz*old.radar.pulse_width_s))*np.hanning(old.radar.pulses_per_cpi).sum()
    expected = (common.products.rd_channels[:, :, 0, 0].T*scale).astype(np.complex64)
    np.testing.assert_array_equal(legacy.rd_complex, expected)
    assert legacy.metadata.simulation_engine == 'unified'
    assert legacy.metadata.compatibility_notes


def test_reused_compatibility_session_rejects_mismatched_configuration():
    from radar_platform.simulator import simulate_frame
    old=default_request();config,_=convert_legacy(old)
    session=SimulationSession(config)
    first=simulate_frame(old,0,session=session)
    second=simulate_frame(old,1,session=session)
    assert first.metadata.simulation_engine==second.metadata.simulation_engine=='unified'
    old.platform.speed_mps=0
    with pytest.raises(ValueError,match='another configuration'):
        simulate_frame(old,0,session=session)
