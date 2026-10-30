from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


C0 = 299_792_458.0


class RadarConfig(BaseModel):
    carrier_hz: float = Field(10.0e9, ge=1.0e8, le=100.0e9)
    bandwidth_hz: float = Field(60.0e6, ge=1.0e6, le=500.0e6)
    sample_rate_hz: float = Field(150.0e6, ge=2.0e6, le=500.0e6)
    pulse_width_s: float = Field(20.0e-6, gt=0.0, le=100.0e-6)
    prf_hz: float = Field(15.0e3, ge=1.0e3, le=100.0e3)
    pulses_per_cpi: Literal[32, 64, 128] = 64
    beam_half_angle_deg: float = Field(4.0, ge=1.0, le=10.0)

    @model_validator(mode="after")
    def validate_waveform(self) -> "RadarConfig":
        pri = 1.0 / self.prf_hz
        if self.pulse_width_s >= pri:
            raise ValueError("pulse_width_s must be shorter than PRI")
        if self.sample_rate_hz < self.bandwidth_hz:
            raise ValueError("complex baseband sample_rate_hz must be >= bandwidth_hz")
        return self


class PlatformConfig(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    position_ned_m: tuple[float, float, float] = (0.0, 0.0, -5000.0)
    speed_mps: float = Field(1200.0, ge=0.0, le=4000.0)
    dive_angle_deg: float = Field(80.0, ge=0.0, le=89.9)
    heading_deg: float = Field(0.0, ge=-360.0, le=360.0)
    boresight_ned: tuple[float, float, float] | None = None

    @model_validator(mode="after")
    def validate_boresight(self):
        if self.boresight_ned is not None and sum(x*x for x in self.boresight_ned) <= 0:
            raise ValueError("boresight_ned must be a nonzero finite direction")
        return self


class TargetConfig(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    id: str
    position_ned_m: tuple[float, float, float]
    velocity_ned_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    amplitude: float = Field(1.0, gt=0.0, le=100.0)


class SceneConfig(BaseModel):
    targets: list[TargetConfig] = Field(default_factory=list)
    simplified_clutter_enabled: bool = False
    seed: int = 42

    @model_validator(mode="after")
    def validate_targets(self) -> "SceneConfig":
        if not self.targets:
            raise ValueError("at least one target is required")
        ids = [target.id for target in self.targets]
        if len(ids) != len(set(ids)):
            raise ValueError("target ids must be unique")
        return self


class SequenceConfig(BaseModel):
    frame_count: int = Field(10, ge=1, le=20)
    frame_interval_s: float = Field(0.05, ge=0.005, le=1.0)


class SimulationRequest(BaseModel):
    radar: RadarConfig = Field(default_factory=RadarConfig)
    platform: PlatformConfig = Field(default_factory=PlatformConfig)
    scene: SceneConfig
    sequence: SequenceConfig = Field(default_factory=SequenceConfig)

    @model_validator(mode="after")
    def validate_memory(self) -> "SimulationRequest":
        pri = 1.0 / self.radar.prf_hz
        samples = math.ceil(self.radar.sample_rate_hz * pri)
        # Full complex RD files plus transient signal/FFT workspaces.
        estimated_bytes = (
            self.sequence.frame_count
            * self.radar.pulses_per_cpi
            * samples
            * 8
        )
        if estimated_bytes > 2_000_000_000:
            raise ValueError("estimated output exceeds the 2 GB demo safety limit")
        return self


class DerivedMetrics(BaseModel):
    wavelength_m: float
    range_resolution_m: float
    range_sample_spacing_m: float
    max_unambiguous_range_m: float
    full_echo_max_range_m: float
    max_unambiguous_velocity_mps: float
    velocity_bin_mps: float
    pri_s: float
    cpi_s: float
    num_fast_time_samples: int
    warnings: list[str] = Field(default_factory=list)


class TargetValidation(BaseModel):
    validation_mode: Literal["truth_assisted_analytic_check"] = "truth_assisted_analytic_check"
    target_id: str
    visible: bool
    in_beam: bool
    true_range_m: float
    raw_detected_range_m: float | None = None
    range_doppler_bias_m: float = 0.0
    detected_range_m: float | None = None
    range_error_m: float | None = None
    true_radial_velocity_mps: float
    true_doppler_hz: float
    aliased_doppler_hz: float
    ambiguity_order: int
    apparent_velocity_mps: float
    detected_apparent_velocity_mps: float | None = None
    doppler_bin_error: float | None = None
    expected_range_bin: int | None = None
    expected_doppler_bin: int | None = None
    detected_range_bin: int | None = None
    detected_doppler_bin: int | None = None
    passed: bool = False


class BlindDetection(BaseModel):
    """Peak extraction receives the RD only; matching to truth is a later operation."""
    doppler_bin: int
    range_bin: int
    raw_range_m: float
    apparent_velocity_mps: float
    relative_power_db: float
    range_using_apparent_doppler_m: float
    limitation: str = "Apparent Doppler only; distance correction is invalid for unresolved Doppler ambiguity."


class FrameMetadata(BaseModel):
    simulation_engine: str = 'legacy_reference'
    compatibility_notes: list[str] = Field(default_factory=list)
    frame_index: int
    frame_time_s: float
    elapsed_s: float
    platform_position_ned_m: tuple[float, float, float]
    display_rows: int
    display_cols: int
    range_start_m: float
    range_step_m: float
    doppler_start_hz: float
    doppler_step_hz: float
    velocity_start_mps: float
    velocity_step_mps: float
    range_profile_db: list[float]
    doppler_profile_db: list[float]
    validations: list[TargetValidation]
    analytic_checks_use_truth: bool = True
    blind_detections: list[BlindDetection] = Field(default_factory=list)
    target_positions_ned_m: dict[str, tuple[float,float,float]] = Field(default_factory=dict)
    boresight_ned: tuple[float,float,float] = (0.0,0.0,1.0)


class SimulationCreated(BaseModel):
    job_id: str
    status: str
    derived_metrics: DerivedMetrics
    warnings: list[str]


class JobPublic(BaseModel):
    job_id: str
    status: Literal["queued", "running", "complete", "failed", "cancelled"]
    progress: float
    error: str | None = None
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    derived_metrics: DerivedMetrics
    warnings: list[str]
    frames: list[FrameMetadata]


def default_targets() -> list[TargetConfig]:
    return [
        TargetConfig(id="T1", position_ned_m=(582.0, 0.0, 0.0), amplitude=0.7),
        TargetConfig(id="T2", position_ned_m=(882.0, 0.0, 0.0), amplitude=1.0),
        TargetConfig(id="T3", position_ned_m=(1182.0, 0.0, 0.0), amplitude=0.8),
    ]


def default_request() -> SimulationRequest:
    return SimulationRequest(scene=SceneConfig(targets=default_targets()))


def clean_demo_request() -> SimulationRequest:
    request = default_request().model_copy(deep=True)
    request.platform.speed_mps = 90.0
    return request
