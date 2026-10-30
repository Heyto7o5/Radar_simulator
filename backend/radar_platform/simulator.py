from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .models import C0, FrameMetadata, SimulationRequest, TargetValidation
from .physics import (
    alias_doppler,
    angle_between,
    derived_metrics,
    platform_velocity_ned,
    platform_boresight_ned,
    target_geometry,
)
from .kinematics import position_at
from .detection import detect_rd_peaks


@dataclass(frozen=True)
class Scatterer:
    id: str
    position_ned_m: np.ndarray
    amplitude: float
    phase_rad: float = 0.0
    is_validation_target: bool = True
    velocity_ned_mps: np.ndarray = field(default_factory=lambda: np.zeros(3))


@dataclass
class FrameResult:
    rd_complex: np.ndarray
    display_db: np.ndarray
    metadata: FrameMetadata


def _clutter_scatterers(request: SimulationRequest, frame_position: np.ndarray) -> list[Scatterer]:
    if not request.scene.simplified_clutter_enabled:
        return []
    radar = request.radar
    boresight = platform_boresight_ned(request.platform)
    # One immutable world patch per request. Only its illumination changes.
    initial_position=np.asarray(request.platform.position_ned_m,dtype=float)
    if boresight[2] <= 0.0 or initial_position[2] >= 0.0:
        return []
    t_ground = -initial_position[2] / boresight[2]
    center = initial_position[:2] + t_ground * boresight[:2]
    radius = t_ground * math.tan(math.radians(radar.beam_half_angle_deg))
    radius = min(radius, 750.0)
    spacing = 50.0
    coordinates = np.arange(-radius, radius + spacing, spacing)
    rng = np.random.default_rng(request.scene.seed)
    scatterers: list[Scatterer] = []
    index = 0
    for dx in coordinates:
        for dy in coordinates:
            if dx * dx + dy * dy > radius * radius:
                continue
            phase = float(rng.uniform(-math.pi, math.pi))
            scatterers.append(
                Scatterer(
                    id=f"C{index}",
                    position_ned_m=np.array([center[0] + dx, center[1] + dy, 0.0]),
                    amplitude=0.012,
                    phase_rad=phase,
                    is_validation_target=False,
                )
            )
            index += 1
    return scatterers


def _scene_scatterers(request: SimulationRequest, frame_position: np.ndarray) -> list[Scatterer]:
    targets = [
        Scatterer(
            id=target.id,
            position_ned_m=np.asarray(target.position_ned_m, dtype=np.float64),
            amplitude=target.amplitude,
            velocity_ned_mps=np.asarray(target.velocity_ned_mps,dtype=float),
        )
        for target in request.scene.targets
    ]
    return targets + _clutter_scatterers(request, frame_position)


def _transmit_chirp(sample_times: np.ndarray, pulse_width_s: float, bandwidth_hz: float) -> np.ndarray:
    centered = sample_times - pulse_width_s / 2.0
    slope = bandwidth_hz / pulse_width_s
    return np.exp(1j * math.pi * slope * centered * centered)


def _local_peak(
    power: np.ndarray,
    expected_doppler_bin: int,
    expected_range_bin: int,
    range_radius: int,
    doppler_radius: int = 2,
) -> tuple[int, int]:
    rows, cols = power.shape
    d0 = max(0, expected_doppler_bin - doppler_radius)
    d1 = min(rows, expected_doppler_bin + doppler_radius + 1)
    r0 = max(0, expected_range_bin - range_radius)
    r1 = min(cols, expected_range_bin + range_radius + 1)
    if d0 >= d1 or r0 >= r1:
        return expected_doppler_bin, expected_range_bin
    local = power[d0:d1, r0:r1]
    local_d, local_r = np.unravel_index(int(np.argmax(local)), local.shape)
    return d0 + int(local_d), r0 + int(local_r)


def simulate_frame(request: SimulationRequest, frame_index: int, *, session=None, cancelled=None) -> FrameResult:
    started = time.perf_counter()
    radar = request.radar
    platform = request.platform
    metrics = derived_metrics(radar)
    velocity = platform_velocity_ned(platform)
    boresight = platform_boresight_ned(platform)
    initial_position = np.asarray(platform.position_ned_m, dtype=np.float64)
    frame_time = frame_index * request.sequence.frame_interval_s
    frame_position = initial_position + velocity * frame_time

    pri = metrics.pri_s
    num_pulses = radar.pulses_per_cpi
    num_samples = metrics.num_fast_time_samples
    # Legacy API compatibility: synthesize through the shared session, then
    # adapt layout/scaling for the original response contract. No echo model
    # lives in this compatibility layer.
    from .unified.legacy_config import convert_legacy
    from .unified.session import SimulationSession
    common_request, migration_notes = convert_legacy(request)
    if session is None:
        session=SimulationSession(common_request)
    elif session.request != common_request:
        raise ValueError('compatibility session belongs to another configuration')
    common_frame = session.frame(frame_index,cancelled=cancelled)
    num_samples = common_request.waveform.samples
    chirp_slope = radar.bandwidth_hz / radar.pulse_width_s
    beam_half_angle = math.radians(radar.beam_half_angle_deg)
    tx_count = int(math.ceil(radar.sample_rate_hz * radar.pulse_width_s))
    # Public legacy RD units used unnormalized range and Doppler transforms.
    scale = tx_count * np.hanning(num_pulses).sum()
    rd_complex = np.ascontiguousarray(common_frame.products.rd_channels[:, :, 0, 0].T * scale,
                                      dtype=np.complex64)
    power = np.abs(rd_complex) ** 2
    peak_power = float(np.max(power))
    rd_db = 10.0 * np.log10(power / max(peak_power, 1.0e-30) + 1.0e-12)

    range_axis = np.arange(num_samples, dtype=np.float64) * metrics.range_sample_spacing_m
    doppler_axis = np.fft.fftshift(np.fft.fftfreq(num_pulses, d=pri))
    velocity_axis = doppler_axis * metrics.wavelength_m / 2.0
    mid_time = frame_time + metrics.cpi_s / 2.0
    mid_position = initial_position + velocity * mid_time
    validations: list[TargetValidation] = []
    expected_ranges: list[float] = []

    validation_target_map = {target.id: target for target in request.scene.targets}
    for target_id, target in validation_target_map.items():
        target_position = position_at(target.position_ned_m,target.velocity_ned_mps,mid_time)
        distance, los, radial_velocity = target_geometry(mid_position, velocity, target_position,target.velocity_ned_mps)
        in_beam = angle_between(boresight, los) <= beam_half_angle
        delay = 2.0 * distance / C0
        visible = in_beam and delay + radar.pulse_width_s <= pri
        doppler = 2.0 * radial_velocity / metrics.wavelength_m
        aliased, order = alias_doppler(doppler, radar.prf_hz)
        # An LFM matched filter has a predictable range-Doppler coupling. For
        # an up-chirp and positive closing Doppler, the raw peak moves earlier.
        range_doppler_bias = C0 * doppler / (2.0 * chirp_slope)
        raw_expected_range = distance - range_doppler_bias
        expected_range_bin = int(round(raw_expected_range / metrics.range_sample_spacing_m))
        expected_doppler_bin = int(np.argmin(np.abs(doppler_axis - aliased)))
        validation = TargetValidation(
            target_id=target_id,
            visible=visible,
            in_beam=in_beam,
            true_range_m=distance,
            range_doppler_bias_m=range_doppler_bias,
            true_radial_velocity_mps=radial_velocity,
            true_doppler_hz=doppler,
            aliased_doppler_hz=aliased,
            ambiguity_order=order,
            apparent_velocity_mps=aliased * metrics.wavelength_m / 2.0,
            expected_range_bin=expected_range_bin,
            expected_doppler_bin=expected_doppler_bin,
        )
        if visible and 0 <= expected_range_bin < num_samples:
            range_radius = max(2, int(math.ceil(metrics.range_resolution_m / metrics.range_sample_spacing_m)) + 2)
            detected_doppler_bin, detected_range_bin = _local_peak(
                power,
                expected_doppler_bin,
                expected_range_bin,
                range_radius,
            )
            raw_detected_range = float(range_axis[detected_range_bin])
            detected_range = raw_detected_range + range_doppler_bias
            detected_velocity = float(velocity_axis[detected_doppler_bin])
            doppler_bin_error = float(abs(detected_doppler_bin - expected_doppler_bin))
            range_error = abs(detected_range - distance)
            validation.detected_range_m = detected_range
            validation.raw_detected_range_m = raw_detected_range
            validation.range_error_m = range_error
            validation.detected_apparent_velocity_mps = detected_velocity
            validation.doppler_bin_error = doppler_bin_error
            validation.detected_range_bin = detected_range_bin
            validation.detected_doppler_bin = detected_doppler_bin
            validation.passed = (
                range_error <= metrics.range_sample_spacing_m + 1.0e-6
                and doppler_bin_error <= 1.0
                and not request.scene.simplified_clutter_enabled
            )
        validations.append(validation)
        expected_ranges.append(distance)

    if expected_ranges:
        lower = max(0.0, min(expected_ranges) - 350.0)
        upper = min(metrics.max_unambiguous_range_m, max(expected_ranges) + 350.0)
    else:
        lower, upper = 0.0, metrics.max_unambiguous_range_m
    lower_bin = max(0, int(math.floor(lower / metrics.range_sample_spacing_m)))
    upper_bin = min(num_samples, int(math.ceil(upper / metrics.range_sample_spacing_m)) + 1)
    display_db = np.ascontiguousarray(rd_db[:, lower_bin:upper_bin], dtype="<f4")
    range_profile = np.max(display_db, axis=0)
    doppler_profile = np.max(display_db, axis=1)

    metadata = FrameMetadata(
        simulation_engine='unified',
        compatibility_notes=migration_notes,
        frame_index=frame_index,
        frame_time_s=frame_time,
        elapsed_s=time.perf_counter() - started,
        platform_position_ned_m=tuple(float(value) for value in frame_position),
        display_rows=display_db.shape[0],
        display_cols=display_db.shape[1],
        range_start_m=float(range_axis[lower_bin]),
        range_step_m=metrics.range_sample_spacing_m,
        doppler_start_hz=float(doppler_axis[0]),
        doppler_step_hz=float(radar.prf_hz / num_pulses),
        velocity_start_mps=float(velocity_axis[0]),
        velocity_step_mps=metrics.velocity_bin_mps,
        range_profile_db=[float(value) for value in range_profile],
        doppler_profile_db=[float(value) for value in doppler_profile],
        validations=validations,
        blind_detections=detect_rd_peaks(power,range_axis,doppler_axis,metrics.wavelength_m,chirp_slope,
            range_guard=max(3,int(math.ceil(2*metrics.range_resolution_m/metrics.range_sample_spacing_m)))),
        target_positions_ned_m={t.id:tuple(position_at(t.position_ned_m,t.velocity_ned_mps,mid_time)) for t in request.scene.targets},
        boresight_ned=tuple(boresight),
    )
    return FrameResult(rd_complex=rd_complex, display_db=display_db, metadata=metadata)


def save_frame(result: FrameResult, destination: Path) -> tuple[Path, Path]:
    destination.mkdir(parents=True, exist_ok=True)
    display_path = destination / f"frame_{result.metadata.frame_index:04d}_display.npy"
    complex_path = destination / f"frame_{result.metadata.frame_index:04d}_complex.npy"
    np.save(display_path, result.display_db, allow_pickle=False)
    np.save(complex_path, result.rd_complex, allow_pickle=False)
    return display_path, complex_path
