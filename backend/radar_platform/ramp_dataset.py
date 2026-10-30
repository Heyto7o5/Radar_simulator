from __future__ import annotations

import csv
import json
import math
import os
import re
import uuid
from collections import Counter
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks

from .models import C0


RAMP_SOURCE_URL = "https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection"
RAMP_GOOGLE_DRIVE_ID = "1OfqXXgoFg6xRYZkRqPJye4cQ29Fomh3l"


@dataclass(frozen=True)
class RampRadarConfig:
    carrier_hz: float = 77.0e9
    bandwidth_hz: float = 0.67e9
    chirp_duration_s: float = 60.0e-6
    slope_hz_s: float = 21.0e12
    num_samples: int = 128
    sample_rate_hz: float = 4.0e6
    num_chirps: int = 255
    num_rx: int = 4
    num_tx: int = 2
    frame_period_s: float = 33.33333e-3

    @property
    def wavelength_m(self) -> float:
        return C0 / self.carrier_hz

    @property
    def same_tx_chirp_period_s(self) -> float:
        return self.chirp_duration_s * self.num_tx


RAMP_CONFIG = RampRadarConfig()


def data_root() -> Path | None:
    value = os.environ.get("RADAR_DATA_ROOT")
    return Path(value).expanduser().resolve() if value else None


def dataset_root() -> Path | None:
    root = data_root()
    return root / "ramp_adc_77ghz" if root else None


def _find_files(root: Path, suffix: str, limit: int = 200) -> list[Path]:
    result: list[Path] = []
    if not root.exists():
        return result
    for path in root.rglob(f"*{suffix}"):
        if path.is_file():
            result.append(path)
            if len(result) >= limit:
                break
    return sorted(result)


def _find_radar_frames(root: Path, limit: int = 200) -> list[Path]:
    if not root.exists():
        return []
    preferred: list[Path] = []
    for directory in root.rglob("radar_raw_frame"):
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.mat"):
            if path.is_file():
                preferred.append(path)
                if len(preferred) >= limit:
                    return sorted(preferred)
    if preferred:
        return sorted(preferred)
    fallback: list[Path] = []
    for path in root.rglob("*.mat"):
        if path.is_file():
            fallback.append(path)
            if len(fallback) >= limit:
                break
    return sorted(fallback)


def dataset_status() -> dict[str, Any]:
    root = dataset_root()
    if root is None:
        return {
            "configured": False,
            "root": None,
            "is_real_available": False,
            "real_frame_count_preview": 0,
            "manifest": None,
            "source_url": RAMP_SOURCE_URL,
        }
    manifest_path = root / "manifest.json"
    manifest = None
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {"error": "manifest is unreadable"}
    frames = _find_radar_frames(root / "extracted", limit=201)
    return {
        "configured": True,
        "root": str(root),
        "is_real_available": bool(frames),
        "real_frame_count_preview": min(len(frames), 200),
        "count_is_capped": len(frames) > 200,
        "manifest": manifest,
        "source_url": RAMP_SOURCE_URL,
    }


def list_frames(limit: int = 50) -> list[dict[str, str]]:
    root = dataset_root()
    if root is None:
        return []
    frames = _find_radar_frames(root / "extracted", limit=limit)
    return [
        {"id": str(path.relative_to(root / "extracted")), "name": path.stem}
        for path in frames
    ]


def list_sequences() -> list[str]:
    root = dataset_root()
    if root is None:
        return []
    frames = _find_radar_frames(root / "extracted", limit=500)
    return sorted({str(path.parent.relative_to(root / "extracted")) for path in frames})


def _load_mat_arrays(path: Path) -> dict[str, np.ndarray]:
    try:
        return {key: value for key, value in loadmat(path).items() if not key.startswith("__")}
    except (NotImplementedError, ValueError):
        arrays: dict[str, np.ndarray] = {}
        with h5py.File(path, "r") as handle:
            def visitor(name: str, value: Any) -> None:
                if isinstance(value, h5py.Dataset):
                    arrays[name] = np.asarray(value)
            handle.visititems(visitor)
        return arrays


def normalize_adc_axes(array: np.ndarray) -> np.ndarray:
    expected = (128, 255, 4, 2)
    if array.ndim != 4:
        raise ValueError(f"ADC array must be four-dimensional, got {array.shape}")
    if sorted(array.shape) != sorted(expected):
        raise ValueError(f"ADC dimensions must be a permutation of {expected}, got {array.shape}")
    axes = tuple(array.shape.index(size) for size in expected)
    if len(set(axes)) != 4:
        raise ValueError(f"ADC axes cannot be uniquely identified from {array.shape}")
    normalized = np.transpose(array, axes)
    if not np.iscomplexobj(normalized):raise ValueError('RAMP ADC must contain complex I/Q; real IF requires a separate adapter')
    adc = normalized.astype(np.complex64, copy=False)
    finite_fraction = float(np.mean(np.isfinite(adc.real) & np.isfinite(adc.imag)))
    if finite_fraction < 1.0:
        raise ValueError("ADC contains non-finite samples; FFT contamination must not be silently accepted")
    return np.ascontiguousarray(adc)


def load_real_frame(frame_id: str) -> tuple[np.ndarray, Path]:
    root = dataset_root()
    if root is None:
        raise FileNotFoundError("RADAR_DATA_ROOT is not configured")
    base = (root / "extracted").resolve()
    path = (base / frame_id).resolve()
    if base not in path.parents or path.suffix.lower() != ".mat" or not path.exists():
        raise FileNotFoundError(frame_id)
    arrays = _load_mat_arrays(path)
    candidates: list[tuple[str, np.ndarray]] = []
    for key, value in arrays.items():
        if isinstance(value, np.ndarray) and value.ndim == 4:
            candidates.append((key, value))
    exact = [(key, value) for key, value in candidates if sorted(value.shape) == [2, 4, 128, 255]]
    if len(exact) != 1:
        shapes = {key: value.shape for key, value in candidates}
        raise ValueError(f"expected exactly one ADC array; candidates={shapes}")
    return normalize_adc_axes(exact[0][1]), path


def _companion(path: Path, directory_name: str, suffixes: tuple[str, ...]) -> Path | None:
    parts = list(path.parts)
    for radar_name in ("radar_raw_frame", "radar"):
        if radar_name in parts:
            index = parts.index(radar_name)
            parts[index] = directory_name
            candidate_base = Path(*parts).with_suffix("")
            bases = [candidate_base]
            if candidate_base.stem.isdigit():
                bases.append(candidate_base.with_name(f"{int(candidate_base.stem):010d}"))
            for base in bases:
                for suffix in suffixes:
                    candidate = base.with_suffix(suffix)
                    if candidate.exists():
                        return candidate
    for suffix in suffixes:
        candidate = path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _read_labels(frame_path: Path) -> list[dict[str, Any]]:
    path = _companion(frame_path, "text_labels", (".csv", ".txt"))
    if path is None:
        return []
    labels: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    for row in rows:
        if len(row) < 6:
            continue
        try:
            uid = row[0].strip()
            try:
                numeric_uid=Decimal(uid)
                if numeric_uid.is_finite():uid=format(numeric_uid.normalize(),'f')
            except InvalidOperation:
                pass
            class_id = int(float(row[1]))
            x_m, y_m, width_m, length_m = (float(value) for value in row[2:6])
        except ValueError:
            continue
        labels.append(
            {
                "uid": uid,
                "class_id": class_id,
                "x_m": x_m,
                "y_m": y_m,
                "width_m": width_m,
                "length_m": length_m,
                "range_m": math.hypot(x_m, y_m),
                "radial_velocity_mps": None,
            }
        )
    return labels


def _neighbor_frame(frame_path: Path, offset: int) -> Path | None:
    match = re.fullmatch(r"(.*?)(\d+)", frame_path.stem)
    if match is None:
        return None
    prefix, digits = match.groups()
    number = int(digits) + offset
    if number < 0:
        return None
    candidate = frame_path.with_name(f"{prefix}{number:0{len(digits)}d}{frame_path.suffix}")
    return candidate if candidate.exists() else None


def load_labels(frame_path: Path) -> list[dict[str, Any]]:
    labels = _read_labels(frame_path)
    if not labels:
        return labels
    previous_path = _neighbor_frame(frame_path, -1)
    next_path = _neighbor_frame(frame_path, 1)
    previous_labels=_read_labels(previous_path) if previous_path else []
    following_labels=_read_labels(next_path) if next_path else []
    current_counts=Counter(label['uid'] for label in labels)
    previous_counts=Counter(label['uid'] for label in previous_labels)
    following_counts=Counter(label['uid'] for label in following_labels)
    previous = {label['uid']:label for label in previous_labels if previous_counts[label['uid']]==1}
    following = {label['uid']:label for label in following_labels if following_counts[label['uid']]==1}
    dt = RAMP_CONFIG.frame_period_s
    for ordinal,label in enumerate(labels):
        uid = label["uid"]
        label['instance_key']=f'{uid}:{ordinal}'
        label['tracking_ambiguous']=current_counts[uid]>1
        label['radial_velocity_source']='unavailable'
        label['velocity_is_surveyed_truth']=False
        if label['tracking_ambiguous']:continue
        before=uid in previous and previous[uid]['class_id']==label['class_id']
        after=uid in following and following[uid]['class_id']==label['class_id']
        if before and after:
            label["radial_velocity_mps"] = (
                previous[uid]["range_m"] - following[uid]["range_m"]
            ) / (2.0 * dt)
            label['radial_velocity_source']='central_difference_of_annotation_range'
        elif after:
            label["radial_velocity_mps"] = (
                label["range_m"] - following[uid]["range_m"]
            ) / dt
            label['radial_velocity_source']='forward_difference_of_annotation_range'
        elif before:
            label["radial_velocity_mps"] = (
                previous[uid]["range_m"] - label["range_m"]
            ) / dt
            label['radial_velocity_source']='backward_difference_of_annotation_range'
    return labels


def camera_path(frame_path: Path) -> Path | None:
    return _companion(frame_path, "images_0", (".jpg", ".jpeg", ".png"))


def synthesize_adc(labels: list[dict[str, Any]] | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    config = RAMP_CONFIG
    if labels is None:
        labels = [
            {"uid": "S1", "class_id": 2, "x_m": -2.0, "y_m": 7.0, "width_m": 1.8, "length_m": 4.2, "range_m": math.hypot(-2.0, 7.0), "radial_velocity_mps": 1.2},
            {"uid": "S2", "class_id": 0, "x_m": 1.5, "y_m": 11.5, "width_m": 0.7, "length_m": 0.7, "range_m": math.hypot(1.5, 11.5), "radial_velocity_mps": -2.4},
            {"uid": "S3", "class_id": 5, "x_m": 3.0, "y_m": 17.0, "width_m": 2.5, "length_m": 8.0, "range_m": math.hypot(3.0, 17.0), "radial_velocity_mps": 0.4},
        ]
    # None requests the demo; empty real annotations must not invent targets.
    from .ramp_alignment import simulate_track_adc
    scene=[]
    for label in labels:
        r=math.hypot(label['x_m'],label['y_m']);v=float(label.get('radial_velocity_mps') or 0.)
        if r<=0:raise ValueError('target intersects radar')
        scene.append({**label,'range_m':r,'velocity_ned_mps':[-v*label['y_m']/r,-v*label['x_m']/r,0.]})
    adc=simulate_track_adc(scene)
    return adc, labels


def process_adc(adc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config = RAMP_CONFIG
    adc = normalize_adc_axes(adc)
    centered = adc - np.mean(adc, axis=(0, 1), keepdims=True)
    range_window = np.hanning(config.num_samples).astype(np.float32)[:, None, None, None]
    doppler_window = np.hanning(config.num_chirps).astype(np.float32)[None, :, None, None]
    range_fft = np.fft.fft(centered * range_window, axis=0)
    rd_channels = np.fft.fftshift(
        np.fft.fft(range_fft * doppler_window, axis=1),
        axes=1,
    ).astype(np.complex64)
    raw_doppler=np.fft.fftshift(np.fft.fftfreq(config.num_chirps,d=config.same_tx_chirp_period_s))
    closing_doppler=-raw_doppler
    order=np.argsort(closing_doppler)
    rd_channels=rd_channels[:,order]
    power_channels = np.abs(rd_channels) ** 2
    combined_power = np.mean(power_channels, axis=(2, 3))
    combined_db = 10.0 * np.log10(combined_power / max(float(np.max(combined_power)), 1.0e-30) + 1.0e-12)
    # Keep public channel order consistent with tx0_rx0 ... tx1_rx3.
    channel_power = np.transpose(power_channels, (3, 2, 1, 0)).reshape(8, config.num_chirps, config.num_samples)
    channel_peak = np.maximum(np.max(channel_power, axis=(1, 2), keepdims=True), 1.0e-30)
    channel_db = 10.0 * np.log10(channel_power / channel_peak + 1.0e-12)
    range_axis = np.arange(config.num_samples) * config.sample_rate_hz / config.num_samples * C0 / (2.0 * config.slope_hz_s)
    velocity_carrier=config.carrier_hz+config.slope_hz_s*config.num_samples/config.sample_rate_hz/2
    velocity_axis = closing_doppler[order] * C0/velocity_carrier / 2.0
    return (
        np.ascontiguousarray(combined_db.T, dtype="<f4"),
        np.ascontiguousarray(channel_db, dtype="<f4"),
        range_axis.astype(np.float64),
        velocity_axis.astype(np.float64),
    )


def _label_peak_metrics(
    combined_db: np.ndarray,
    range_axis: np.ndarray,
    labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    profile = np.max(combined_db, axis=0)
    peaks, _ = find_peaks(profile, distance=2)
    if peaks.size:
        strongest = peaks[np.argsort(profile[peaks])[-20:]]
    else:
        strongest = np.array([], dtype=int)
    results: list[dict[str, Any]] = []
    for label in labels:
        label_range = float(label["range_m"])
        if strongest.size:
            index = int(strongest[np.argmin(np.abs(range_axis[strongest] - label_range))])
            detected = float(range_axis[index])
            error = abs(detected - label_range)
        else:
            detected = None
            error = None
        results.append(
            {
                **label,
                "nearest_strong_peak_range_m": detected,
                "range_difference_m": error,
                "metric_scope": "exploratory; labels do not identify a unique radar spectral peak",
            }
        )
    return results


@dataclass
class ValidationResult:
    result_id: str
    is_real: bool
    source_label: str
    measured_path: Path
    ideal_path: Path
    channels_path: Path
    camera: Path | None
    metadata: dict[str, Any]


_results: dict[str, ValidationResult] = {}


def process_frame(frame_id: str | None = None) -> ValidationResult:
    is_real = False
    frame_path: Path | None = None
    if frame_id:
        try:
            adc, frame_path = load_real_frame(frame_id)
            labels = load_labels(frame_path)
            is_real = True
        except FileNotFoundError:
            adc, labels = synthesize_adc()
    else:
        frames = list_frames(limit=1)
        if frames:
            adc, frame_path = load_real_frame(frames[0]["id"])
            labels = load_labels(frame_path)
            is_real = True
            frame_id = frames[0]["id"]
        else:
            adc, labels = synthesize_adc()

    measured_db, channels_db, range_axis, velocity_axis = process_adc(adc)
    ideal_adc, _ = synthesize_adc(labels)
    ideal_db, _, _, _ = process_adc(ideal_adc)
    result_id = uuid.uuid4().hex
    root = dataset_root()
    if root is None:
        output_dir = Path(os.environ.get("RADAR_RUNTIME_ROOT", ".runtime")).resolve() / "validation" / result_id
    else:
        output_dir = root / "processed" / result_id
    output_dir.mkdir(parents=True, exist_ok=True)
    measured_path = output_dir / "measured.npy"
    ideal_path = output_dir / "ideal.npy"
    channels_path = output_dir / "channels.npy"
    np.save(measured_path, measured_db, allow_pickle=False)
    np.save(ideal_path, ideal_db, allow_pickle=False)
    np.save(channels_path, channels_db, allow_pickle=False)
    label_metrics = _label_peak_metrics(measured_db, range_axis, labels)
    metadata = {
        "result_id": result_id,
        "is_real": is_real,
        "source_label": "真实公开 ADC" if is_real else "合成回退数据",
        "frame_id": frame_id,
        "adc_shape": list(adc.shape),
        "rd_shape": list(measured_db.shape),
        "range_start_m": float(range_axis[0]),
        "range_step_m": float(range_axis[1] - range_axis[0]),
        "velocity_start_mps": float(velocity_axis[0]),
        "velocity_step_mps": float(velocity_axis[1] - velocity_axis[0]),
        "labels": label_metrics,
        "channel_names": [f"tx{tx}_rx{rx}" for tx in range(2) for rx in range(4)],
        "radar_config": RAMP_CONFIG.__dict__,
        "velocity_convention":"positive closing; tx*conj(rx) up-chirp convention",
        "velocity_carrier_hz":RAMP_CONFIG.carrier_hz+RAMP_CONFIG.slope_hz_s*RAMP_CONFIG.num_samples/RAMP_CONFIG.sample_rate_hz/2,
        "ideal_scene_available":bool(labels),
        "ideal_velocity_model":"radial-only approximation from annotations; unavailable speeds use explicit zero assumption",
        "display_normalization":"per-image peak normalized for display only; not quantitative alignment MAE",
        "quantitative_alignment_scope":"use common-reference batch RD/RAD validation; this display endpoint is not a calibrated RAD benchmark",
        "source_url": RAMP_SOURCE_URL,
        "validation_scope": [
            "验证 ADC 维度、Range/Doppler 处理链和物理坐标轴。",
            "标签距离与强峰差异仅为探索性指标，不代表唯一目标关联。",
            "不验证绝对幅度、材料散射、微多普勒或高空场景保真度。",
        ],
    }
    result = ValidationResult(
        result_id=result_id,
        is_real=is_real,
        source_label=metadata["source_label"],
        measured_path=measured_path,
        ideal_path=ideal_path,
        channels_path=channels_path,
        camera=camera_path(frame_path) if frame_path else None,
        metadata=metadata,
    )
    _results[result_id] = result
    return result


def get_result(result_id: str) -> ValidationResult:
    try:
        return _results[result_id]
    except KeyError as exc:
        raise KeyError(result_id) from exc


def result_array(result_id: str, view: str, channel: str = "combined") -> np.ndarray:
    result = get_result(result_id)
    if view == "ideal":
        return np.load(result.ideal_path, mmap_mode="r", allow_pickle=False)
    if channel == "combined":
        return np.load(result.measured_path, mmap_mode="r", allow_pickle=False)
    if not channel.startswith("tx") or "_rx" not in channel:
        raise ValueError(channel)
    try:
        tx_text, rx_text = channel.split("_rx")
        tx = int(tx_text[2:])
        rx = int(rx_text)
    except ValueError as exc:
        raise ValueError(channel) from exc
    if tx not in (0, 1) or rx not in (0, 1, 2, 3):
        raise ValueError(channel)
    channels = np.load(result.channels_path, mmap_mode="r", allow_pickle=False)
    return channels[tx * 4 + rx]
