from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from radar_platform.ramp_dataset import (
    dataset_status,
    load_labels,
    normalize_adc_axes,
    process_adc,
    process_frame,
    synthesize_adc,
)


def test_normalize_adc_axis_permutation() -> None:
    original = np.zeros((128, 255, 4, 2), dtype=np.complex64)
    permuted = np.transpose(original, (1, 3, 0, 2))
    normalized = normalize_adc_axes(permuted)
    assert normalized.shape == (128, 255, 4, 2)
    assert normalized.dtype == np.complex64


def test_normalize_adc_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match="permutation"):
        normalize_adc_axes(np.zeros((128, 256, 4, 2)))


def test_synthetic_fallback_processing() -> None:
    adc, labels = synthesize_adc()
    combined, channels, range_axis, velocity_axis = process_adc(adc)
    assert adc.shape == (128, 255, 4, 2)
    assert combined.shape == (255, 128)
    assert channels.shape == (8, 255, 128)
    assert len(labels) == 3
    assert np.all(np.diff(range_axis) > 0)
    assert np.all(np.diff(velocity_axis) > 0)


def test_missing_dataset_uses_explicit_fallback(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("RADAR_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("RADAR_RUNTIME_ROOT", str(tmp_path / "runtime"))
    result = process_frame()
    assert result.is_real is False
    assert result.metadata["source_label"] == "合成回退数据"
    assert result.metadata["is_real"] is False


def test_dataset_status_without_configuration(monkeypatch) -> None:
    monkeypatch.delenv("RADAR_DATA_ROOT", raising=False)
    status = dataset_status()
    assert status["configured"] is False
    assert status["is_real_available"] is False


def test_neighbor_labels_produce_closing_radial_velocity(tmp_path: Path) -> None:
    radar_dir = tmp_path / "sequence" / "radar_raw_frame"
    label_dir = tmp_path / "sequence" / "text_labels"
    radar_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    for index, distance in enumerate((10.0, 9.0, 8.0)):
        savemat(radar_dir / f"frame_{index:04d}.mat", {"dummy": np.array([1])})
        (label_dir / f"frame_{index:04d}.csv").write_text(
            f"track-1,2,0,{distance},1.8,4.2\n",
            encoding="utf-8",
        )
    labels = load_labels(radar_dir / "frame_0001.mat")
    assert len(labels) == 1
    assert labels[0]["radial_velocity_mps"] == pytest.approx(30.0, rel=1e-5)
