#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ROOT = Path("/nas/datasets/hyt/datasets/radar_open").resolve()
DATASET_DIR = "ramp_adc_77ghz"
GOOGLE_DRIVE_ID = "1OfqXXgoFg6xRYZkRqPJye4cQ29Fomh3l"
SOURCE_URL = "https://github.com/Xiangyu-Gao/Raw_ADC_radar_dataset_for_automotive_object_detection"


def root_from_env() -> Path:
    return Path(os.environ.get("RADAR_DATA_ROOT", str(DEFAULT_ROOT))).expanduser().resolve()


def safe_root() -> Path:
    root = root_from_env()
    project = Path(__file__).resolve().parents[1]
    if root == project or project in root.parents or root in project.parents:
        raise SystemExit("Refusing to place external radar data inside or above the source project")
    if not str(root).startswith("/nas/"):
        raise SystemExit("RADAR_DATA_ROOT must be an explicit /nas path for this project")
    return root


def paths() -> dict[str, Path]:
    base = safe_root() / DATASET_DIR
    return {
        "base": base,
        "raw": base / "raw",
        "extracted": base / "extracted",
        "processed": base / "processed",
        "manifest": base / "manifest.json",
    }


def ensure_layout() -> dict[str, Path]:
    result = paths()
    for key in ("raw", "extracted", "processed"):
        result[key].mkdir(parents=True, exist_ok=True)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"error": "manifest is invalid"}


def save_manifest(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def command_status(_: argparse.Namespace) -> None:
    p = paths()
    manifest = load_manifest(p["manifest"])
    raw_files = [path for path in p["raw"].glob("*") if path.is_file()] if p["raw"].exists() else []
    extracted_sample = next(p["extracted"].rglob("*.mat"), None) if p["extracted"].exists() else None
    print(json.dumps({
        "root": str(p["base"]),
        "raw_files": [{"name": path.name, "bytes": path.stat().st_size} for path in raw_files],
        "has_extracted_mat": extracted_sample is not None,
        "extracted_sample": str(extracted_sample) if extracted_sample else None,
        "manifest": manifest,
    }, ensure_ascii=False, indent=2))


def command_download(_: argparse.Namespace) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise SystemExit("Install project dependencies first: pip install -e '.[dev]'") from exc
    p = ensure_layout()
    output_directory = str(p["raw"]) + os.sep
    print(f"Downloading official archive to {p['raw']}")
    downloaded = gdown.download(
        id=GOOGLE_DRIVE_ID,
        output=output_directory,
        quiet=False,
        resume=True,
    )
    if not downloaded:
        raise SystemExit("Download did not complete; rerun the command to resume")
    path = Path(downloaded).resolve()
    manifest = load_manifest(p["manifest"])
    manifest.update({
        "dataset": "Raw ADC Data of 77GHz MMWave Radar for Automotive Object Detection",
        "source_url": SOURCE_URL,
        "google_drive_id": GOOGLE_DRIVE_ID,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "archive": {"path": str(path), "bytes": path.stat().st_size},
        "license_note": "Dataset is stored locally for research validation; consult the official source for dataset terms.",
    })
    save_manifest(p["manifest"], manifest)
    print(f"Downloaded: {path}")


def command_verify(_: argparse.Namespace) -> None:
    p = paths()
    files = [path for path in p["raw"].glob("*") if path.is_file()] if p["raw"].exists() else []
    if not files:
        raise SystemExit("No downloaded file found")
    manifest = load_manifest(p["manifest"])
    verified = []
    for path in files:
        print(f"Hashing {path.name} ...")
        verified.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest["verified_at"] = datetime.now(timezone.utc).isoformat()
    manifest["verified_files"] = verified
    save_manifest(p["manifest"], manifest)
    print(json.dumps(verified, indent=2))


def command_extract(_: argparse.Namespace) -> None:
    p = ensure_layout()
    archives = [path for path in p["raw"].glob("*") if path.is_file()]
    if not archives:
        raise SystemExit("No downloaded archive found")
    archive = max(archives, key=lambda path: path.stat().st_size)
    try:
        shutil.unpack_archive(str(archive), str(p["extracted"]))
    except (shutil.ReadError, ValueError) as exc:
        raise SystemExit(f"Unsupported archive format for {archive.name}: {exc}") from exc
    manifest = load_manifest(p["manifest"])
    manifest["extracted_at"] = datetime.now(timezone.utc).isoformat()
    manifest["extracted_root"] = str(p["extracted"])
    save_manifest(p["manifest"], manifest)
    print(f"Extracted to {p['extracted']}")


def command_prepare_demo(args: argparse.Namespace) -> None:
    p = ensure_layout()
    count = int(getattr(args, "count", 3))
    archives = [path for path in p["raw"].glob("*.zip") if path.is_file()]
    radar_frames = sorted(
        path for path in p["extracted"].rglob("*.mat")
        if "radar_raw_frame" in {part.lower() for part in path.parts}
    )[:count]
    if not archives and not radar_frames:
        raise SystemExit("No MAT frame or ZIP archive found; download the dataset first")
    if archives:
        archive = max(archives, key=lambda path: path.stat().st_size)
        with zipfile.ZipFile(archive) as handle:
            names = handle.namelist()
            name_set = set(names)
            if len(radar_frames) >= count:
                radar_names = [str(path.relative_to(p["extracted"])) for path in radar_frames]
            else:
                radar_names = sorted(
                    name for name in names
                    if name.lower().endswith(".mat") and "radar_raw_frame" in name.lower()
                )[:count]
            if not radar_names:
                radar_names = sorted(name for name in names if name.lower().endswith(".mat"))[:count]
            if not radar_names:
                raise SystemExit("The archive contains no MAT radar frames")
            selected_names: set[str] = set(radar_names)
            for radar_name in radar_names:
                radar_path = Path(radar_name)
                numeric_stem = f"{int(radar_path.stem):010d}" if radar_path.stem.isdigit() else radar_path.stem
                for directory in ("images_0", "text_labels"):
                    companion_stem = str(
                        radar_path.with_name(numeric_stem).with_suffix("")
                    ).replace("radar_raw_frame", directory)
                    for suffix in (".jpg", ".jpeg", ".png", ".csv", ".txt"):
                        candidate = companion_stem + suffix
                        if candidate in name_set:
                            selected_names.add(candidate)
            extraction_root = p["extracted"].resolve()
            for name in sorted(selected_names):
                destination = (extraction_root / name).resolve()
                if extraction_root not in destination.parents:
                    raise SystemExit(f"Unsafe archive member: {name}")
                if destination.exists():
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with handle.open(name) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
        radar_frames = sorted(
            path for path in p["extracted"].rglob("*.mat")
            if "radar_raw_frame" in {part.lower() for part in path.parts}
        )[:count]
    selected = radar_frames[:count]
    demo_dir = p["processed"] / "demo_sources"
    demo_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for path in selected:
        index.append({"frame": str(path.relative_to(p["extracted"])), "bytes": path.stat().st_size})
    (demo_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    manifest = load_manifest(p["manifest"])
    manifest["demo_prepared_at"] = datetime.now(timezone.utc).isoformat()
    manifest["demo_frames"] = index
    save_manifest(p["manifest"], manifest)
    print(f"Prepared demo index with {len(index)} frames at {demo_dir / 'index.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Manage external radar datasets under /nas")
    subparsers = parser.add_subparsers(dest="command", required=True)
    handlers = {
        "status": command_status,
        "download": command_download,
        "verify": command_verify,
        "extract": command_extract,
        "prepare-demo": command_prepare_demo,
    }
    for command, handler in handlers.items():
        subparser = subparsers.add_parser(command)
        subparser.add_argument("dataset", choices=["ramp_adc"])
        if command == "prepare-demo":
            subparser.add_argument("--count", type=int, default=3, choices=range(1, 201))
        subparser.set_defaults(handler=handler)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
