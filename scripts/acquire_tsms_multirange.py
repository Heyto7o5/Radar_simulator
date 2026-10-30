#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
import py7zr


ARTICLE_ID = 30027313
CORNER_FILE_IDS = [
    57684175,
    57685747,
    57687643,
    57687646,
    57690181,
    57690184,
    57690187,
    57700387,
    57700390,
    57700393,
    57700396,
    57700399,
    57718876,
    57718873,
    57718879,
]
ROOT = Path("/nas/datasets/hyt/datasets/radar_open/tsms_drone_sample")


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def distance_in_archive(path: Path) -> int:
    with py7zr.SevenZipFile(path, "r") as archive:
        for name in archive.getnames():
            match = re.search(r"Corner Reflector_(\d+)m_\d+\.mat$", name)
            if match:
                return int(match.group(1))
    raise ValueError(f"No corner-reflector distance found in {path}")


def extract_measurements(path: Path, distance: int, count: int = 60) -> Path:
    output = ROOT / "extracted" / f"corner_reflector_{distance}m"
    existing = list(output.rglob("*.mat")) if output.exists() else []
    if len(existing) >= count:
        return output
    output.mkdir(parents=True, exist_ok=True)
    targets = [
        f"Raw File/Corner Reflector_{distance}m_{index:03d}.mat"
        for index in range(1, count + 1)
    ]
    with py7zr.SevenZipFile(path, "r") as archive:
        archive.extract(path=output, targets=targets)
    return output


def download_range(file_id: int, file_descriptor: int, start: int, stop: int) -> tuple[int, int]:
    expected = stop - start + 1
    for attempt in range(1, 6):
        try:
            with httpx.Client(trust_env=False, follow_redirects=False, timeout=120.0) as client:
                redirect = client.get(f"https://ndownloader.figshare.com/files/{file_id}")
                if redirect.status_code not in {301, 302, 303, 307, 308}:
                    redirect.raise_for_status()
                location = redirect.headers["location"]
                response = client.get(
                    location,
                    headers={"Range": f"bytes={start}-{stop}"},
                )
                response.raise_for_status()
                if len(response.content) != expected:
                    raise IOError(
                        f"short range {start}-{stop}: {len(response.content)} != {expected}"
                    )
                os.pwrite(file_descriptor, response.content, start)
                return start, stop
        except Exception:
            if attempt == 5:
                raise
    raise RuntimeError("unreachable")


def direct_resumable_download(
    file_id: int,
    destination: Path,
    total_size: int,
    workers: int = 4,
    chunk_size: int = 32 * 1024 * 1024,
) -> None:
    state_path = destination.with_suffix(destination.suffix + ".ranges.json")
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        prefix = int(state["prefix_bytes"])
        completed = set(state["completed"])
    else:
        prefix = destination.stat().st_size if destination.exists() else 0
        if prefix > total_size:
            raise ValueError(f"partial file is larger than expected: {destination}")
        completed: set[str] = set()
        state = {"prefix_bytes": prefix, "completed": []}
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    ranges = []
    start = prefix
    while start < total_size:
        stop = min(total_size - 1, start + chunk_size - 1)
        key = f"{start}-{stop}"
        if key not in completed:
            ranges.append((start, stop, key))
        start = stop + 1
    if not ranges:
        state_path.unlink(missing_ok=True)
        return

    fd = os.open(destination, os.O_RDWR | os.O_CREAT)
    os.ftruncate(fd, total_size)
    completed_bytes = prefix + sum(
        int(key.split("-")[1]) - int(key.split("-")[0]) + 1
        for key in completed
    )
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(download_range, file_id, fd, begin, end): (key, begin, end)
                for begin, end, key in ranges
            }
            for future in as_completed(futures):
                key, begin, end = futures[future]
                future.result()
                completed.add(key)
                completed_bytes += end - begin + 1
                state["completed"] = sorted(completed, key=lambda item: int(item.split("-")[0]))
                temporary = state_path.with_suffix(state_path.suffix + ".tmp")
                temporary.write_text(json.dumps(state, indent=2), encoding="utf-8")
                temporary.replace(state_path)
                print(
                    f"file {file_id}: {completed_bytes / total_size * 100:.1f}% "
                    f"({completed_bytes / 1e9:.2f}/{total_size / 1e9:.2f} GB)",
                    flush=True,
                )
    finally:
        os.close(fd)
    state_path.unlink(missing_ok=True)


def main() -> None:
    ROOT.joinpath("raw").mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"https://api.figshare.com/v2/articles/{ARTICLE_ID}") as response:
        article = json.load(response)
    metadata = {int(item["id"]): item for item in article["files"]}
    mapping: dict[str, dict] = {}

    for file_id in CORNER_FILE_IDS:
        item = metadata[file_id]
        destination = ROOT / "raw" / f"CornerReflector_{file_id}.7z"
        complete = destination.exists() and destination.stat().st_size == int(item["size"])
        range_state = destination.with_suffix(destination.suffix + ".ranges.json")
        if complete and range_state.exists():
            complete = False
        if not complete:
            print(f"Downloading file {file_id}: {destination.name}", flush=True)
            direct_resumable_download(
                file_id,
                destination,
                int(item["size"]),
            )
        actual_md5 = md5(destination)
        if actual_md5 != item["computed_md5"]:
            raise SystemExit(
                f"MD5 mismatch for {destination.name}: {actual_md5} != {item['computed_md5']}"
            )
        distance = distance_in_archive(destination)
        extracted = extract_measurements(destination, distance)
        mapping[str(distance)] = {
            "figshare_file_id": file_id,
            "archive": str(destination),
            "bytes": destination.stat().st_size,
            "md5": actual_md5,
            "extracted": str(extracted),
            "measurements": len(list(extracted.rglob("*.mat"))),
        }
        manifest = ROOT / "multirange_manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "dataset": "TSMS-Drone",
                    "doi": "10.25452/figshare.plus.30027313.v1",
                    "distances": dict(sorted(mapping.items(), key=lambda pair: int(pair[0]))),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Ready: {distance} m ({len(mapping)}/6 distances)", flush=True)
        if len(mapping) >= 6:
            print(f"Acquisition target reached. Manifest: {manifest}", flush=True)
            return

    raise SystemExit("All corner-reflector files were processed but fewer than six distances found")


if __name__ == "__main__":
    main()
