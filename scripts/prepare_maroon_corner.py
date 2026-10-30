#!/usr/bin/env python3
"""Fetch the small, paired geometry/measurement benchmark; never into source."""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

COMMIT = "19dcaf137773ed0ba95209d04726fbc9d455bd6a"
SOURCE = "https://github.com/nihofm/polarized-radar-pathtracing"
FILES = {
    "signal.npz": "data/corner_reflector/signal.npz",
    "mesh_recreated.obj": "data/corner_reflector/mesh_recreated.obj",
    "alignment.json": "data/corner_reflector/alignment.json",
    "Tx.dat": "data/Tx.dat",
    "Rx.dat": "data/Rx.dat",
    "SOURCE_README.md": "data/corner_reflector/README.md",
    "SOURCE_LICENSE.txt": "LICENSE",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/nas/datasets/hyt/datasets/radar_open/maroon_corner"))
    args = parser.parse_args()
    root = args.root.resolve()
    project = Path(__file__).resolve().parents[1]
    if root == project or project in root.parents:
        raise SystemExit("Raw measurements must be outside the source project")
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    entries = []
    for name, remote in FILES.items():
        url = f"https://raw.githubusercontent.com/nihofm/polarized-radar-pathtracing/{COMMIT}/{remote}"
        path = raw / name
        if not path.exists():
            payload = urllib.request.urlopen(url, timeout=60).read()
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(payload)
            temporary.replace(path)
        payload = path.read_bytes()
        if len(payload) < 20 or payload.startswith(b"404:"):
            raise ValueError(f"invalid source file {path}")
        entries.append({"file": name, "source_url": url, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
        print(name, len(payload), flush=True)
    manifest = {
        "dataset": "MAROON-associated corner reflector reduced capture",
        "source": SOURCE,
        "commit": COMMIT,
        "retrieved_utc": datetime.now(timezone.utc).isoformat(),
        "license": "CC BY-NC 4.0 (source repository and bundled data)",
        "measurement": "mean of 10 object captures minus mean of 10 empty-room captures from another session",
        "signal_axes": ["tx", "rx", "frequency"],
        "has_coherent_slow_time": False,
        "files": entries,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(root / "manifest.json")


if __name__ == "__main__":
    main()
