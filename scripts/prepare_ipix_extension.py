#!/usr/bin/env python3
"""Download the next two published IPIX records, without selecting on results."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request

RECORDS = [
    {"id": 19, "filename": "19931107_145028_starea.cdf", "excluded_gates": [7, 8, 9]},
    {"id": 25, "filename": "19931108_213827_starea.cdf", "excluded_gates": [6, 7, 8]},
]
SOURCE = "https://soma.ece.mcmaster.ca/ipix/dartmouth/"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/nas/datasets/hyt/datasets/radar_open/ipix_dartmouth"))
    args = parser.parse_args()
    root = args.root.resolve()
    project = Path(__file__).resolve().parents[1]
    if root == project or project in root.parents:
        raise SystemExit("raw data must remain outside source project")
    (root / "raw").mkdir(parents=True, exist_ok=True)

    def acquire(record):
        path = root / "raw" / record["filename"]
        url = SOURCE + "data/" + record["filename"]
        if not path.exists():
            with urllib.request.urlopen(url, timeout=50) as response:
                data = response.read(20_000_001)
            if len(data) > 20_000_000 or not data.startswith(b"CDF"):
                raise ValueError("unexpected CDF response")
            part = path.with_suffix(".cdf.part")
            part.write_bytes(data)
            part.replace(path)
        data = path.read_bytes()
        if not data.startswith(b"CDF"):
            raise ValueError("invalid cached CDF")
        result = dict(record, url=url, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        print(json.dumps(result), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        files = list(executor.map(acquire, RECORDS))
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "next two rows after #17 and #18 in official table, selected before evaluation",
        "source": SOURCE + "datasets.html", "files": files,
        "terms": "Research citation and communication conditions on source page; no external messages sent.",
    }
    (root / "extension_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
