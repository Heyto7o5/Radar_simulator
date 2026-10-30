#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import py7zr


def main() -> None:
    parser = argparse.ArgumentParser(description="Selectively extract TSMS corner-reflector RD measurements")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--distance", type=int, default=2)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    targets = [
        f"Raw File/Corner Reflector_{args.distance}m_{index:03d}.mat"
        for index in range(1, args.count + 1)
    ]
    with py7zr.SevenZipFile(args.archive, "r") as archive:
        available = set(archive.getnames())
        missing = [target for target in targets if target not in available]
        if missing:
            raise SystemExit(f"Archive does not contain {missing[0]}")
        archive.extract(path=args.output, targets=targets)
    print(f"Extracted {len(targets)} measurements to {args.output}")


if __name__ == "__main__":
    main()

