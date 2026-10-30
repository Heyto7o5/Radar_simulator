"""Project-contained, atomic experiment storage."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def local_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'path outside authorized project: {path}')
    return path


def read_json(path):
    return json.loads(local_path(path).read_text())


def write_json(path, value):
    path = local_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()
