import importlib.util
import io
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('dart_subset', SCRIPTS / 'prepare_dart_radar_subset.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_prefix_seek_read_and_no_fabricated_gaps():
    f = module.PrefixFile(io.BytesIO(b'abcdefghij'), 10, budget=6)
    assert f.read(2) == b'ab'
    f.seek(4)
    assert f.read(2) == b'ef'
    f.seek(-10, 2)
    buf = bytearray(3)
    assert f.readinto(buf) == 3 and buf == b'abc'
    f.seek(7)
    with pytest.raises(ValueError, match='budget'): f.read(1)
    f.seek(10)
    assert f.read(1) == b''


def test_truncated_source_is_error():
    f = module.PrefixFile(io.BytesIO(b'a'), 10, budget=10)
    with pytest.raises(IOError, match='truncated'): f.read(3)
