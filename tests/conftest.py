"""Test storage respecting application-layer project containment."""
from pathlib import Path
import tempfile
import pytest


@pytest.fixture
def project_tmp_path():
    # Application I/O deliberately rejects /tmp and arbitrary external paths.
    # Keep that production boundary; allocate isolated test data inside the repo.
    base = Path(__file__).resolve().parents[1] / '.runtime' / 'test-sandboxes'
    base.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='case-', dir=base) as path:
        yield Path(path)
