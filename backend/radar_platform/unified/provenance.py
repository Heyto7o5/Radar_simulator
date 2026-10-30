"""Record the local solver sources used when a new job is submitted."""
import hashlib
import io
from pathlib import Path
import sys
import zipfile


def source_snapshot():
    package=Path(__file__).resolve().parents[1]
    hashes={};buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(package.rglob('*.py')):
            raw=path.read_bytes();name='backend/radar_platform/'+path.relative_to(package).as_posix()
            hashes[name]=hashlib.sha256(raw).hexdigest();z.writestr(name,raw)
        project=package.parents[1]
        for name in ['pyproject.toml','scripts/run_scene.py']:
            path=project/name
            if path.is_file():
                raw=path.read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest();z.writestr(name,raw)
    return buffer.getvalue(),{'python_version':sys.version,'files_sha256':hashes,
        'source_hash':hashlib.sha256(''.join(k+v for k,v in sorted(hashes.items())).encode()).hexdigest(),
        'scope':'On-disk source snapshot at submission. Restart the service after code edits to keep loaded modules aligned; not a full environment lock.'}
