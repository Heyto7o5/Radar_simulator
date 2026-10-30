"""Public offline execution: the same model sessions used by the HTTP API."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from .unified.presets import presets
from .unified.provenance import source_snapshot
from .unified.session import session_from_config
from .unified.storage import write_json


def run_configuration(raw, output, *, migration_notes=(), on_frame=None):
    """Persist a model-specific acquisition without overwriting existing runs."""
    session = session_from_config(raw)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    session.save_inputs(output)
    snapshot, provenance = source_snapshot()
    (output / 'source.zip').write_bytes(snapshot)
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'data_origin': session.data_origin, 'status': 'running',
        'estimate': session.estimate, 'migration_notes': list(migration_notes),
        'source_sha256': provenance['files_sha256'], 'provenance': provenance,
        'request_sha256': hashlib.sha256((output / 'request.json').read_bytes()).hexdigest(),
        'frames': [],
    }
    write_json(output / 'run.json', manifest)
    started = time.perf_counter()
    try:
        for index in range(session.frame_count):
            result = session.frame(index)
            meta = session.save_frame(output, result)
            record = {'frame_index': index, 'elapsed_s': meta['elapsed_s'],
                      'warnings': meta.get('warnings', []), 'directory': f'frame_{index:04d}'}
            manifest['frames'].append(record)
            write_json(output / 'run.json', manifest)
            if on_frame is not None:
                on_frame(record)
        manifest['status'] = 'complete'
    except BaseException as exc:
        manifest['status'] = 'cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed'
        manifest['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        manifest['total_elapsed_s'] = time.perf_counter() - started
        write_json(output / 'run.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', type=Path)
    source.add_argument('--legacy-config', type=Path, help='Migrate historical high-altitude JSON')
    source.add_argument('--preset', choices=list(presets()))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    notes = []
    if args.legacy_config:
        from .models import SimulationRequest
        from .unified.legacy_config import convert_legacy
        request, notes = convert_legacy(SimulationRequest.model_validate_json(args.legacy_config.read_text()))
        raw = request.model_dump(mode='json')
    else:
        raw = json.loads(args.config.read_text()) if args.config else presets()[args.preset]
    result = run_configuration(raw, args.output, migration_notes=notes,
        on_frame=lambda row: print(json.dumps(row, ensure_ascii=False), flush=True))
    print(json.dumps({'output': str(args.output), 'status': result['status'],
                      'elapsed_s': result['total_elapsed_s']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
