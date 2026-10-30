"""One installed entry point for simulation, validation and service startup."""
import argparse
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'run':
        from .runner import main as run
        return run(argv[1:])
    parser = argparse.ArgumentParser(prog='radar-platform', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('run', help='Generate an acquisition; run --help lists options')
    sub.add_parser('capabilities', help='Print scope and evidence boundaries')
    preset = sub.add_parser('presets', help='List or export built-in dynamic scenes')
    preset.add_argument('--name')
    check = sub.add_parser('check', help='Validate configuration without synthesis')
    check.add_argument('--config', type=Path, required=True)
    check.add_argument('--motion', action='store_true')
    serve = sub.add_parser('serve', help='Serve API and built UI; no authentication')
    serve.add_argument('--host', default=os.getenv('RADAR_API_HOST', '127.0.0.1'))
    serve.add_argument('--port', type=int, default=int(os.getenv('RADAR_API_PORT', '8000')))
    serve.add_argument('--reload', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'serve':
        import uvicorn
        return uvicorn.run('radar_platform.api:app', host=args.host, port=args.port, reload=args.reload)
    if args.command == 'capabilities':
        from .capabilities import capabilities
        result = capabilities()
    elif args.command == 'presets':
        from .unified.presets import presets
        values = presets()
        if args.name is not None and args.name not in values:
            parser.error('unknown preset: ' + args.name)
        result = values[args.name] if args.name else {name: value['name'] for name, value in values.items()}
    elif args.command == 'check':
        from .configuration import inspect_configuration
        result = inspect_configuration(raw := json.loads(args.config.read_text()))
        if args.motion and result['status'] != 'unsupported':
            from .unified.schema import Simulation
            from .unified.motion_check import motion_error_report
            try:
                result['motion_check'] = motion_error_report(Simulation.model_validate(raw))
                if result['motion_check']['status'] == 'risk':
                    result['status'] = 'risk'
            except ValueError as exc:
                result.update(status='unsupported', motion_error=str(exc))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if result['status'] == 'unsupported':
            raise SystemExit(2)
        return
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
