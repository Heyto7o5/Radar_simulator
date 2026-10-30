#!/usr/bin/env python3
"""Check dynamic-scene applicability without generating ADC or starting a server."""
import argparse,json
from pathlib import Path
from radar_platform.unified.applicability import inspect_configuration


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--motion',action='store_true',help='Also compare linearized and instantaneous geometric paths')
    parser.add_argument('--motion-points',type=int,default=256)
    parser.add_argument('--motion-slots',type=int,default=16)
    parser.add_argument('--motion-fast-samples',type=int,default=5)
    args=parser.parse_args()
    raw=json.loads(args.config.read_text())
    result=inspect_configuration(raw)
    if args.motion and result['status']!='unsupported':
        from radar_platform.unified.schema import Simulation
        from radar_platform.unified.motion_check import motion_error_report
        result['motion_check']=motion_error_report(Simulation.model_validate(raw),
            max_scatterers=args.motion_points,max_slots=args.motion_slots,fast_samples=args.motion_fast_samples)
        if result['motion_check']['status']=='risk':result['status']='risk'
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x') as handle:handle.write(text+'\n')
    print(text)
    if result['status']=='unsupported':raise SystemExit(2)


if __name__=='__main__':main()
