#!/usr/bin/env python3
"""Export DART inputs to common configuration; this command does not simulate."""
import argparse
import json
from pathlib import Path
from radar_platform.dart_scene_adapter import prepare_scene


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['geometry','trajectory','radar-config','output']:
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--frame-index',type=int,default=0)
    parser.add_argument('--constant-motion',action='store_true')
    parser.add_argument('--hard-beam',action='store_true')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Refusing to overwrite existing configuration')
    config=prepare_scene(args.geometry,args.trajectory,args.radar_config,
        valid_frame_index=args.frame_index,sampled_trajectory=not args.constant_motion,smooth_beam=not args.hard_beam)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:handle.write(config.model_dump_json(indent=2)+'\n')
    print(json.dumps({'configuration':str(args.output),'simulation_performed':False,'frame_index':args.frame_index}))


if __name__=='__main__':main()
