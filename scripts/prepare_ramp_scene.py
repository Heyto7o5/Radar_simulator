#!/usr/bin/env python3
"""Prepare a common configuration from explicit label inputs; does not simulate.

Input JSON is a list of labels with uid, class_id, x_m, y_m and
velocity_ned_mps. Missing velocity must be resolved explicitly by the caller.
"""
import argparse
import hashlib
import json
from pathlib import Path
from radar_platform.ramp_alignment import track_configuration


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--east-sign',type=int,choices=[-1,1],default=-1)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Refusing to overwrite existing configuration')
    payload=args.labels.read_bytes();labels=json.loads(payload)
    if not isinstance(labels,list):raise ValueError('input JSON must be an explicit list of labels')
    config=track_configuration(labels,east_sign=args.east_sign)
    config.input_provenance.append('Input label SHA256: '+hashlib.sha256(payload).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:handle.write(config.model_dump_json(indent=2)+'\n')
    print(json.dumps({'configuration':str(args.output),'objects':len(labels),'simulation_performed':False}))


if __name__=='__main__':main()
