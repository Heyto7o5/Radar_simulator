#!/usr/bin/env python3
"""Standalone navigation inference: only frozen model and an RDA observation."""
import argparse
import json
from radar_platform.navigation.io import local_path,write_json
from radar_platform.navigation.observation import Observation
from radar_platform.navigation.inference import Predictor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',required=True);p.add_argument('--observation',required=True)
    p.add_argument('--output',required=True);p.add_argument('--threads',type=int,default=4)
    args=p.parse_args();output=local_path(args.output)
    if output.exists():raise FileExistsError('existing prediction preserved')
    estimate,diagnostic=Predictor(args.model_dir,args.threads).predict(Observation.load(local_path(args.observation)))
    result=dict(relative_position_body_m=estimate.relative_position_body_m.tolist(),
        velocity_body_mps=estimate.velocity_body_mps.tolist(),coordinate_order=['forward','right','up'],
        valid=estimate.valid,error_scale_m=estimate.error_scale_m,time_s=estimate.time_s,diagnostic=diagnostic)
    write_json(output,result);print(json.dumps(result))


if __name__=='__main__':main()
