#!/usr/bin/env python3
"""Paired uncorrected plant trajectories; radar synthesis is unnecessary here.

Commands stay identically zero, so observations cannot affect this baseline.
This is explicitly a plant-only control, never an RDA-guided flight.
"""
import argparse
import json
import numpy as np
from navigation_campaign import cases
from radar_platform.navigation.scene import navigation_scene
from radar_platform.navigation.dynamics import State,Command,Limits,propagate
from radar_platform.navigation.episode import Evaluator
from radar_platform.navigation.io import local_path,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    args=p.parse_args();root=local_path(args.output);root.mkdir(parents=True,exist_ok=True);cfg,_=navigation_scene()
    results=[]
    for spec in cases():
        state=State(0.,np.array(spec['position_neu_m']),np.array(spec['velocity_neu_mps']),spec['yaw_rad'])
        evaluator=Evaluator(cfg);evaluator.observe([state]);limits=Limits(**spec['limits'])
        termination='timeout'
        for step in range(int(spec['max_time_s']/.02)):
            state,samples=propagate(state,Command.coast(),(step+1)*.02,limits);evaluator.observe(samples[1:])
            if evaluator.failure:termination=evaluator.failure;break
        summary=evaluator.summary(state,termination);summary.update(episode=spec['id'],mode='plant_only_no_correction',
            radar_frames_generated=0,reason='Radar observations have no causal effect under identically zero control')
        folder=root/spec['id'];folder.mkdir(exist_ok=True);write_json(folder/'protocol.json',spec)
        write_json(folder/'truth_trajectory.json',evaluator.samples);write_json(folder/'complete.json',summary);results.append(summary)
    write_json(root/'results.json',results);write_json(root/'status.json',dict(status='complete',cases=len(results)))
    print(json.dumps(dict(cases=len(results),successes=sum(r['success'] for r in results))))


if __name__=='__main__':main()
