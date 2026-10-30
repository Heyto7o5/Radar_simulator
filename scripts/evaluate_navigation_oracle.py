#!/usr/bin/env python3
"""Truth-control plant references with the identical radar acquisition clock.

Images cannot influence the explicit truth controller. A full physical default
episode is used to verify the omitted-acquisition baseline trajectory exactly.
"""
import argparse
import json
import numpy as np
from navigation_campaign import cases

from radar_platform.navigation.scene import navigation_scene
from radar_platform.navigation.acquisition import cpi_duration
from radar_platform.navigation.timeline import covered_cpi_end
from radar_platform.navigation.dynamics import State,Command,Limits,propagate
from radar_platform.navigation.controller import Controller,Estimate
from radar_platform.navigation.episode import Evaluator,default_episode
from radar_platform.navigation.io import local_path,write_json,read_json


def reference(spec,cfg,gentle_terminal=False):
    duration=cpi_duration(cfg);state=State(0.,np.array(spec['position_neu_m']),np.array(spec['velocity_neu_mps']),spec['yaw_rad'])
    limits=Limits(**spec['limits']);controller=Controller(limits,gentle_terminal=gentle_terminal);evaluator=Evaluator(cfg);evaluator.observe([state])
    command=Command.coast();next_correction=0.;termination='timeout';updates=0
    for k in range(int(np.floor((spec['max_time_s']-duration)/.25))+1):
        start=k*.25
        if start>state.time_s:
            state,samples=propagate(state,command,start,limits);evaluator.observe(samples[1:])
        if evaluator.failure:termination=evaluator.failure;break
        state,samples=propagate(state,command,covered_cpi_end(cfg,start),limits);evaluator.observe(samples[1:])
        if evaluator.failure:termination=evaluator.failure;break
        if k and state.time_s+1e-9>=next_correction:
            y=state.label();command,info=controller.step(Estimate(state.time_s,y[:3],y[3:]));updates+=1
            next_correction=state.time_s+spec['correction_interval_s']
            if info['estimated_arrival']:termination='estimated_arrival';break
    result=evaluator.summary(state,termination)
    result.update(mode='plant_only_truth_reference',episode=spec['id'],corrections=updates,radar_frames_generated=0,
        gentle_terminal=gentle_terminal,
        reason='Truth controller ignores radar images; physical acquisition delays retained exactly')
    return result,evaluator.samples


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--physical-reference',required=True)
    p.add_argument('--gentle-terminal',action='store_true')
    args=p.parse_args();root=local_path(args.output);root.mkdir(parents=True,exist_ok=True);cfg,_=navigation_scene()
    default,trace=reference(default_episode(),cfg)
    actual=read_json(local_path(args.physical_reference)/'truth_trajectory.json')
    assert trace==actual,'plant-only reference changed the physical episode trajectory'
    write_json(root/'physical_clock_equivalence.json',dict(full_trace_identical=True,samples=len(trace),
        physical_reference=str(local_path(args.physical_reference)),default=default))
    results=[]
    for spec in cases():
        result,trace=reference(spec,cfg,gentle_terminal=args.gentle_terminal);folder=root/spec['id'];folder.mkdir(exist_ok=True)
        write_json(folder/'protocol.json',spec);write_json(folder/'truth_trajectory.json',trace)
        write_json(folder/'complete.json',result);results.append(result)
    write_json(root/'results.json',results);write_json(root/'status.json',dict(status='complete',cases=len(results)))
    print(json.dumps(dict(cases=len(results),successes=sum(r['success'] for r in results),physical_reference_trace_identical=True)))


if __name__=='__main__':main()
