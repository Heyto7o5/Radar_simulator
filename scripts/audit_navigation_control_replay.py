#!/usr/bin/env python3
"""Reproduce learned commands while denying all episode truth/scene file reads."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

from radar_platform.navigation.io import ROOT,local_path,read_json,write_json
from radar_platform.navigation.controller import Controller
from radar_platform.navigation.dynamics import Command,Limits
from radar_platform.navigation.inference import Predictor
from radar_platform.navigation.observation import Observation
from radar_platform.navigation.state_filter import RadarStateFilter
from radar_platform.navigation.episode import model_fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=local_path(args.run);output=local_path(args.output)
    if not (root/'complete.json').exists():raise ValueError('episode must finish before control replay audit')
    protocol=read_json(root/'protocol.json')
    if protocol['mode']!='learned':raise ValueError('truth/no-correction references are not radar-only evidence')
    models=local_path(protocol['model_dir']);limits=Limits(**protocol['spec']['limits'])
    filter_config=protocol.get('state_filter')
    rows=[json.loads(line) for line in (root/'controls.jsonl').read_text().splitlines() if line]
    output.mkdir(parents=True,exist_ok=False);forbidden=[];opened=[]
    training=read_json(models/'training_protocol.json')
    operational_names={'training_protocol.json','status.json','acquisition_axes.npz','acquisition_timing.json','uncertainty.json','metrics.json'}
    operational_names.update(f'seed_{s}.pt' for s in training['seeds'])
    allowed_model_files={models/name for name in operational_names}
    def guard(event,arguments):
        if event!='open' or not arguments or not isinstance(arguments[0],(str,bytes)):return
        path=Path(arguments[0]).resolve()
        if path.is_relative_to(ROOT/'.runtime'):
            allowed=path in allowed_model_files or path.is_relative_to(root/'observations') or path.is_relative_to(output)
            if not allowed:forbidden.append(str(path));raise PermissionError('control audit forbids experiment truth and scene access')
            opened.append(str(path))
    sys.addaudithook(guard)
    predictor=Predictor(models,threads=4);controller=Controller(limits,gentle_terminal=protocol.get('gentle_terminal',False))
    filt=RadarStateFilter(filter_config['measurement_std'],limits) if filter_config else None
    previous=Command.coast();comparisons=[]
    for row in rows:
        observation=Observation.load(root/'observations'/f'{row["frame"]:04d}.npz')
        estimate,_=predictor.predict(observation)
        recorded_raw=row.get('raw_prediction',row['prediction'])
        raw=np.r_[estimate.relative_position_body_m,estimate.velocity_body_mps]
        expected_raw=np.r_[recorded_raw['relative_position_body_m'],recorded_raw['velocity_body_mps']]
        np.testing.assert_allclose(raw,expected_raw,atol=1e-4,rtol=1e-5)
        if filt is not None:estimate,_=filt.update(estimate,previous)
        command,control=controller.step(estimate)
        expected=row['command'];actual=np.r_[command.acceleration_body_mps2,command.yaw_rate_rad_s]
        target=np.r_[expected['acceleration_body_mps2'],expected['yaw_rate_rad_s']]
        np.testing.assert_allclose(actual,target,atol=1e-4,rtol=1e-4)
        if control['estimated_arrival']!=row['control']['estimated_arrival']:raise AssertionError('arrival decision differs')
        if control['mode']!=row['control']['mode']:raise AssertionError('control mode differs')
        comparisons.append(dict(frame=row['frame'],time_s=row['time_s'],
            maximum_raw_prediction_difference=float(np.max(abs(raw-expected_raw))),
            maximum_command_difference=float(np.max(abs(actual-target)))))
        previous=command
    write_json(output/'audit.json',dict(commands_reproduced=len(rows),all_commands_match=True,
        model_sha256=model_fingerprint(models),
        filter_enabled=filt is not None,forbidden_read_attempts=forbidden,
        scope='Only radar observations, frozen model/calibration, actuator limits and replayed command history; episode truth and scene data denied',
        comparisons=comparisons,opened_experiment_files=sorted(set(opened))))
    print(json.dumps(dict(commands=len(rows),filter_enabled=filt is not None,forbidden_read_attempts=forbidden,
        max_command_difference=max((r['maximum_command_difference'] for r in comparisons),default=0.))))


if __name__=='__main__':main()
