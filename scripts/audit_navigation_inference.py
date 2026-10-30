#!/usr/bin/env python3
"""Run frozen inference with filesystem reads of experiment truth denied."""
import argparse
import json
import shutil
import sys
from pathlib import Path
import numpy as np

from radar_platform.navigation.io import ROOT,local_path,read_json,write_json
from radar_platform.navigation.inference import Predictor
from radar_platform.navigation.observation import Observation
from radar_platform.navigation.episode import model_fingerprint


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--models',required=True);p.add_argument('--output',required=True)
    p.add_argument('--split',choices=['val','test'],default='test')
    args=p.parse_args();data=local_path(args.data);models=local_path(args.models);output=local_path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    records=[r for r in read_json(data/'protocol.json')['records'] if r['split']==args.split]
    if not records:raise ValueError('requested evaluation split is empty')
    selected=[0,len(records)//2,len(records)-1]
    for index in selected:shutil.copyfile(data/'cases'/records[index]['id']/'rda.npz',output/f'observation_{index}.npz')
    with np.load(models/f'{args.split}_predictions.npz',allow_pickle=False) as f:expected=f['prediction'].copy()
    training=read_json(models/'training_protocol.json')
    operational_names={'training_protocol.json','status.json','acquisition_axes.npz','acquisition_timing.json','uncertainty.json','metrics.json'}
    operational_names.update(f'seed_{s}.pt' for s in training['seeds'])
    allowed_model_files={models/name for name in operational_names}
    opened=[];denied=[]
    def audit(event,arguments):
        if event!='open' or not arguments or not isinstance(arguments[0],(str,bytes)):return
        path=Path(arguments[0]).resolve()
        if path.is_relative_to(ROOT/'.runtime'):
            if not (path in allowed_model_files or path.is_relative_to(output)):
                denied.append(str(path));raise PermissionError('runtime truth/scene access prohibited by inference audit')
            opened.append(str(path))
    sys.addaudithook(audit)
    predictor=Predictor(models,threads=4)
    results=[]
    for index in selected:
        estimate,diagnostic=predictor.predict(Observation.load(output/f'observation_{index}.npz'))
        actual=np.r_[estimate.relative_position_body_m,estimate.velocity_body_mps]
        difference=float(np.max(abs(actual-expected[index])))
        np.testing.assert_allclose(actual,expected[index],rtol=1e-5,atol=1e-4)
        results.append(dict(test_index=index,maximum_state_difference=difference,prediction=actual.tolist(),valid=estimate.valid))
    write_json(output/'audit.json',dict(truth_and_scene_reads_denied=True,attempted_forbidden_reads=denied,
        split=args.split,model_sha256=model_fingerprint(models),samplewise_model_predictions_and_labels_denied_during_inference=True,
        predictions_match_frozen_batch=True,results=results,opened_experiment_files=sorted(set(opened)),
        scope='Estimator constructor and prediction evaluated with only frozen model and isolated radar observations accessible under runtime'))
    print(json.dumps(dict(cases=len(results),maximum_difference=max(r['maximum_state_difference'] for r in results),forbidden_reads=denied)))


if __name__=='__main__':main()
