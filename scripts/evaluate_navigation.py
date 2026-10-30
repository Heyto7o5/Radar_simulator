#!/usr/bin/env python3
"""Frozen held-out evaluation, nearest-neighbor baseline and input shuffling."""
import argparse
import hashlib
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from radar_platform.navigation.io import local_path,read_json,write_json,digest
from radar_platform.navigation.learning import RadarDataset,metrics,infer_loader
from radar_platform.navigation.model import RadarNavigationNet
from radar_platform.navigation.episode import model_fingerprint


def compressed_bank(dataset):
    bank=[]
    for record in dataset.records:
        x=np.load(dataset.root/'cases'/record['id']/'input.npy',allow_pickle=False)
        row=x[:,::2,::2,::2,::2].astype(np.float32).ravel()
        bank.append(row/max(float(np.linalg.norm(row)),1e-10))
    return np.stack(bank)


def nearest(bank,labels,queries,k):
    predictions=[]
    for start in range(0,len(queries),32):
        distances=np.maximum(0,2-2*queries[start:start+32]@bank.T)
        ix=np.argsort(distances,axis=1)[:,:k]
        weights=1/np.maximum(np.take_along_axis(distances,ix,axis=1),1e-6)**2
        predictions.append(np.sum(labels[ix]*weights[:,:,None],axis=1)/weights.sum(axis=1,keepdims=True))
    return np.concatenate(predictions)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--model-dir',required=True)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args();root=local_path(args.model_dir);torch.set_num_threads(args.threads)
    if read_json(root/'status.json')['status']!='complete':raise ValueError('training incomplete')
    protocol=read_json(root/'training_protocol.json')
    trained_records=[r for r in read_json(local_path(protocol['data'])/'protocol.json')['records'] if r['split'] in ['train','val']]
    evaluated_records=[r for r in read_json(local_path(args.data)/'protocol.json')['records'] if r['split'] in ['train','val']]
    if trained_records!=evaluated_records:raise ValueError('evaluation view changed the frozen training/validation split or records')
    if protocol['architecture'] in ['RadarNavigationHeatmap_v2','RadarNavigationHeatmap_v3']:
        from radar_platform.navigation.model_v2 import RadarNavigationHeatmap
        from radar_platform.navigation.model_v3 import RadarNavigationAnchored
        from radar_platform.navigation.learning_v2 import RadarDatasetV2,infer_loader_v2
        dataset_class=RadarDatasetV2;inference=infer_loader_v2
        model_class=RadarNavigationAnchored if protocol['architecture']=='RadarNavigationHeatmap_v3' else RadarNavigationHeatmap
    else:
        dataset_class=RadarDataset;model_class=RadarNavigationNet;inference=infer_loader
    frozen={f'seed_{s}.pt':hashlib.sha256((root/f'seed_{s}.pt').read_bytes()).hexdigest() for s in protocol['seeds']}
    if (root/'test_frozen_weights.json').exists():
        if read_json(root/'test_frozen_weights.json')!=frozen:raise ValueError('test previously evaluated different weights')
    else:write_json(root/'test_frozen_weights.json',frozen)
    train,val,test=[dataset_class(args.data,s) for s in ['train','val','test']]
    full_protocol=read_json(local_path(args.data)/'protocol.json')
    expected_counts=protocol.get('planned_full_counts',full_protocol.get('counts',{}))
    if expected_counts.get('test')!=len(test):raise ValueError('test count differs from the predeclared complete evaluation')
    evaluation=dict(data_directory=str(local_path(args.data)),
        records_sha256=digest(full_protocol['records']),base_config_sha256=digest(read_json(local_path(args.data)/'base_config.json')),
        test_cases=len(test),frozen_weights=frozen,model_sha256=model_fingerprint(root),
        scope='Complete predeclared test split; model selection uses train/validation only')
    if (root/'evaluation_protocol.json').exists():
        previous=read_json(root/'evaluation_protocol.json')
        for key in ['records_sha256','base_config_sha256','test_cases','frozen_weights','model_sha256']:
            if previous.get(key)!=evaluation[key]:raise ValueError('existing frozen test evaluation belongs to another dataset/model')
    write_json(root/'evaluation_protocol.json',evaluation)
    test_loader=DataLoader(test,batch_size=args.batch_size)
    predictions=[];probabilities=[];results={}
    for seed in protocol['seeds']:
        model=model_class();model.load_state_dict(torch.load(root/f'seed_{seed}.pt',map_location='cpu',weights_only=True)['state_dict'])
        pred,prob=inference(model,test_loader);predictions.append(pred);probabilities.append(prob)
        results[f'cnn_seed_{seed}']=metrics(pred,test.labels,test.valid)
    ensemble=np.mean(predictions,axis=0);results['cnn_ensemble']=metrics(ensemble,test.labels,test.valid)
    results['cnn_ensemble_all_states']=metrics(ensemble,test.labels)
    calibration=read_json(root/'uncertainty.json')
    spread=np.linalg.norm(np.std(predictions,axis=0)[:,:3],axis=1)
    error_scale=(spread+calibration['floor_m'])*calibration['multiplier']
    accepted=(np.mean(probabilities,axis=0)>=calibration['valid_probability_threshold'])&(error_scale<=calibration['maximum_error_scale_m'])
    errors=np.linalg.norm(ensemble[:,:3]-test.labels[:,:3],axis=1)
    has_building=any(o['id']=='center_landmark' for o in read_json(local_path(args.data)/'base_config.json')['scene']['objects'])
    results['evaluation_counts']=dict(total_states=len(test),eligible_regression_states=int(np.count_nonzero(test.valid)),
        ineligible_regression_states=int(np.count_nonzero(test.valid==0)),accepted_estimates=int(accepted.sum()),
        accepted_ineligible_states=int(np.count_nonzero(accepted&(test.valid==0))),
        accepted_estimates_over_3m=int(np.count_nonzero(accepted&(errors>3))),
        eligibility_definition=('At least one physical building cell contributes; not a measured detection/SNR criterion' if has_building else
                                'Terrain-only control: all finite radar observations are eligible; no building visibility label'),
        definition='Conditional, all-state and accepted-state metrics are all retained. No cases removed.')
    results['cnn_accepted_estimates']=metrics(ensemble,test.labels,accepted)
    results['shuffled_input_control']=metrics(ensemble[np.random.default_rng(31415).permutation(len(test))],test.labels,test.valid)
    bank=compressed_bank(train);vb=compressed_bank(val);tb=compressed_bank(test)
    eligible=train.valid>0;bank=bank[eligible];labels=train.labels[eligible]
    candidates={k:metrics(nearest(bank,labels,vb,k),val.labels,val.valid) for k in [1,4,8]}
    k=min(candidates,key=lambda key:candidates[key]['position_mean_m']+.5*candidates[key]['velocity_mean_mps'])
    knn=nearest(bank,labels,tb,k)
    results['nearest_neighbor']=metrics(knn,test.labels,test.valid)
    results['nearest_neighbor_all_states']=metrics(knn,test.labels)
    results['training_mean']=metrics(np.tile(labels.mean(axis=0),(len(test),1)),test.labels,test.valid)
    write_json(root/'baseline_selection.json',dict(k=k,validation=candidates))
    np.savez_compressed(root/'test_predictions.npz',prediction=ensemble,truth=test.labels,valid=test.valid,
        per_seed=np.asarray(predictions),probability=np.mean(probabilities,axis=0),nearest_neighbor=knn,
        accepted=accepted,error_scale_m=error_scale)
    write_json(root/'test_metrics.json',results)
    write_json(root/'status.json',{**read_json(root/'status.json'),'test_evaluated':True,'test_cases':len(test)})
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
