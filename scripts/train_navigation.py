#!/usr/bin/env python3
"""Select three radar-only regressors on validation and freeze before testing."""
import argparse
import hashlib
import zipfile
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import json
import os
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader

from radar_platform.navigation.io import local_path,write_json,read_json
from radar_platform.navigation.model import RadarNavigationNet,ARCHITECTURE
from radar_platform.navigation.learning import RadarDataset,metrics,infer_loader
from radar_platform.navigation.observation import Observation


def train_one(job):
    args,seed=job;root=local_path(args['output']);torch.set_num_threads(args['threads'])
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    train=RadarDataset(args['data'],'train');val=RadarDataset(args['data'],'val')
    loader=DataLoader(train,batch_size=args['batch_size'],shuffle=True,num_workers=0)
    validation=DataLoader(val,batch_size=args['batch_size'],shuffle=False,num_workers=0)
    model=RadarNavigationNet()
    # Stable statistics from training inputs only, not evaluation frames.
    total=torch.zeros(16);squares=torch.zeros(16);count=0
    for x,_,_,_ in DataLoader(train,batch_size=args['batch_size'],shuffle=False,num_workers=0):
        total+=x.sum(dim=(0,1,3,4,5));squares+=(x*x).sum(dim=(0,1,3,4,5))
        count+=x.shape[0]*x.shape[1]*x.shape[3]*x.shape[4]*x.shape[5]
    mean=total/count;std=torch.sqrt((squares/count-mean**2).clamp_min(1e-6))
    model.input_mean.copy_(mean.reshape(1,16,1,1,1));model.input_std.copy_(std.reshape(1,16,1,1,1))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args['epochs'],eta_min=1e-5)
    best=float('inf');best_epoch=-1;history=[];start=time.time()
    for epoch in range(args['epochs']):
        model.train();loss_sum=0.
        for x,dt,y,valid in loader:
            optimizer.zero_grad(set_to_none=True)
            prediction,logit=model(x,dt)
            per=torch.nn.functional.smooth_l1_loss((prediction-y)/model.output_scale,
                                                   torch.zeros_like(y),reduction='none').mean(dim=1)
            loss=(per*valid).sum()/valid.sum().clamp_min(1)+.1*torch.nn.functional.binary_cross_entropy_with_logits(logit,valid)
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            loss_sum+=float(loss.detach())*len(x)
        scheduler.step();vp,prob=infer_loader(model,validation)
        result=metrics(vp,val.labels,val.valid)
        score=result['position_mean_m']+.5*result['velocity_mean_mps']
        row=dict(epoch=epoch+1,train_loss=loss_sum/len(train),validation=result,
                 validity_accuracy=float(np.mean((prob>=.5)==(val.valid>0))),elapsed_s=time.time()-start)
        history.append(row)
        if score<best:
            best=score;best_epoch=epoch+1
            torch.save(dict(state_dict=model.state_dict(),seed=seed,epoch=best_epoch,
                validation_score=best,architecture=ARCHITECTURE,input_shape=[2,16,96,21,21],
                label='ground-target relative position and platform velocity; body forward/right/up at second CPI end'),
                root/f'seed_{seed}.tmp.pt')
            (root/f'seed_{seed}.tmp.pt').replace(root/f'seed_{seed}.pt')
        write_json(root/f'history_{seed}.json',history)
        print(json.dumps(dict(seed=seed,**row,best_epoch=best_epoch)),flush=True)
        if epoch+1-best_epoch>=args['patience']:break
    return dict(seed=seed,best_epoch=best_epoch,best_validation_score=best,seconds=time.time()-start)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=300);p.add_argument('--patience',type=int,default=50)
    p.add_argument('--threads',type=int,default=8);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,23,37])
    p.add_argument('--parallel-seeds',type=int,default=3)
    p.add_argument('--evaluate-test',action='store_true',help='Only set after validation protocol is frozen; otherwise leave test unopened')
    args=vars(p.parse_args());data=local_path(args['data']);root=local_path(args['output'])
    if read_json(data/'status.json')['status']!='complete':raise ValueError('physical dataset incomplete')
    root.mkdir(parents=True,exist_ok=False)
    source_files=[local_path(__file__),local_path('backend/radar_platform/navigation/model.py'),
                  local_path('backend/radar_platform/navigation/learning.py'),local_path('backend/radar_platform/navigation/observation.py'),
                  local_path('backend/radar_platform/navigation/inference.py')]
    write_json(root/'source_sha256.json',{str(f.relative_to(local_path('.'))):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files})
    with zipfile.ZipFile(root/'training_source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for f in source_files:archive.write(f,str(f.relative_to(local_path('.'))))
    write_json(root/'training_protocol.json',{**args,'architecture':ARCHITECTURE,'device':'cpu',
        'selection':'min validation position mean + 0.5 * velocity mean; test remains unopened until frozen',
        'torch_version':torch.__version__})
    write_json(root/'status.json',dict(status='training',pid=os.getpid()))
    with ProcessPoolExecutor(max_workers=args['parallel_seeds'],mp_context=get_context('spawn')) as pool:
        futures=[pool.submit(train_one,(args,seed)) for seed in args['seeds']]
        summaries=[]
        for f in as_completed(futures):summaries.append(f.result());write_json(root/'training_summary.json',summaries)
    torch.set_num_threads(args['threads'])
    models=[]
    for seed in args['seeds']:
        model=RadarNavigationNet()
        model.load_state_dict(torch.load(root/f'seed_{seed}.pt',map_location='cpu',weights_only=True)['state_dict'])
        model.eval();models.append(model)
    results={}
    for split in (['val','test'] if args['evaluate_test'] else ['val']):
        dataset=RadarDataset(data,split);loader=DataLoader(dataset,batch_size=args['batch_size'])
        predictions=[];probabilities=[]
        for model in models:
            pred,prob=infer_loader(model,loader);predictions.append(pred);probabilities.append(prob)
        ensemble=np.mean(predictions,axis=0);valid=dataset.valid>0
        spread=np.linalg.norm(np.std(predictions,axis=0)[:,:3],axis=1)
        if split=='val':
            error=np.linalg.norm(ensemble[:,:3]-dataset.labels[:,:3],axis=1)
            floor=max(.25,float(np.median(error[valid])))
            multiplier=float(np.percentile(error[valid]/(spread[valid]+floor),80))
            write_json(root/'uncertainty.json',dict(floor_m=floor,multiplier=multiplier,
                maximum_error_scale_m=3.,valid_probability_threshold=.5,
                description='Validation residual scale, not a certified probability of correctness or unknown-scene detector'))
        results[split]=metrics(ensemble,dataset.labels,dataset.valid)
        results[split]['validity_accuracy']=float(np.mean((np.mean(probabilities,axis=0)>=.5)==valid))
        np.savez_compressed(root/f'{split}_predictions.npz',prediction=ensemble,truth=dataset.labels,valid=valid,
                            per_seed=np.asarray(predictions),probability=np.mean(probabilities,axis=0))
        if split=='test':
            shuffled=np.random.default_rng(31415).permutation(len(ensemble))
            results['shuffled_input_control']=metrics(ensemble[shuffled],dataset.labels,dataset.valid)
    record=read_json(data/'protocol.json')['records'][0]
    observation=Observation.load(data/'cases'/record['id']/'rda.npz')
    np.savez(root/'acquisition_axes.npz',**{name:getattr(observation,name) for name in
             ['range_m','velocity_mps','elevation_deg','azimuth_deg']})
    write_json(root/'metrics.json',results)
    write_json(root/'status.json',dict(status='complete',seeds=args['seeds']))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
