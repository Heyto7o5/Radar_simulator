#!/usr/bin/env python3
"""Learn radar landmark heatmaps with physical coordinate decoding."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import hashlib
import json
import os
import random
import shutil
import time
import zipfile
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from radar_platform.navigation.io import local_path,read_json,write_json,digest
from radar_platform.navigation.model_v2 import RadarNavigationHeatmap,ARCHITECTURE_V2
from radar_platform.navigation.learning_v2 import RadarDatasetV2,infer_loader_v2
from radar_platform.navigation.learning import metrics


def train_one(job):
    args,seed=job;root=local_path(args['output']);data=local_path(args['data'])
    torch.set_num_threads(args['threads']);torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    train=RadarDatasetV2(data,'train');val=RadarDatasetV2(data,'val')
    loader=DataLoader(train,batch_size=args['batch_size'],shuffle=True,num_workers=0)
    validation=DataLoader(val,batch_size=args['batch_size'],shuffle=False,num_workers=0)
    base=read_json(data/'base_config.json');building=next((x for x in base['scene']['objects'] if x['id']=='center_landmark'),None)
    roof=0. if building is None else -building['position_ned_m'][2]+building['size_m'][2]/2
    model=RadarNavigationHeatmap(np.load(data/'coordinates.npy'),roof_height_m=roof)
    total=torch.zeros(20);square=torch.zeros(20);count=0
    for x,_,_,_ in DataLoader(train,batch_size=args['batch_size'],shuffle=False,num_workers=0):
        total+=x.sum(dim=(0,1,3,4,5));square+=(x*x).sum(dim=(0,1,3,4,5))
        count+=x.shape[0]*x.shape[1]*x.shape[3]*x.shape[4]*x.shape[5]
    mean=total/count;std=torch.sqrt((square/count-mean**2).clamp_min(1e-6))
    model.input_mean.copy_(mean.reshape(1,20,1,1,1));model.input_std.copy_(std.reshape(1,20,1,1,1))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args['epochs'],eta_min=1e-5)
    best=float('inf');best_epoch=-1;history=[];started=time.time()
    for epoch in range(args['epochs']):
        model.train();loss_sum=0.
        for x,baseline,y,valid in loader:
            optimizer.zero_grad(set_to_none=True);prediction,logit,heatmap=model(x,baseline)
            position=F.smooth_l1_loss((prediction[:,:3]-y[:,:3])/torch.tensor([5.,5.,3.]),torch.zeros_like(y[:,:3]),reduction='none').mean(dim=1)
            velocity=F.smooth_l1_loss((prediction[:,3:]-y[:,3:])/torch.tensor([3.,3.,1.]),torch.zeros_like(y[:,3:]),reduction='none').mean(dim=1)
            target=model.spatial_target(y)
            heat=-torch.sum(target*F.log_softmax(heatmap,dim=1),dim=1)
            loss=((position+.5*velocity+.25*heat)*valid).sum()/valid.sum().clamp_min(1)+.1*F.binary_cross_entropy_with_logits(logit,valid)
            if not torch.isfinite(loss):raise ValueError('nonfinite heatmap learning loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
            loss_sum+=float(loss.detach())*len(x)
        scheduler.step();vp,prob=infer_loader_v2(model,validation);result=metrics(vp,val.labels,val.valid)
        score=result['position_mean_m']+.5*result['velocity_mean_mps']
        row=dict(epoch=epoch+1,train_loss=loss_sum/len(train),validation=result,
            validity_accuracy=float(np.mean((prob>=.5)==(val.valid>0))),elapsed_s=time.time()-started)
        history.append(row)
        if score<best:
            best=score;best_epoch=epoch+1
            torch.save(dict(state_dict=model.state_dict(),seed=seed,epoch=best_epoch,validation_score=best,
                architecture=ARCHITECTURE_V2,input_shape=[2,20,96,21,21],
                priors='Fixed calibration building roof height and radar acquisition grid; no runtime geometry/pose input',
                label='ground-target relative position and platform velocity, body forward/right/up at latest CPI end'),root/f'seed_{seed}.tmp.pt')
            (root/f'seed_{seed}.tmp.pt').replace(root/f'seed_{seed}.pt')
        write_json(root/f'history_{seed}.json',history)
        print(json.dumps(dict(seed=seed,**row,best_epoch=best_epoch)),flush=True)
        if epoch+1-best_epoch>=args['patience']:break
    return dict(seed=seed,best_epoch=best_epoch,validation_score=best,elapsed_s=time.time()-started)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--epochs',type=int,default=200);p.add_argument('--patience',type=int,default=35)
    p.add_argument('--threads',type=int,default=16);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--seeds',type=int,nargs='+',default=[11,23,37]);p.add_argument('--parallel-seeds',type=int,default=3)
    args=vars(p.parse_args());root=local_path(args['output']);data=local_path(args['data'])
    if read_json(data/'status.json')['status']!='complete':raise ValueError('feature cache incomplete')
    root.mkdir(parents=True,exist_ok=False)
    files=[local_path(__file__)]+[local_path(f'backend/radar_platform/navigation/{name}.py') for name in
           ['features_v2','model_v2','learning_v2','learning','observation','inference']]
    write_json(root/'source_sha256.json',{str(f.relative_to(local_path('.'))):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    with zipfile.ZipFile(root/'training_source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for f in files:archive.write(f,str(f.relative_to(local_path('.'))))
    dataset_protocol=read_json(data/'protocol.json')
    development=any(r.get('formal_split')=='train' and r['split']=='val' for r in dataset_protocol['records'])
    write_json(root/'training_protocol.json',{**args,'architecture':ARCHITECTURE_V2,'device':'cpu','torch_version':torch.__version__,
        'development_only':development,'dataset_scope':dataset_protocol.get('scope',dataset_protocol.get('split','declared physical dataset')),
        'dataset_protocol_sha256':digest(dataset_protocol),'dataset_counts':dataset_protocol['counts'],
        'planned_full_counts':dataset_protocol.get('planned_full_counts',dataset_protocol['counts']),
        'selection':'validation mean position error plus 0.5 times mean velocity error',
        'test_accessed':False,'input':'Numeric RDA spectra and native Doppler moments only',
        'spatial_supervision':'Training labels converted to a Gaussian around the fixed calibration building roof center'})
    write_json(root/'status.json',dict(status='training',pid=os.getpid()))
    with ProcessPoolExecutor(max_workers=args['parallel_seeds'],mp_context=get_context('spawn')) as pool:
        summaries=[]
        for f in as_completed([pool.submit(train_one,(args,s)) for s in args['seeds']]):
            summaries.append(f.result());write_json(root/'training_summary.json',summaries)
    torch.set_num_threads(args['threads']);val=RadarDatasetV2(data,'val');loader=DataLoader(val,batch_size=args['batch_size'])
    predictions=[];probabilities=[]
    for seed in args['seeds']:
        model=RadarNavigationHeatmap();model.load_state_dict(torch.load(root/f'seed_{seed}.pt',map_location='cpu',weights_only=True)['state_dict'])
        pred,prob=infer_loader_v2(model,loader);predictions.append(pred);probabilities.append(prob)
    ensemble=np.mean(predictions,axis=0);valid=val.valid>0
    error=np.linalg.norm(ensemble[:,:3]-val.labels[:,:3],axis=1);spread=np.linalg.norm(np.std(predictions,axis=0)[:,:3],axis=1)
    floor=max(.25,float(np.median(error[valid])));multiplier=float(np.percentile(error[valid]/(spread[valid]+floor),80))
    write_json(root/'uncertainty.json',dict(floor_m=floor,multiplier=multiplier,maximum_error_scale_m=3.,valid_probability_threshold=.5,
        description='Validation residual scale, not certified probabilistic confidence'))
    result=metrics(ensemble,val.labels,val.valid);result['validity_accuracy']=float(np.mean((np.mean(probabilities,axis=0)>=.5)==valid))
    write_json(root/'metrics.json',dict(val=result))
    np.savez_compressed(root/'val_predictions.npz',prediction=ensemble,truth=val.labels,valid=valid,per_seed=np.asarray(predictions),probability=np.mean(probabilities,axis=0))
    shutil.copyfile(data/'acquisition_axes.npz',root/'acquisition_axes.npz')
    write_json(root/'status.json',dict(status='complete',seeds=args['seeds'],test_evaluated=False))
    print(json.dumps(dict(val=result)),flush=True)


if __name__=='__main__':main()
