#!/usr/bin/env python3
"""Fit a zero-anchored radar velocity head using train labels and validation only."""
import argparse
import hashlib
import json
import os
import shutil
import zipfile
import numpy as np
import torch
from torch.utils.data import DataLoader

from radar_platform.navigation.io import local_path,read_json,write_json
from radar_platform.navigation.learning import metrics
from radar_platform.navigation.learning_v2 import RadarDatasetV2
from radar_platform.navigation.model_v3 import RadarNavigationAnchored,ARCHITECTURE_V3


@torch.inference_mode()
def descriptors(model,dataset,batch_size):
    model.eval();bases=[];designs=[]
    for x,b,_,_ in DataLoader(dataset,batch_size=batch_size,shuffle=False):
        _,_,_,latent,_=model.encode_shared(x)
        base,design=model.velocity_design(latent,b)
        bases.append(base.numpy());designs.append(design.numpy())
    return np.concatenate(bases),np.concatenate(designs)


@torch.inference_mode()
def predictions(model,dataset,batch_size):
    outputs=[];probabilities=[];model.eval()
    for x,b,_,_ in DataLoader(dataset,batch_size=batch_size,shuffle=False):
        value,valid,_=model(x,b);outputs.append(value.numpy());probabilities.append(torch.sigmoid(valid).numpy())
    return np.concatenate(outputs),np.concatenate(probabilities)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--source-models',required=True);p.add_argument('--output',required=True)
    p.add_argument('--threads',type=int,default=16);p.add_argument('--batch-size',type=int,default=16)
    args=p.parse_args();torch.set_num_threads(args.threads)
    source=local_path(args.source_models);root=local_path(args.output);data=local_path(args.data)
    if read_json(source/'status.json')['status']!='complete':raise ValueError('source learning incomplete')
    root.mkdir(parents=True,exist_ok=False)
    protocol=read_json(source/'training_protocol.json')
    if protocol['architecture']!='RadarNavigationHeatmap_v2':raise ValueError('calibration requires the frozen v2 spatial model')
    protocol.update(architecture=ARCHITECTURE_V3,source_model_directory=str(source),output=str(root),data=str(data),
        velocity_head='Ridge correction on products of native-Doppler velocity and frozen radar embeddings; no additive velocity intercept',
        velocity_selection='Train ridge coefficients; select alpha on validation only; formal test unopened')
    write_json(root/'training_protocol.json',protocol);write_json(root/'status.json',dict(status='calibrating',pid=os.getpid()))
    train=RadarDatasetV2(data,'train');val=RadarDatasetV2(data,'val');outputs=[];probabilities=[];reports=[]
    cfg=read_json(data/'base_config.json');w=cfg['waveform'];a=cfg['array']
    period=a['loop_period_s'] or a['num_tx']*w['slot_period_s']
    offsets=np.array(a['tx_offsets_s'] or [k*w['slot_period_s'] for k in range(a['num_tx'])])
    end=(w['loops']-1)*period+offsets.max()+w['adc_start_s']+w['samples']/w['sample_rate_hz']
    midpoint=(w['loops']-1)*period/2+offsets.mean()+w['adc_start_s']+w['samples']/(2*w['sample_rate_hz'])
    ratio=(end-midpoint)/.25
    for seed in protocol['seeds']:
        checkpoint=torch.load(source/f'seed_{seed}.pt',map_location='cpu',weights_only=True)
        model=RadarNavigationAnchored();missing,unexpected=model.load_state_dict(checkpoint['state_dict'],strict=False)
        assert set(missing)=={'velocity_feature_scale','velocity_correction','velocity_time_ratio'} and not unexpected
        model.velocity_time_ratio.fill_(ratio)
        base,x=descriptors(model,train,args.batch_size);vbase,vx=descriptors(model,val,args.batch_size)
        take=train.valid>0;vtake=val.valid>0
        scale=np.maximum(np.sqrt(np.mean(x[take].astype(np.float64)**2,axis=0)),1e-3)
        xx=x[take]/scale;yy=train.labels[take,3:]-base[take];vxx=vx/scale
        matrix=xx.T@xx/len(xx);rhs=xx.T@yy/len(xx)
        candidates=[];best=None
        for alpha in [1e-4,1e-3,.01,.1,1.,10.]:
            coefficient=np.linalg.solve(matrix+alpha*np.eye(len(scale)),rhs)
            velocity=vbase+vxx@coefficient
            error=np.linalg.norm(velocity[vtake]-val.labels[vtake,3:],axis=1)
            score=float(error.mean());candidates.append(dict(alpha=alpha,velocity_mean_mps=score,velocity_p90_mps=float(np.percentile(error,90))))
            if best is None or score<best[0]:best=(score,alpha,coefficient)
        model.velocity_feature_scale.copy_(torch.from_numpy(scale.astype(np.float32)))
        model.velocity_correction.copy_(torch.from_numpy(best[2].astype(np.float32)))
        value,prob=predictions(model,val,args.batch_size);outputs.append(value);probabilities.append(prob)
        validation=metrics(value,val.labels,val.valid)
        new={**checkpoint,'architecture':ARCHITECTURE_V3,'state_dict':model.state_dict(),
             'source_validation_score':checkpoint['validation_score'],
             'validation_score':validation['position_mean_m']+.5*validation['velocity_mean_mps'],
             'velocity_calibration_alpha':best[1],'source_weight_sha256':hashlib.sha256((source/f'seed_{seed}.pt').read_bytes()).hexdigest()}
        torch.save(new,root/f'seed_{seed}.pt')
        reports.append(dict(seed=seed,selected_alpha=best[1],candidates=candidates,validation=validation))
        write_json(root/'velocity_calibration.json',reports);print(json.dumps(reports[-1]),flush=True)
    ensemble=np.mean(outputs,axis=0);visible=val.valid>0
    # Spatial decoder/positions are exactly the frozen v2 model, so retain its position calibration.
    for name in ['acquisition_axes.npz','uncertainty.json']:shutil.copyfile(source/name,root/name)
    result=metrics(ensemble,val.labels,val.valid);result['validity_accuracy']=float(np.mean((np.mean(probabilities,axis=0)>=.5)==visible))
    write_json(root/'metrics.json',dict(val=result))
    np.savez_compressed(root/'val_predictions.npz',prediction=ensemble,truth=val.labels,valid=visible,
        per_seed=np.asarray(outputs),probability=np.mean(probabilities,axis=0))
    with np.load(source/'val_predictions.npz',allow_pickle=False) as previous:
        difference=float(np.max(abs(previous['prediction'][:,:3]-ensemble[:,:3])))
        np.testing.assert_allclose(previous['prediction'][:,:3],ensemble[:,:3],rtol=1e-5,atol=1e-4)
    write_json(root/'spatial_preservation.json',dict(max_position_difference_m=difference,spatial_weights_unchanged=True))
    write_json(root/'acquisition_timing.json',dict(frame_interval_s=.25,cpi_duration_s=float(end),velocity_time_ratio=float(ratio)))
    files=[local_path(__file__)]+[local_path(f'backend/radar_platform/navigation/{name}.py') for name in ['model_v2','model_v3','features_v2','inference','observation']]
    write_json(root/'source_sha256.json',{str(f.relative_to(local_path('.'))):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    with zipfile.ZipFile(root/'training_source.zip','x',zipfile.ZIP_DEFLATED) as archive:
        for f in files:archive.write(f,str(f.relative_to(local_path('.'))))
    write_json(root/'status.json',dict(status='complete',seeds=protocol['seeds'],test_evaluated=False))
    print(json.dumps(dict(val=result,spatial_difference=difference)),flush=True)


if __name__=='__main__':main()
