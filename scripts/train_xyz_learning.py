#!/usr/bin/env python3
"""Train joint XYZ regression, select on validation only, then evaluate once."""
import argparse,copy,hashlib,json,random,time,zipfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
from xyz_model import XYZNet


def write(p,obj):p.write_text(json.dumps(obj,indent=2)+'\n')


def metrics(pred,truth):
    error=pred-truth
    r2=1-np.sum(error**2,axis=0)/np.maximum(np.sum((truth-truth.mean(axis=0))**2,axis=0),1e-12)
    norms=np.linalg.norm(error,axis=1)
    return dict(axis_mae_m=np.mean(abs(error),axis=0).tolist(),axis_rmse_m=np.sqrt(np.mean(error**2,axis=0)).tolist(),
                axis_r2=r2.tolist(),mean_3d_error_m=float(norms.mean()),p90_3d_error_m=float(np.percentile(norms,90)),
                prediction_axis_std_m=pred.std(axis=0).tolist(),truth_axis_std_m=truth.std(axis=0).tolist(),
                cases=len(truth))


@torch.inference_mode()
def infer(model,x,batch_size=32):
    model.eval()
    return torch.cat([model(x[k:k+batch_size]) for k in range(0,len(x),batch_size)]).cpu().numpy()


def train_seed(seed,x,y,vx,vy,root,epochs,patience,batch_size):
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    model=XYZNet()
    model.input_mean.copy_(x.mean(dim=(0,2,3,4),keepdim=True))
    model.input_std.copy_(x.std(dim=(0,2,3,4),keepdim=True).clamp_min(1e-6))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs,eta_min=1e-5)
    loader=DataLoader(TensorDataset(x,y),batch_size=batch_size,shuffle=True,num_workers=0)
    best=float('inf');best_epoch=-1;history=[];started=time.perf_counter()
    for epoch in range(epochs):
        model.train();loss_sum=0.
        for bx,by in loader:
            optimizer.zero_grad(set_to_none=True)
            pred=model(bx);loss=torch.mean(((pred-by)/model.xyz_scale)**2)
            if not torch.isfinite(loss):raise ValueError('nonfinite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step();loss_sum+=float(loss)*len(bx)
        scheduler.step()
        vp=infer(model,vx);score=metrics(vp,vy.numpy())['mean_3d_error_m']
        row=dict(epoch=epoch+1,train_normalized_mse=loss_sum/len(x),val_mean_3d_error_m=score)
        history.append(row)
        if score<best:
            best=score;best_epoch=epoch+1
            torch.save(dict(state_dict=model.state_dict(),seed=seed,epoch=best_epoch,
                            validation_mean_3d_error_m=best,architecture='XYZNet_v1',
                            coordinate_order=['north_m','east_m','up_m'],label_time_s=0,
                            input_shape=[2,64,16,24]),root/f'seed_{seed}.tmp.pt')
            (root/f'seed_{seed}.tmp.pt').replace(root/f'seed_{seed}.pt')
        if epoch%10==0:print(json.dumps(dict(seed=seed,**row,best_val=best)),flush=True)
        if epoch+1-best_epoch>=patience:break
    checkpoint=torch.load(root/f'seed_{seed}.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(checkpoint['state_dict']);model.eval()
    write(root/f'history_{seed}.json',history)
    return model,dict(seed=seed,best_epoch=best_epoch,best_val_mean_3d_error_m=best,
                      epochs_run=len(history),training_seconds=time.perf_counter()-started,
                      parameters=sum(p.numel() for p in model.parameters()))


def fit_baselines(x,y,vx,vy):
    a=x.reshape(len(x),-1).astype(np.float32)
    b=vx.reshape(len(vx),-1).astype(np.float32)
    # Amplitude-pattern nearest neighbors, k selected solely on validation.
    an=a/np.linalg.norm(a,axis=1,keepdims=True);bn=b/np.linalg.norm(b,axis=1,keepdims=True)
    dist=np.maximum(0,2-2*bn@an.T)
    candidates=[]
    for k in [1,4,8]:
        ix=np.argsort(dist,axis=1)[:,:k];weights=1/np.maximum(np.take_along_axis(dist,ix,axis=1),1e-6)**2
        pred=np.sum(y[ix]*weights[:,:,None],axis=1)/weights.sum(axis=1,keepdims=True)
        candidates.append((metrics(pred,vy)['mean_3d_error_m'],k))
    chosen=min(candidates)[1]
    # Dual ridge is a linear, train-only alternative to the CNN.
    center=a.mean(axis=0);ac=a-center;yc=y-y.mean(axis=0);kernel=ac@ac.T/a.shape[1]
    vk=(b-center)@ac.T/a.shape[1]
    ridge=[]
    for alpha in [1e-5,1e-4,1e-3,1e-2,.1,1.]:
        coef=np.linalg.solve(kernel+alpha*np.eye(len(a)),yc)
        pred=vk@coef+y.mean(axis=0)
        ridge.append((metrics(pred,vy)['mean_3d_error_m'],alpha,coef))
    score,alpha,coef=min(ridge,key=lambda q:q[0])
    return dict(neighbors=chosen,normalized_train=an,labels=y,feature_center=center,centered_train=ac,
                ridge_alpha=alpha,ridge_coef=coef,label_mean=y.mean(axis=0),
                validation_knn=candidates,validation_ridge=[(q[0],q[1]) for q in ridge])


def baseline_predict(model,x):
    flat=x.reshape(len(x),-1)
    norm=flat/np.linalg.norm(flat,axis=1,keepdims=True)
    d=np.maximum(0,2-2*norm@model['normalized_train'].T)
    ix=np.argsort(d,axis=1)[:,:model['neighbors']]
    w=1/np.maximum(np.take_along_axis(d,ix,axis=1),1e-6)**2
    knn=np.sum(model['labels'][ix]*w[:,:,None],axis=1)/w.sum(axis=1,keepdims=True)
    ridge=(flat-model['feature_center'])@model['centered_train'].T/flat.shape[1]@model['ridge_coef']+model['label_mean']
    return {'train_mean':np.tile(model['label_mean'],(len(x),1)),'knn':knn,'ridge':ridge}


def seed_worker(job):
    seed,data,root,epochs,patience,batch_size,threads=job
    torch.set_num_threads(threads)
    arrays=[torch.from_numpy(np.load(Path(data)/name)) for name in ['train_x.npy','train_y.npy','val_x.npy','val_y.npy']]
    _,info=train_seed(seed,*arrays,Path(root),epochs,patience,batch_size)
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=300);p.add_argument('--patience',type=int,default=50)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--seeds',type=int,nargs='+',default=[11,23,37])
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--parallel-seeds',type=int,default=3)
    args=p.parse_args();torch.set_num_threads(args.threads)
    if json.loads((args.data/'status.json').read_text())['status']!='complete':raise ValueError('dataset incomplete')
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),Path(__file__).with_name('xyz_model.py'),Path('backend/radar_platform/xyz_features.py')]
    with zipfile.ZipFile(root/'training_source.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,str(f))
    write(root/'source_sha256.json',{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    write(root/'training_protocol.json',dict(epochs=args.epochs,patience=args.patience,seeds=args.seeds,
        checkpoint_selection='minimum validation mean 3D Euclidean error; test not used',
        torch_version=str(torch.__version__),device='cpu',threads=args.threads,input='two numeric RDA magnitude frames',
        coordinate_order=['north_m','east_m','up_m'],label_time_s=0))
    train_x=np.load(args.data/'train_x.npy');train_y=np.load(args.data/'train_y.npy')
    val_x=np.load(args.data/'val_x.npy');val_y=np.load(args.data/'val_y.npy')
    x,y,vx,vy=map(torch.from_numpy,(train_x,train_y,val_x,val_y))
    models=[];training=[]
    with ProcessPoolExecutor(max_workers=min(args.parallel_seeds,len(args.seeds)),mp_context=get_context('spawn')) as pool:
        jobs=[(seed,str(args.data),str(root),args.epochs,args.patience,args.batch_size,args.threads) for seed in args.seeds]
        for future in as_completed([pool.submit(seed_worker,job) for job in jobs]):
            training.append(future.result());write(root/'training_summary.json',training)
    for seed in args.seeds:
        model=XYZNet();checkpoint=torch.load(root/f'seed_{seed}.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(checkpoint['state_dict']);model.eval();models.append(model)
    baselines=fit_baselines(train_x,train_y,val_x,val_y)
    write(root/'baseline_selection.json',{k:baselines[k] for k in ['neighbors','ridge_alpha','validation_knn','validation_ridge']})
    results={}
    for split in ['test','grid5','building_height','noise10']:
        sx=np.load(args.data/f'{split}_x.npy');sy=np.load(args.data/f'{split}_y.npy')
        predictions=baseline_predict(baselines,sx)
        for seed,model in zip(args.seeds,models):predictions[f'cnn_seed_{seed}']=infer(model,torch.from_numpy(sx))
        predictions['cnn_ensemble']=np.mean([predictions[f'cnn_seed_{s}'] for s in args.seeds],axis=0)
        results[split]={name:metrics(pred,sy) for name,pred in predictions.items()}
        if split=='test':
            permutation=np.random.default_rng(888).permutation(len(sx))
            results[split]['shuffled_input_control']=metrics(predictions['cnn_ensemble'][permutation],sy)
        np.savez(root/f'{split}_predictions.npz',truth_xyz_m=sy,**predictions)
    write(root/'metrics.json',results)
    with np.load(args.data/'cases'/json.loads((args.data/'protocol.json').read_text())['records'][0]['id']/'rda.npz') as f:
        np.savez(root/'acquisition_axes.npz',**{k:f[k] for k in ['range_m','velocity_mps','azimuth_deg','frame_times_s']})
    write(root/'status.json',dict(status='complete',trained_seeds=args.seeds,test_cases=len(np.load(args.data/'test_y.npy'))))
    print(json.dumps(dict(training=training,test_metrics=results['test'])),flush=True)


if __name__=='__main__':main()
