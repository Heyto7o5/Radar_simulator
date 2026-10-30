"""Training-only dataset access; inference lives in a separate module."""
import numpy as np
import torch
from torch.utils.data import Dataset

from .io import local_path,read_json


class RadarDataset(Dataset):
    def __init__(self,root,split):
        self.root=local_path(root)
        self.records=[r for r in read_json(self.root/'protocol.json')['records'] if r['split']==split]
        self.labels=[];self.valid=[]
        for record in self.records:
            truth=read_json(self.root/'cases'/record['id']/'truth.json')
            self.labels.append(truth['state_body']);self.valid.append(truth['valid_landmark_observation'])
        self.labels=np.asarray(self.labels,np.float32);self.valid=np.asarray(self.valid,np.float32)

    def __len__(self):return len(self.records)

    def __getitem__(self,index):
        feature=np.load(self.root/'cases'/self.records[index]['id']/'input.npy',allow_pickle=False).astype(np.float32)
        return torch.from_numpy(feature),torch.tensor(.25),torch.from_numpy(self.labels[index]),torch.tensor(self.valid[index])


def metrics(prediction,labels,valid=None):
    prediction=np.asarray(prediction);labels=np.asarray(labels)
    if valid is not None:prediction=prediction[np.asarray(valid)>0];labels=labels[np.asarray(valid)>0]
    if not len(labels):return dict(cases=0)
    error=prediction-labels;position=np.linalg.norm(error[:,:3],axis=1);velocity=np.linalg.norm(error[:,3:],axis=1)
    return dict(cases=len(labels),axis_mae=np.mean(abs(error),axis=0).tolist(),
        axis_rmse=np.sqrt(np.mean(error**2,axis=0)).tolist(),
        position_mean_m=float(position.mean()),position_p90_m=float(np.percentile(position,90)),
        velocity_mean_mps=float(velocity.mean()),velocity_p90_mps=float(np.percentile(velocity,90)))


@torch.inference_mode()
def infer_loader(model,loader):
    model.eval();predictions=[];probabilities=[]
    for x,dt,_,_ in loader:
        prediction,logit=model(x,dt)
        predictions.append(prediction.numpy());probabilities.append(torch.sigmoid(logit).numpy())
    return np.concatenate(predictions),np.concatenate(probabilities)
