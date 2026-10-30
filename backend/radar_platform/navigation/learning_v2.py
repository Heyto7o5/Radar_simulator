"""Training access for cached v2 radar features; labels are separate files."""
import numpy as np
import torch

from .learning import RadarDataset


class RadarDatasetV2(RadarDataset):
    def __getitem__(self,index):
        folder=self.root/'cases'/self.records[index]['id']
        features=np.load(folder/'input.npy',allow_pickle=False).astype(np.float32)
        velocity=np.load(folder/'velocity_baseline.npy',allow_pickle=False)
        return torch.from_numpy(features),torch.from_numpy(velocity),torch.from_numpy(self.labels[index]),torch.tensor(self.valid[index])


@torch.inference_mode()
def infer_loader_v2(model,loader):
    model.eval();predictions=[];probabilities=[]
    for x,baseline,_,_ in loader:
        prediction,logit,_=model(x,baseline)
        predictions.append(prediction.numpy());probabilities.append(torch.sigmoid(logit).numpy())
    return np.concatenate(predictions),np.concatenate(probabilities)
