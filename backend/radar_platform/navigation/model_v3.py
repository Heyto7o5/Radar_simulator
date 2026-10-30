"""V2 landmark decoder with a zero-anchored, learned radar velocity calibration."""
import torch
from torch import nn
from torch.nn import functional as F

from .model_v2 import RadarNavigationHeatmap

ARCHITECTURE_V3='RadarNavigationHeatmap_v3'


class RadarNavigationAnchored(RadarNavigationHeatmap):
    def __init__(self,coordinates=None,roof_height_m=5.5):
        super().__init__(coordinates,roof_height_m)
        self.register_buffer('velocity_feature_scale',torch.ones(387))
        self.register_buffer('velocity_correction',torch.zeros(387,3))
        self.register_buffer('velocity_time_ratio',torch.tensor(.051128))

    def encode_shared(self,features):
        batch,frames,channels,r,e,a=features.shape
        x=(features.reshape(batch*frames,channels,r,e,a)-self.input_mean)/self.input_std
        one=self.first(x);two=self.second(one);three=self.third(two)
        def temporal(tensor):
            t=tensor.reshape(batch,frames,*tensor.shape[1:])
            return torch.cat([t[:,0],t[:,1]],dim=1)
        deep=temporal(three)
        latent=self.state_head[:-1](deep)
        return one,two,deep,latent,temporal

    def velocity_design(self,latent,baseline):
        current=baseline[:,-1]+self.velocity_time_ratio*(baseline[:,-1]-baseline[:,0])
        augmented=torch.cat([latent,torch.ones((len(latent),1),device=latent.device,dtype=latent.dtype)],dim=1)
        design=torch.einsum('bi,bj->bij',current,augmented).flatten(1)
        return current,design

    def forward(self,features,baseline_velocity):
        one,two,deep,latent,temporal=self.encode_shared(features)
        decoded=self.bottleneck(deep)
        decoded=self.decode_second(torch.cat([F.interpolate(decoded,size=two.shape[-3:],mode='trilinear',align_corners=True),temporal(two)],dim=1))
        decoded=self.decode_first(torch.cat([F.interpolate(decoded,size=one.shape[-3:],mode='trilinear',align_corners=True),temporal(one)],dim=1))
        logits=F.interpolate(self.spatial_score(decoded),size=(96,21,21),mode='trilinear',align_corners=True).flatten(1)
        probability=torch.softmax(logits,dim=1)
        raw=self.state_head[-1](latent)
        position=probability@self.coordinates.reshape(-1,3)+2.5*torch.tanh(raw[:,:3])
        position=position-torch.stack([self.roof_height_m*0,self.roof_height_m*0,self.roof_height_m])
        current,design=self.velocity_design(latent,baseline_velocity)
        velocity=current+(design/self.velocity_feature_scale)@self.velocity_correction
        return torch.cat([position,velocity],dim=1),raw[:,6],logits
