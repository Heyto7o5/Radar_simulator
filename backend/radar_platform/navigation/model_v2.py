"""Temporal radar heatmap with explicit range-angle coordinate decoding."""
import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE_V2='RadarNavigationHeatmap_v2'


def block(input_channels,output_channels,stride=1):
    return nn.Sequential(nn.Conv3d(input_channels,output_channels,3,stride=stride,padding=1),
                         nn.GroupNorm(4,output_channels),nn.SiLU())


class RadarNavigationHeatmap(nn.Module):
    def __init__(self,coordinates=None,roof_height_m=5.5):
        super().__init__()
        if coordinates is None:coordinates=torch.zeros(96,21,21,3)
        self.register_buffer('coordinates',torch.as_tensor(coordinates,dtype=torch.float32))
        self.register_buffer('roof_height_m',torch.tensor(float(roof_height_m)))
        self.register_buffer('input_mean',torch.zeros(1,20,1,1,1))
        self.register_buffer('input_std',torch.ones(1,20,1,1,1))
        self.first=block(20,8,2);self.second=block(8,16,2);self.third=block(16,24,2)
        self.bottleneck=nn.Sequential(nn.Conv3d(48,24,1),nn.GroupNorm(4,24),nn.SiLU())
        self.decode_second=block(56,16);self.decode_first=block(32,8)
        self.spatial_score=nn.Conv3d(8,1,1)
        self.state_head=nn.Sequential(nn.Flatten(),nn.Linear(48*12*3*3,128),nn.SiLU(),
                                      nn.Dropout(.05),nn.Linear(128,7))
        nn.init.zeros_(self.state_head[-1].weight);nn.init.zeros_(self.state_head[-1].bias)

    def forward(self,features,baseline_velocity):
        batch,frames,channels,r,e,a=features.shape
        x=(features.reshape(batch*frames,channels,r,e,a)-self.input_mean)/self.input_std
        one=self.first(x);two=self.second(one);three=self.third(two)
        def temporal(tensor):
            t=tensor.reshape(batch,frames,*tensor.shape[1:])
            return torch.cat([t[:,0],t[:,1]],dim=1)
        deep=temporal(three)
        decoded=self.bottleneck(deep)
        decoded=self.decode_second(torch.cat([F.interpolate(decoded,size=two.shape[-3:],mode='trilinear',align_corners=True),temporal(two)],dim=1))
        decoded=self.decode_first(torch.cat([F.interpolate(decoded,size=one.shape[-3:],mode='trilinear',align_corners=True),temporal(one)],dim=1))
        logits=F.interpolate(self.spatial_score(decoded),size=(96,21,21),mode='trilinear',align_corners=True).flatten(1)
        probability=torch.softmax(logits,dim=1)
        radar_position=probability@self.coordinates.reshape(-1,3)
        raw=self.state_head(deep)
        position=radar_position+2.5*torch.tanh(raw[:,:3])
        position=position-torch.stack([self.roof_height_m*0,self.roof_height_m*0,self.roof_height_m])
        velocity=baseline_velocity[:,-1]+torch.tensor([6.,6.,2.],device=raw.device)*raw[:,3:6]
        return torch.cat([position,velocity],dim=1),raw[:,6],logits

    def spatial_target(self,state):
        roof=state[:,:3]+torch.stack([self.roof_height_m*0,self.roof_height_m*0,self.roof_height_m])
        difference=(self.coordinates[None]-roof[:,None,None,None,:])/torch.tensor([3.,3.,1.5],device=state.device)
        return torch.softmax(-.5*torch.sum(difference**2,dim=-1).flatten(1),dim=1)
