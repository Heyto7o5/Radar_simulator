"""Two radar-volume encoders and a learned temporal position/velocity head."""
import torch
from torch import nn

ARCHITECTURE='RadarNavigationNet_v1'


class RadarNavigationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('input_mean',torch.zeros(1,16,1,1,1))
        self.register_buffer('input_std',torch.ones(1,16,1,1,1))
        self.register_buffer('output_center',torch.tensor([0.,0.,-35.,0.,0.,0.]))
        self.register_buffer('output_scale',torch.tensor([35.,35.,20.,6.,6.,2.]))
        self.encoder=nn.Sequential(
            nn.Conv3d(16,8,3,stride=2,padding=1),nn.GroupNorm(4,8),nn.SiLU(),
            nn.Conv3d(8,16,3,stride=2,padding=1),nn.GroupNorm(4,16),nn.SiLU(),
            nn.Conv3d(16,24,3,stride=2,padding=1),nn.GroupNorm(4,24),nn.SiLU(),
            nn.Flatten(),nn.Linear(24*12*3*3,128),nn.SiLU())
        self.temporal=nn.Sequential(nn.Linear(257,128),nn.SiLU(),nn.Dropout(.05),nn.Linear(128,7))

    def forward(self,features,interval_s):
        b,t,d,r,e,a=features.shape
        encoded=self.encoder((features.reshape(b*t,d,r,e,a)-self.input_mean)/self.input_std)
        fused=torch.cat([encoded.reshape(b,t*128),interval_s.reshape(b,1)],dim=1)
        raw=self.temporal(fused)
        return self.output_center+self.output_scale*raw[:,:6],raw[:,6]
