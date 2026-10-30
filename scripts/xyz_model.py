"""Compact two-frame 3D CNN; outputs North/East/Up in meters directly."""
import torch
from torch import nn


class XYZNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('input_mean',torch.zeros(1,2,1,1,1))
        self.register_buffer('input_std',torch.ones(1,2,1,1,1))
        self.register_buffer('xyz_center',torch.tensor([0.,0.,32.]))
        self.register_buffer('xyz_scale',torch.tensor([18.,12.,10.]))
        self.encoder=nn.Sequential(
            nn.Conv3d(2,8,3,stride=2,padding=1),nn.GroupNorm(4,8),nn.SiLU(),
            nn.Conv3d(8,16,3,stride=2,padding=1),nn.GroupNorm(4,16),nn.SiLU(),
            nn.Conv3d(16,24,3,stride=2,padding=1),nn.GroupNorm(4,24),nn.SiLU())
        # Preserve spatial arrangement; global pooling would discard position cues.
        self.head=nn.Sequential(nn.Flatten(),nn.Linear(24*8*2*3,128),nn.SiLU(),nn.Dropout(.05),nn.Linear(128,3))

    def forward(self,inputs):
        x=(inputs-self.input_mean)/self.input_std
        return self.xyz_center+self.xyz_scale*self.head(self.encoder(x))
