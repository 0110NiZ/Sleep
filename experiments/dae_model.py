"""Residual 1-D convolutional denoising autoencoder; input/output in microvolts."""
import torch
from torch import nn

class DAE(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.register_buffer('scale', torch.tensor(float(scale)))
        self.encoder=nn.Sequential(nn.Conv1d(1,16,9,2,4),nn.GELU(),
            nn.Conv1d(16,32,9,2,4),nn.GELU(),nn.Conv1d(32,64,9,2,4),nn.GELU())
        self.bottleneck=nn.Sequential(nn.Conv1d(64,64,7,padding=12,dilation=4),nn.GELU())
        self.decoder=nn.Sequential(nn.ConvTranspose1d(64,32,8,2,3),nn.GELU(),
            nn.ConvTranspose1d(32,16,8,2,3),nn.GELU(),nn.ConvTranspose1d(16,1,8,2,3))
        # An identity initial mapping preserves the signal before learning a correction.
        nn.init.zeros_(self.decoder[-1].weight);nn.init.zeros_(self.decoder[-1].bias)

    def forward(self,x):
        z=x/self.scale
        return (z+self.decoder(self.bottleneck(self.encoder(z))))*self.scale
