# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import torch
import torch.nn as nn

from model.base import BaseModule
from model.encoder import MelEncoder
from model.postnet import PostNet
from model.diffusion_IDMap import Diffusion
from model.utils import sequence_mask, fix_len_compatibility, mse_loss
from model.GST import GST

class informer(nn.Module):
    def __init__(self, spk_dim, channels=1):
        super(informer, self).__init__()
        self.linear1 = nn.Linear(spk_dim, 512)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu1 = nn.ReLU()

        self.linear2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu2 = nn.ReLU()
        self.condition_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)
        
        self.linear3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(channels)
        self.relu3 = nn.ReLU()
        self.linear = nn.Linear(512, spk_dim)


    def forward(self, x):
        out = self.linear1(x)

        out = self.bn1(out)
        out = self.relu1(out)

        out = self.linear2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.linear3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.linear(out)
        
        return out
    
# "average voice" encoder as the module parameterizing the diffusion prior
class encoder_g(nn.Module):
    def __init__(self, spk_dim, channels=1):
        super(encoder_g, self).__init__()
        self.linear1 = nn.Linear(spk_dim, 512)

    def forward(self, x):
        out = self.linear1(x)
        return out



# the whole voice conversion model consisting of the "average voice" encoder 
# and the diffusion-based speaker-conditional decoder
class DiffVC(BaseModule):
    def __init__(self, n_feats, channels, filters, heads, layers, kernel, 
                 dropout, window_size, enc_dim, spk_dim, use_ref_t, dec_dim, 
                 beta_min, beta_max):
        super(DiffVC, self).__init__()
        self.n_feats = n_feats
        self.channels = channels
        self.filters = filters
        self.heads = heads
        self.layers = layers
        self.kernel = kernel
        self.dropout = dropout
        self.window_size = window_size
        self.enc_dim = enc_dim
        self.spk_dim = spk_dim
        self.use_ref_t = use_ref_t
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.encoder = informer(spk_dim=spk_dim)
        self.encoder_g = encoder_g(spk_dim=spk_dim)
        self.decoder = Diffusion(n_feats, dec_dim, spk_dim, use_ref_t, 
                                 beta_min, beta_max)
        self.cov = torch.nn.Conv1d(in_channels=256, out_channels=80, kernel_size=3, stride=1, padding=1)
        
        # self.cov = torch.nn.Linear(256, 80)

    def load_encoder(self, enc_path):
        enc_dict = torch.load(enc_path, map_location=lambda loc, storage: loc)
        self.encoder.load_state_dict(enc_dict, strict=False)
        
    def forward_(self, x, n_timesteps, g, mode='ml'):
        """
        Generates mel-spectrogram from source mel-spectrogram conditioned on
        target speaker embedding. Returns:
            1. 'average voice' encoder outputs
            2. decoder outputs
        
        Args:
            x (torch.Tensor): batch of source speaker embedding.
            g (torch.Tensor): batch of source samples.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            mode (string, optional): sampling method. Can be one of:
              'pf' - probability flow sampling (Euler scheme for ODE)
              'em' - Euler-Maruyama SDE solver
              'ml' - Maximum Likelihood SDE solver
        """

        if len(g.shape) == 3:
            g = g.squeeze()
        g = self.encoder_g(g)
        mean = self.encoder(x)

        b = x.shape[0]
        max_length_new = max(x.shape)
        mean_new = torch.zeros((b, self.n_feats, max_length_new), dtype=x.dtype, 
                                device=x.device)

        z = mean_new
        z += torch.rand_like(mean_new, device=mean_new.device)*2-1
        # z += torch.randn_like(mean_new, device=mean_new.device)

        y = self.decoder.forward_(z, 1, mean, g, n_timesteps, mode)
        return y
    

    # @torch.no_grad()
    def compute_loss(self, x, g):
        """
        Computes diffusion (score matching) loss.
            
        Args:
            x (torch.Tensor): batch of source speaker embedding
            g (torch.Tensor): samples
        """
        # g = self.fc(g).transpose(1,2).expand(-1,-1,128)
        b = g.shape[0]
        mask = torch.ones(b, 1 , 512).cuda()
        if len(g.shape) == 3:
            g = g.squeeze()
        g = self.encoder_g(g)    
        mean = self.encoder(x)
        diff_loss = self.decoder.compute_loss(x, g, mask, mean)
        return diff_loss
