# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import torch

from model.base import BaseModule
from model.modules_IDMap import GradLogPEstimator_linear

class Diffusion(BaseModule):
    def __init__(self, n_feats, dim_unet, dim_spk, use_ref_t, beta_min, beta_max):
        super(Diffusion, self).__init__()
        # self.estimator = GradLogPEstimator(dim_unet, dim_spk, use_ref_t)
        self.estimator = GradLogPEstimator_linear()
        # self.n_feats = n_feats
        self.n_feats = n_feats
        self.dim_unet = dim_unet
        self.dim_spk = dim_spk
        # self.use_ref_t = use_ref_t
        self.use_ref_t = False
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_beta(self, t):
        beta = self.beta_min + (self.beta_max - self.beta_min) * t
        return beta

    def get_gamma(self, s, t, p=1.0, use_torch=False):
        beta_integral = self.beta_min + 0.5*(self.beta_max - self.beta_min)*(t + s)
        beta_integral *= (t - s)
        if use_torch:
            gamma = torch.exp(-0.5*p*beta_integral).unsqueeze(-1).unsqueeze(-1)
        else:
            gamma = math.exp(-0.5*p*beta_integral)
        return gamma

    def get_mu(self, s, t):
        a = self.get_gamma(s, t)
        b = 1.0 - self.get_gamma(0, s, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return a * b / c

    def get_nu(self, s, t):
        a = self.get_gamma(0, s)
        b = 1.0 - self.get_gamma(s, t, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return a * b / c

    def get_sigma(self, s, t):
        a = 1.0 - self.get_gamma(0, s, p=2.0)
        b = 1.0 - self.get_gamma(s, t, p=2.0)
        c = 1.0 - self.get_gamma(0, t, p=2.0)
        return math.sqrt(a * b / c)

    def compute_diffused_mean(self, x0, mask, mean, t, use_torch=False):
        x0_weight = self.get_gamma(0, t, use_torch=use_torch)
        mean_weight = 1.0 - x0_weight
        xt_mean = x0 * x0_weight + mean * mean_weight
        return xt_mean * mask

    def forward_diffusion(self, x0, mask, mean, t):
        xt_mean = self.compute_diffused_mean(x0, mask, mean, t, use_torch=True)
        variance = 1.0 - self.get_gamma(0, t, p=2.0, use_torch=True)
        z = torch.rand(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)*2-1
        xt = xt_mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, g, mask, mean, ref, ref_mask, mean_ref, c, 
                          n_timesteps, mode):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = 1.0 - i*h
            time = t * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
            beta_t = self.get_beta(t)
            xt_ref = [self.compute_diffused_mean(ref, ref_mask, mean_ref, t)]
#            for j in range(15):
#                xt_ref += [self.compute_diffused_mean(ref, ref_mask, mean_ref, (j+0.5)/15.0)]
            xt_ref = torch.stack(xt_ref, 1)
            if mode == 'pf':
                dxt = 0.5 * (mean - xt - self.estimator(xt, mask, mean, xt_ref, ref_mask, c, time)) * (beta_t * h)
            else:
                if mode == 'ml':
                    kappa = self.get_gamma(0, t - h) * (1.0 - self.get_gamma(t - h, t, p=2.0))
                    kappa /= (self.get_gamma(0, t) * beta_t * h)
                    kappa -= 1.0
                    omega = self.get_nu(t - h, t) / self.get_gamma(0, t)
                    omega += self.get_mu(t - h, t)
                    omega -= (0.5 * beta_t * h + 1.0)
                    sigma = self.get_sigma(t - h, t)
                else:
                    kappa = 0.0
                    omega = 0.0
                    sigma = math.sqrt(beta_t * h)
                dxt = (mean - xt) * (0.5 * beta_t * h + omega)
                dxt -= self.estimator(xt, g, mask, mean, xt_ref, ref_mask, c, time) * (1.0 + kappa) * (beta_t * h)
                # dxt += (torch.randn_like(z, device=z.device)*2-1) * sigma
                dxt += ((torch.rand_like(z, device=z.device))*2-1) * sigma
            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def forward(self, z, mask, mean, ref, ref_mask, mean_ref, c, 
                n_timesteps, mode):
        if mode not in ['pf', 'em', 'ml']:
            print('Inference mode must be one of [pf, em, ml]!')
            return z
        return self.reverse_diffusion(z, mask, mean, ref, ref_mask, mean_ref, c, 
                                      n_timesteps, mode)
        
    def reverse(self, z, mask, mean, g, n_timesteps, mode, temp=0, loss=False):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = 1.0 - i*h
            time = t * torch.ones(z.shape[0], dtype=z.dtype, device=z.device)
            beta_t = self.get_beta(t)
            # xt_ref = [self.compute_diffused_mean(ref, ref_mask, mean_ref, t)]
#            for j in range(15):
#                xt_ref += [self.compute_diffused_mean(ref, ref_mask, mean_ref, (j+0.5)/15.0)]
            # xt_ref = torch.stack(xt_ref, 1)
            if mode == 'pf':
                dxt = 0.5 * (mean - xt - self.estimator(xt, mask, mean, xt_ref, ref_mask, c, time)) * (beta_t * h)
            else:
                if mode == 'ml':
                    kappa = self.get_gamma(0, t - h) * (1.0 - self.get_gamma(t - h, t, p=2.0))
                    kappa /= (self.get_gamma(0, t) * beta_t * h)
                    kappa -= 1.0
                    omega = self.get_nu(t - h, t) / self.get_gamma(0, t)
                    omega += self.get_mu(t - h, t)
                    omega -= (0.5 * beta_t * h + 1.0)
                    sigma = self.get_sigma(t - h, t)
                else:
                    kappa = 0.0
                    omega = 0.0
                    sigma = math.sqrt(beta_t * h)
                dxt = (mean -xt) * (0.5 * beta_t * h + omega)
                # dxt -= self.estimator(xt, mask, mean, xt_ref, ref_mask, c, time) * (1.0 + kappa) * (beta_t * h)
                dxt -= self.estimator(xt, g, torch.Tensor(mask).cuda(), mean, time, 0)* (1.0 + kappa)* (beta_t * h)
                dxt += ((torch.rand_like(z, device=z.device))*2-1) * sigma
            xt = (xt - dxt) * mask
        return xt

    def forward_(self, z, mask, mean, g, n_timesteps, mode):
        if mode not in ['pf', 'em', 'ml']:
            print('Inference mode must be one of [pf, em, ml]!')
            return z
        return self.reverse(z, mask, mean, g, n_timesteps, mode)