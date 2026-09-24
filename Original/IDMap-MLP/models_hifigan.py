import math
import torch
from torch import nn
from torch.nn import functional as F

import commons
import modules
from GST import GST
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from commons import init_weights
import random

class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):

        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

class Generator_spkemb_gst(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator_spkemb_gst, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        # self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2
        self.merge=Conv1d(256,512, 7, 1, padding=3)
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)
        # self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        if gin_channels != 0:
            self.cond = nn.Conv1d(512, 256, 1)

    def forward(self, x, g, gst):
        x = x + gst
        x= self.merge(x) + g
        x= F.leaky_relu(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()

class Synthesizer_spkasrbn_gst(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    inter_channels,
    hidden_channels,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    gin_channels=0,
    **kwargs):

    super().__init__()
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.gin_channels = gin_channels
    # self.enc_p = PosteriorEncoder(128, inter_channels, hidden_channels, 5, 1, 16)#768, inter_channels, hidden_channels, 5, 1, 16)
    self.gst=GST()
    self.dec=Generator_spkemb_gst(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
    self.linear_layer = torch.nn.Linear(512, 256)

  
  def forward(self,c,ref_mel,g):
    g = g.transpose(1,2).expand(-1,-1,128)

    style_emb=self.gst(ref_mel.transpose(1,2))
    
    style_emb=style_emb.transpose(1,2).expand(-1,-1,128)

    y_hat=self.dec(c,g,style_emb)  

    return y_hat

  def infer(self, c,mel,g):
    style_emb = self.gst(mel.transpose(1,2))
    style_emb = style_emb.transpose(1,2).expand(-1,-1,c.shape[2])
    g = g.unsqueeze(2).expand(-1,-1,c.shape[2])
    y_hat= self.dec(c,g,style_emb) 
    
    return y_hat

  def extract_gst(self, mel):
    # mel:[batch,bins,frame]
    # style_emb:[batch,1,256]
    style_emb=self.gst(mel.transpose(1,2))
    return style_emb
