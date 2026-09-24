import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from models_hifigan import Synthesizer_spkasrbn_gst
import e2e_utils
from tqdm import tqdm
from glob import glob
import random
import numpy as np
from mel_processing import mel_spectrogram_torch
import random

import e2e_utils
import json
from tqdm import tqdm
import random
import pdb
from model.vc_IDMap import DiffVC
import params

n_mels = 1
sampling_rate = params.sampling_rate
n_fft = params.n_fft
hop_size = params.hop_size

channels = params.channels
filters = params.filters
layers = params.layers
kernel = params.kernel
dropout = params.dropout
heads = params.heads
window_size = params.window_size
enc_dim = params.enc_dim

dec_dim = params.dec_dim
# spk_dim = params.spk_dim
spk_dim = 512
use_ref_t = False
beta_min = params.beta_min
beta_max = params.beta_max

random_seed = params.seed
test_size = params.test_size
# 训练设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IDMap_diff = DiffVC(n_mels, channels, filters, heads, layers, kernel,
                        dropout, window_size, enc_dim, spk_dim, use_ref_t,
                        dec_dim, beta_min, beta_max).cuda()
IDMap_diff.load_state_dict(torch.load("../data/IDMap_Diff/Diffusion_IDMap.pt"))


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

if __name__ == "__main__":
    ## load model
    hps=e2e_utils.get_hparams(config_path="../data/configs/gst.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path ='../data/model_HiFiGAN/G.pth'

    net_g = Synthesizer_spkasrbn_gst(**hps.model).cuda().eval()
    _ = e2e_utils.load_checkpoint(checkpoint_path, net_g, None)
    
    ## load feature
    c_path = "../data/libri_dev/content/84-121123-0001.pt"
    fai_vector = torch.load("../data/fai.pt").cuda()
    fai_vector = torch.tensor(fai_vector, dtype=torch.float32).unsqueeze(0)
    
    ## generate speaker vector
    ID = 1234
    np.random.seed(ID)
    torch.manual_seed(ID)
    identity = torch.Tensor(np.random.normal(0, 1, (2, 512))).cuda()
    if len(identity.shape) == 2:
        identity = identity.unsqueeze(1)
    fai_vector = fai_vector.repeat(2, 1, 1).cuda()
    g = IDMap_diff.forward_(fai_vector, 10, identity)
    g = g[0]
    g = F.normalize(g, dim=-1)
    g = g.cpu().detach().numpy().flatten()
    
    ## inference
    with torch.no_grad():
        audio=net_g.infer(c,mel,g)
        audio = audio[0].data.float().cpu()
    
    save_path = os.path.join("./anonymize_wav","84-121123-0001-diff.wav")
    torchaudio.save(wav_path_my,audio,16000,bits_per_sample=16)
    
    
    