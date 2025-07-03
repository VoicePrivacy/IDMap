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
import torchaudio
import numpy as np
from mel_processing import mel_spectrogram_torch
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

if __name__ == "__main__":
    ## load model
    hps=e2e_utils.get_hparams(config_path="../data/configs/gst.json")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path ='../data/model_HiFiGAN/G.pth'

    net_g = Synthesizer_spkasrbn_gst(**hps.model).cuda().eval()
    _ = e2e_utils.load_checkpoint(checkpoint_path, net_g, None)

    from model_idmap_mlp import IDMap_MLP
    Anonymizer =IDMap_MLP().cuda()
    Anonymizer.load_state_dict(torch.load('../data/IDMap_MLP/N/best_N_45_01.pth'))
    
    ## load feature
    c_path = "../data/libri_dev/content/84-121123-0001.pt"
    fai_vector = torch.load("/home/zyliu/data-78/train-tts100/train/spkembs/19_198_000000_000000.pt").cuda()
    fai_vector = torch.tensor(fai_vector, dtype=torch.float32).unsqueeze(0)
    
    wav_path = "../data/ESD/wav/1.wav"
    wav, _ = torchaudio.load(wav_path)
    mel = mel_spectrogram_torch(wav.squeeze(1),1280,80,16000,320,1280,0,8000)
    
    ## generate speaker vector
    ID = 1234
    np.random.seed(ID)
    
    # identity = np.random.uniform(-1, 1, 1000)
    identity = np.random.normal(0, 1, size=(1, 1, 512))
    identity = torch.tensor(identity).cuda()
    g = Anonymizer.infer(fai_vector, identity).squeeze(0)
    
    ## inference
    with torch.no_grad():
        audio=net_g.infer(c,mel,g)
        audio = audio[0].data.float().cpu()
    
    save_path = os.path.join("./anonymize_wav","84-121123-0001-anon.wav")
    torchaudio.save(save_path,audio,16000,bits_per_sample=16)
    
    
    