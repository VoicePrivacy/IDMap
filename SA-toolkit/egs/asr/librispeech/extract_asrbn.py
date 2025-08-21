import argparse
import torch
from satools import infer_helper
import satools.utils
import torch
import torchaudio
from pathlib import Path
import os 

def process_mel_file(audio_path):
    wav,_ = torchaudio.load(audio_path)
    wav =wav.cuda()
    asrbn = asrbn_model.extract_bn(wav).permute(0, 2, 1)
    return asrbn

model_path = "./exp/chain/bn_tdnnf_wav2vec2_train_clean_960_vq_48/final.pt"
asrbn_model = infer_helper.load_model(model_path, from_file=__file__, load_weight=True)
asrbn_model = asrbn_model.cuda().eval()

audio_path = "example/1.wav"
asrbn = process_mel_file(audio_path)
print(asrbn.shape)
