import torch
import torch.nn as nn
import os
import random
import sys
import numpy as np
import pdb
import torch.nn.functional as F
import json
import random
from typing import List

np.random.seed(1992)

torch.autograd.set_detect_anomaly(True)

########################################################
# 全部改成线性层
# 定义基础的残差块，适用于 [batch, 1, 128] 的输入
class BasicResBlock_Auxiliary_Pre_processor(nn.Module):
    def __init__(self, channels):
        super(BasicResBlock_Auxiliary_Pre_processor, self).__init__()
        # Auxiliary
        self.Auxiliary = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(channels),
            nn.ReLU()
        )
        
        # Pre-processor
        self.pre_processor = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512)
        )


    def forward(self, x, condition):
        identity = self.pre_processor(condition)
        out = self.Auxiliary(x)
        out = torch.cat((out, identity), dim=-1)
        return out

# 定义残差网络，用于处理 [batch, 1, 256] 格式的输入
class Auxiliary_Pre_processor(nn.Module):
    def __init__(self, input_channels=1, num_blocks=3):
        super(Auxiliary_Pre_processor, self).__init__()
        self.blocks = nn.ModuleList([BasicResBlock_Auxiliary_Pre_processor(input_channels) for _ in range(num_blocks)])

    def forward(self, x, condition):
        for block in self.blocks:
            x = block(x, condition)
        return x

# 定义全连接层
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.linear1 = nn.Linear(512, 512)
        self.relu1 = nn.ReLU()
        # self.linear2 = nn.Linear(512, 512)
        # self.relu2 = nn.ReLU()
        # self.linear3 = nn.Linear(512, 512)
        # self.dropout = nn.Dropout(0.2)
        
        self.fc = nn.Linear(512+512, 512)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu1(x)
        x = self.linear1(x)
        # x = self.relu1(x)
        # x = self.dropout(x)   
        # x = self.linear2(x)                              
        # x = self.relu2(x)
        # x = self.dropout(x)
        # x = self.linear3(x)
        return x

class IDMap_MLP(nn.Module):
    def __init__(self, condition_file_path='./generated_identitys_110000.json', 
                 num_blocks=1, trainning_flag=False, identities=None):
        super(IDMap_MLP, self).__init__()
        self.au_pre_proc = Auxiliary_Pre_processor(num_blocks=1)  
        self.generator = generator() 
        
        self.dict = {}  
        self.IDVs = identities if identities is not None else {}  
        self.current_max_label = 0  
        
        if os.path.exists(condition_file_path):
            try:
                with open(condition_file_path, 'r') as f:
                    self.IDVs = json.load(f)
                if self.IDVs:
                    self.current_max_label = max(map(int, self.IDVs.keys()))
            except Exception as e:
                print(f"Fail: {e}")
    
    def _get_label(self, filename: str) -> int:
        """
        Args:
            filename: audio file name
            
        Returns:
            assigned label
        """
        speaker = self._extract_speaker_from_filename(filename)
        if speaker in self.dict:
            return self.dict[speaker]   
        
        new_label = self.current_max_label + random.randint(1, 1000)
        self.dict[speaker] = new_label
        self.current_max_label = new_label
        return new_label
    
    def _extract_speaker_from_filename(self, filename: str) -> str:
        if '_' in filename:
            return filename.split('_')[0]
        elif '-' in filename: 
            return filename.split('-')[0]
        else:
            return filename
    
    def _assign_condition(self, label: int) -> int:
        return label
    
    def infer(self, fai_vector, identity):
        # fai_vector:[1,1,512], identity:[1,1,512]
        x = self.au_pre_proc(fai_vector, identity)
        x = self.generator(x)
        return x.detach()
    
    def forward(self, x, filenames):
        assert len(filenames) == x.size(0), "Filenames length must match the batch size"

        conditions = []
        for filename in filenames:
            condition_idx = self._get_label(filename)
            # check if the IDs exist
            if str(condition_idx) in self.IDVs:
                condition = torch.tensor(
                    self.IDVs[str(condition_idx)], 
                    dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0).to(x.device)
            else:
                random.seed(condition_idx)  
                torch.manual_seed(condition_idx) 
                feature_dim = 512 
                condition = torch.randn(1, 1, feature_dim).to(x.device)
                self.IDVs[str(condition_idx)] = condition.squeeze().cpu().tolist()
            conditions.append(condition)
        conditions = torch.cat(conditions, dim=0)  # [batch_size, 1, feature_dim]
        
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
            
        x = self.au_pre_proc(x, conditions)
        x = self.generator(x)

        return x, conditions