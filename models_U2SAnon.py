import torch
import torch.nn as nn
import os
import random
import sys
import numpy as np
import pdb
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR
import pickle
import torchvision.models as models
import hashlib
import torch.nn.functional as F
np.random.seed(1992)

torch.autograd.set_detect_anomaly(True)

def get_label(filename):
    # 使用 SHA-256 哈希函数，并取出前16个字符作为唯一标识符
    return int(hashlib.sha256(filename.encode()).hexdigest(), 16) % (10 ** 16)

########################################################
# 适用于 [batch, 1, 128] 的输入
class UIDV_processor_Identity_remover(nn.Module):
    def __init__(self, channels):
        super(UIDV_processor_Identity_remover_basic, self).__init__()
        # self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.linear1 = nn.Linear(512, 512)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu1 = nn.ReLU()
        # self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)

        self.linear2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu2 = nn.ReLU()
        self.condition_conv = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)
        

        # self.conv3 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.linear3 = nn.Linear(512, 512)
        self.bn3 = nn.BatchNorm1d(channels)
        self.relu3 = nn.ReLU()
        self.linear = nn.Linear(512, 512)
        # self.fc = nn.Linear(1024, 512)

        self.identity_linear() = nn.Linear(512, 512)
        self.condition_relu = nn.ReLU()
        self.identity_linear2() = nn.Linear(512, 512)
        
    def forward(self, x, condition):
        
        # UIDV Processor
        identity = self.identity_linear2(condition)
        # identity = self.condition_relu(identity)
        # identity = self.identity_linear2(identity)

        # Identity remover 
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

        out = torch.cat((out, identity), dim=-1)
        # out = self.fc(out)

        return out

# 定义残差网络，用于处理 [batch, 1, 256] 格式的输入
class UIDV_processor_Identity_remover(nn.Module):
    def __init__(self, input_channels=1, num_blocks=1):
        super(UIDV_processor_Identity_remover, self).__init__()
        # 目前效果最好的anonymizer
        # self.blocks = nn.ModuleList([BasicResBlock(input_channels) for _ in range(num_blocks)])

        self.blocks = nn.ModuleList([UIDV_processor_Identity_remover_basic(input_channels) for _ in range(num_blocks)])

    def forward(self, x, condition):
        for block in self.blocks:
            x = block(x, condition)
        return x

# 定义全连接层
class Post_processor(nn.Module):
    def __init__(self):
        super(Post_processor, self).__init__()
        self.linear1 = nn.Linear(1024, 512)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(512, 512)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(512, 512)
        # self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x =self.linear1(x)
        x = self.relu1(x)
        # x = self.dropout(x)   
        x = self.linear2(x)                              
        x = self.relu2(x)
        # x = self.dropout(x)
        x = self.linear3(x)
        return x

import json
class U2SAnon(nn.Module):
    def __init__(self, condition_file_path='/home/zyliu/project/VQMIVC-124/utils_feature/generated_identitys_110000.json', num_blocks=1, trainning_flag=False, identites=None):
        super(U2SAnon, self).__init__()
        self.transform = UIDV_processor_Identity_remover(num_blocks=num_blocks)
        self.post_processor = Post_processor()
        self.trainning_flag = trainning_flag
        
        if identites:
            self.identites = identites
        else:
            self.identites = None

        # Load pre-generated identities from the file
        with open(condition_file_path, 'r') as f:
            self.identities = np.array(json.load(f))
        self.num_identities = len(self.identities)
        self.assigned_labels = {}  # To track assigned identities

    def _get_label(self, filename):
        # Generate a unique label using the hash of the filename
        return int(hashlib.md5(filename.encode()).hexdigest(), 16) % (10 ** 8)

    def _assign_condition(self, label):
        # If the label has already been assigned, return the assigned condition index
        if label in self.assigned_labels:
            return self.assigned_labels[label]

        # If there are still unassigned identities, assign a new one
        if len(self.assigned_labels) < self.num_identities:
            condition_idx = len(self.assigned_labels)
            self.assigned_labels[label] = condition_idx
        else:
            # If all identities are assigned, randomly select an existing condition
            condition_idx = np.random.randint(0, self.num_identities)
            self.assigned_labels[label] = condition_idx

        return condition_idx

    def forward(self, x, filenames):
        # Ensure filenames length matches batch size
        assert len(filenames) == x.size(0), "Filenames length must match the batch size"

        identities = []
        for i, filename in enumerate(filenames):
            # Generate a unique label for the filename
            label = self._get_label(filename)
            # Assign a condition based on the label
            condition_idx = self._assign_condition(label)
            # Retrieve the assigned condition
            condition = torch.tensor(self.identities[condition_idx], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(x.device)  # [1, 1, feature_dim]
            identities.append(condition)

        # Concatenate identities into a batch
        identities = torch.cat(identities, dim=0)  # [batch_size, 1, feature_dim]
            
        # Process through ResNet
        if len(x.shape) == 2:
            x = x.unsqueeze(1)
        x = self.transform(x, identities)

        # Process through post_processor
        x = self.post_processor(x)
        
        if self.trainning_flag:
            return x, identities
        else:
            return x
    
