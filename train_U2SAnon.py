import os
import torch
from torch.utils.data import Dataset, DataLoader
import glob
from models_U2SAnon import U2SAnon
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import torch.nn.functional as F  # 导入函数库

import sys

# 获取命令行参数
args = sys.argv
# 打印所有参数
print("命令行参数:", args)

# 解析参数
if len(args) < 2:
    print("Usage: python script.py <input_file> <output_file>")
    sys.exit(1)
    
identity_path = args[1]

seed = 45
# torch.manual_seed(seed)

torch.autograd.set_detect_anomaly(True)
# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, original_data_paths, target_data_paths):
        self.original_data_paths = original_data_paths
        self.target_data_paths = target_data_paths

    def __len__(self):
        return len(self.original_data_paths)

    def __getitem__(self, idx):
        # 加载原始数据
        original_data = np.load(self.original_data_paths[idx])

        # 提取文件名作为标签
        original_filename = os.path.basename(self.original_data_paths[idx])

        return original_data, original_filename

# 加载数据路径函数
def load_data(original_root, target_root):
    original_data_paths = []
    target_data_paths = []

    # subdirs = ['dev_enroll_spkemb', 'test_enroll_spkemb', 'dev_trials_f_spkemb', 'dev_trials_m_spkemb', 'test_trials_f_spkemb', 'test_trials_m_spkemb', '360_spkemb']
    subdirs = ['100_spkemb']
    # 遍历子目录
    for subdir in subdirs:
        # Change the path to your training dataset, original_dataset and target_dataset can be different, in our experiment, they are the same. 
        original_paths = glob.glob(os.path.join('/home4T_0/cyguo_data/zyliu/project/VQMIVC/train-tts360/train/spkembs', '*', '*.npy'), recursive=True)
        target_paths = glob.glob(os.path.join(target_root, subdir, '*.npy'), recursive=True)
        
        # 确保顺序一致
        original_paths.sort()
        target_paths.sort()

        original_data_paths.extend(original_paths)
        target_data_paths.extend(target_paths)
    
    print(f"data num:{len(original_data_paths)}, {len(target_data_paths)}")

    return original_data_paths, target_data_paths

def classify_features_by_person(file_names, original_batch, anony_batch, target_batch, target_person_names):
    """
    将原始数据和anony数据按照人名进行分类。

    Args:
        file_names (tuple): 包含文件名的元组。
        original_batch (torch.Tensor): 原始数据的特征张量，形状为 [batch, 1, 512]。
        anony_batch (torch.Tensor): anony数据的特征张量,形状为 [batch, 1, 512]。

    Returns:
        dict: 按人名分类的特征字典。
    """
    features_by_person = {}

    for file_name in file_names:
        # 提取人名（假设人名在'-'前面）
        person_name = file_name.split('-')[0]
        idx = file_names.index(file_name) 

        # 将特征添加到对应的人名中
        if person_name not in features_by_person:
            features_by_person[person_name] = {'original': [], 'anony': [], 'target': [], 'target_id': []}
        features_by_person[person_name]['original'].append(original_batch[idx])
        features_by_person[person_name]['anony'].append(anony_batch[idx])
        features_by_person[person_name]['target'].append(target_batch[idx])
        features_by_person[person_name]['target_id'].append(target_person_names[idx])

    return features_by_person

# 创建 CosineEmbeddingLoss 实例
c_loss = nn.CosineEmbeddingLoss(reduction='mean')
cos_similarity = nn.CosineSimilarity(dim=2, eps=1e-8)

class AnonymizationLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5):
        super(AnonymizationLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.cosine_loss = nn.CosineEmbeddingLoss(reduction='mean')
        self.alpha = alpha  # 权重系数用于调整MSE损失
        self.beta = beta    # 权重系数用于调整余弦相似度损失

    def forward(self, anonymized_data, target_data):
        mse_loss = self.mse_loss(anonymized_data, target_data)
        target = torch.ones(anonymized_data.shape[0]).cuda()
        cosine_loss = self.cosine_loss(anonymized_data.squeeze(1), target_data.squeeze(1), target)
        total_loss = self.alpha * mse_loss + self.beta * cosine_loss
        return mse_loss, cosine_loss



def calculate_cosine_similarity_loss(features_by_person):
    """
    计算每对特征之间的余弦相似性损失。

    Args:
        features_by_person (dict): 按人名分类的特征字典。

    Returns:
        float: 平均余弦相似性损失。
    """
    c_loss = nn.CosineEmbeddingLoss(reduction='mean')
    total_loss = 0.0
    num_pairs = 0

    for person_name, features_dict in features_by_person.items():
        anony_features = features_dict['anony']
        target = features_dict['target']
        ori = features_dict['original']
        target_id = features_dict['target_id']
        num_samples = len(anony_features)

        # 仅当特征数量大于1时才计算余弦相似性损失
        if num_samples > 1:
            # 计算每对特征之间的余弦相似性损失
            for i in range(num_samples):
                for j in range(i + 1, num_samples):
                    # 假设特征是1D向量（形状：[batch, 512]）
                    feature_i = anony_features[i]
                    feature_j = anony_features[j]


                    # 创建标签（相似对为1，不相似对为-1）
                    target_i = target_id[i]
                    target_j = target_id[j]

                    # 设置标签（相似对为1，不相似对为-1）
                    if target_i == target_j:
                        labels = torch.ones(1)
                    else:
                        labels = torch.ones(1) * -1

                    # 计算余弦相似度
                    # cosine_similarity = torch.nn.functional.cosine_similarity(target_i, target_j)

                    # # 设置标签（相似对为1，不相似对为-1）
                    # if cosine_similarity > 0.8:
                    #     labels = torch.ones(1)
                    # else:
                    #     labels = torch.ones(1) * -1

                    # 计算余弦相似性损失
                    loss = c_loss(feature_i, feature_j, labels.cuda())
                    total_loss = total_loss + loss.item()
                    num_pairs += 1
    # 计算平均损失
    if num_pairs > 0:
        mean_loss = total_loss / num_pairs
    else:
        mean_loss = total_loss
    print(f"共有{num_pairs}进行区分")
    return mean_loss

# def calculate_cosine_similarity_loss(features_by_person):
#     """
#     计算每对特征之间的余弦相似性损失，包括同类别和不同类别的特征。

#     Args:
#         features_by_person (dict): 按人名分类的特征字典。

#     Returns:
#         float: 平均余弦相似性损失。
#     """
#     c_loss = nn.CosineEmbeddingLoss(reduction='mean')
#     total_loss_same = 0.0  # 同类别的总损失
#     total_loss_diff = 0.0  # 不同类别的总损失
#     num_same_pairs = 0
#     num_diff_pairs = 0

#     person_names = list(features_by_person.keys())

#     # 计算同类别特征的余弦相似性损失
#     for person_name, features_dict in features_by_person.items():
#         anony_features = features_dict['anony']
#         num_samples = len(anony_features)

#         # 仅当特征数量大于1时才计算同类别的余弦相似性损失
#         if num_samples > 1:
#             for i in range(num_samples):
#                 for j in range(i + 1, num_samples):
#                     feature_i = anony_features[i]
#                     feature_j = anony_features[j]

#                     # 计算余弦相似性损失
#                     cosine_similarity = torch.nn.functional.cosine_similarity(feature_i, feature_j)

#                     # 设置标签（相似对为1，不相似对为-1）
#                     labels = torch.ones(1) if cosine_similarity > 0.8 else torch.ones(1) * -1

#                     # 计算损失
#                     loss = c_loss(feature_i, feature_j, labels.cuda())
#                     total_loss_same += loss.item()
#                     num_same_pairs += 1

#     # 计算不同类别特征的余弦相似性损失
#     for i in range(len(person_names)):
#         for j in range(i + 1, len(person_names)):
#             person_i = person_names[i]
#             person_j = person_names[j]

#             anony_features_i = features_by_person[person_i]['anony']
#             anony_features_j = features_by_person[person_j]['anony']

#             # 计算跨类别的余弦相似性损失
#             for feature_i in anony_features_i:
#                 for feature_j in anony_features_j:
#                     cosine_similarity = torch.nn.functional.cosine_similarity(feature_i, feature_j)

#                     # 标签设置为 -1 因为这是不同类别
#                     labels = torch.ones(1) * -1

#                     # 计算损失
#                     loss = c_loss(feature_i, feature_j, labels.cuda())
#                     total_loss_diff += loss.item()
#                     num_diff_pairs += 1

#     # 计算平均损失
#     mean_loss_same = total_loss_same / num_same_pairs if num_same_pairs > 0 else 0.0
#     mean_loss_diff = total_loss_diff / num_diff_pairs if num_diff_pairs > 0 else 0.0

#     # print(f"共有{num_same_pairs}个同类别对进行区分，{num_diff_pairs}个不同类别对进行区分")
#     # print(f"同类别平均损失: {mean_loss_same}, 不同类别平均损失: {mean_loss_diff}")

#     # 最终返回同类别和不同类别的平均损失
#     return mean_loss_same, mean_loss_diff
#     # return mean_loss_diff    

import torch
import random

######
# 仅在spk上使用
def compute_triplet_loss(features_by_person, margin=0.2):
    """
    计算按人名分类的特征的 triplet loss。

    Args:
        features_by_person (dict): 按人名分类的特征字典,包含原始数据和anony数据。
        margin (float): triplet loss中的margin。

    Returns:
        float: 计算得到的 triplet loss。
    """
    triplet_losses = []

    for person, features in features_by_person.items():
        original_features = features['original']
        anony_features = features['anony']

        # 如果特征数量大于1，计算 triplet loss
        if len(original_features) > 1:
            anchor_feature = random.choice(original_features)
            positive_feature = random.choice(anony_features)
            negative_feature = random.choice(anony_features)

            # 计算余弦距离
            pos_distance = torch.cosine_similarity(anchor_feature, positive_feature, dim=2)
            neg_distance = torch.cosine_similarity(anchor_feature, negative_feature, dim=2)

            # 计算 triplet loss
            triplet_loss = torch.clamp(pos_distance - neg_distance + margin, min=0)
            triplet_losses.append(triplet_loss.item())

    # 取均值作为最终的 triplet loss
    if triplet_losses:
        mean_triplet_loss = sum(triplet_losses) / len(triplet_losses)
    else:
        mean_triplet_loss = 0.0

    return mean_triplet_loss


# 确保identities和pseudo-xvector之间有正相关的区分度
def compute_correlation_loss(anonymized_data, conditions):
    # 将conditions和anonymized_data展平成[batch_size, feature_dim]
    conditions_flat = conditions.squeeze(1)  # [batch_size, 512]
    anonymized_data_flat = anonymized_data.squeeze(1)  # [batch_size, 512]
    
    # 对conditions和anonymized_data进行归一化
    conditions_normalized = F.normalize(conditions_flat, p=2, dim=1)
    anonymized_data_normalized = F.normalize(anonymized_data_flat, p=2, dim=1)
    
    # 计算conditions和anonymized_data的余弦相似度矩阵
    cos_sim_conditions = torch.mm(conditions_normalized, conditions_normalized.t())  # [batch_size, batch_size]
    cos_sim_anonymized = torch.mm(anonymized_data_normalized, anonymized_data_normalized.t())  # [batch_size, batch_size]
    
    # 创建掩码，排除自身比较
    mask = ~torch.eye(cos_sim_conditions.size(0), dtype=torch.bool, device=cos_sim_conditions.device)
    
    # 找出conditions相似度接近1或小于0.2的索引
    idx_pos = (cos_sim_conditions > 0.9) & mask  # conditions相似度接近1
    idx_neg = (cos_sim_conditions < 0.2) & mask   # conditions相似度小于0.2
    
    # 使用 mask 和条件筛选出 cos_sim_conditions 中不相似的元素
    cos_sim_dissimilar = cos_sim_conditions[(cos_sim_conditions < 0.2) & mask]
    # 计算不相似元素的均值
    if cos_sim_dissimilar.numel() > 0:
        mean_dissimilarity = cos_sim_dissimilar.mean()
    else:
        mean_dissimilarity = torch.tensor(0.0, device=cos_sim_conditions.device)  # 如果没有不相似的元素，设为 0.0
    
    
    # 计算相似度的均值和标准差
    mean_conditions = cos_sim_conditions.mean()
    std_conditions = cos_sim_conditions.std()

    k = 3
    # 动态设置 margin，基于 conditions 的均值和标准差
    margin = mean_conditions + k * std_conditions
    margin = torch.clamp(margin, min=0.0, max=0.2)  # 限制在合理范围内（0.0 到 0.3 之间）
    
    # 对于相似的identites（cos_sim = 1 ），希望anonymized_data也相似
    if idx_pos.any():
        anchor_pos = anonymized_data_flat[idx_pos.nonzero()[:, 0]]
        positive_pos = anonymized_data_flat[idx_pos.nonzero()[:, 1]]
        labels_pos = torch.ones(anchor_pos.size(0)).to(anonymized_data.device)  # 标签为1
        cos_loss_fn = nn.CosineEmbeddingLoss(reduction='mean', margin=0.0)  
        loss_pos = cos_loss_fn(anchor_pos, positive_pos, labels_pos)
    else:
        loss_pos = torch.tensor(0.0, device=anonymized_data.device)
    
    # 对于不相似的identites（cos_sim < 0.2），希望anonymized_data也不相似
    if idx_neg.any():
        anchor_neg = anonymized_data_flat[idx_neg.nonzero()[:, 0]]
        negative_neg = anonymized_data_flat[idx_neg.nonzero()[:, 1]]
        labels_neg = -torch.ones(anchor_neg.size(0)).to(anonymized_data.device)  # 标签为-1
        cos_loss_fn = nn.CosineEmbeddingLoss(reduction='mean', margin=margin)  # 可以调整margin
        loss_neg = cos_loss_fn(anchor_neg, negative_neg, labels_neg)
    else:
        loss_neg = torch.tensor(0.0, device=anonymized_data.device)
    
    # 总损失为正负损失之和
    loss = loss_pos + loss_neg
    return loss



import json

global_person_list = set()

def extract_person_names(original_filenames):
    """
    从文件名tuple中提取人的名字，并更新全局人名列表。

    Args:
        original_filenames (tuple): 文件名的tuple，类似('1547-130184-0077.npy', '4427-12471-0022.npy')

    Returns:
        tuple: 只包含人的名字的tuple，例如 ('1547', '4427')
    """
    person_names = []
    for file_name in original_filenames:
        # 提取人的名字，假设文件名格式为 '1547-130184-0077.npy'
        
        if seed == 1992:
            if '-' in file_name:
                person_name = file_name.split('-')[0] + f'{seed}'
            else:
                person_name = file_name.split('_')[0] + f'{seed}'
        else:
            # person_name = file_name.split('-')[0]
            if '-' in file_name:
                person_name = file_name.split('-')[0]
            else:
                person_name = file_name.split('_')[0]
        person_names.append(person_name)

        # 更新全局人名列表
        if person_name not in global_person_list:
            global_person_list.add(person_name)

    return tuple(person_names)

def save_person_list_to_file(global_person_list, file_path):
    """
    将全局人名列表保存到指定的文件中。

    Args:
        global_person_list (set): 全局人名列表。
        file_path (str): 要保存的文件路径。
    """
    # 将set转换为list，便于存储
    person_list = list(global_person_list)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    # 将人名列表保存为JSON文件
    with open(file_path, 'w') as file:
        json.dump(person_list, file, ensure_ascii=False, indent=4)
    
    print(f"全局人名列表已保存到: {file_path}")


# 模型训练函数
def train_anonymizer(model, dataloader, criterion, optimizer, scheduler, num_epochs, model_save_path, log_dir):
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=log_dir)
    best_loss = float('inf')
    best_loss_mini = float('inf')
    target_person = {}
    ite = 20

    for epoch in range(num_epochs):
        running_loss = 0.0
        model.zero_grad()
        model.train()
        
        print(f"epoch:{epoch}")
        for batch_idx, (original_batch, original_filenames) in enumerate(dataloader):
            print(f"epoch_batch_id:{batch_idx}")
            original_batch = original_batch.to(device)
            if len(original_batch.shape) == 2:
                original_batch = original_batch.unsqueeze(1)
                
            perm = torch.randperm(original_batch.size(0))  # 生成打乱索引
            target_batch = original_batch[perm]            # 按照打乱的索引重新排列 original_batch
            target_filenames = [original_filenames[i] for i in perm]  # 同样打乱 original_filenames
            
            cos_loss = 0.0
            num_same_pairs = 0
            
            # Get ID
            target_person_names = extract_person_names(target_filenames)

            # add ground-truth speaker embeddings
            for i, person_name in enumerate(target_person_names):
                if person_name not in target_person:
                    # you can choose utterance spkemb as ground-truth speaker embeddings
                    # or use speaker level spkemb as ground-truth speaker embeddings as we described in the paper.
                    target_person[person_name] = target_batch[i].unsqueeze(0)  
                else:
                    target_batch[i] = target_person[person_name].squeeze(0)  

            # [batch,1,512]-->[batch,1,512]
            anonymized_data, conditions = model(original_batch, target_person_names)
            correlation_loss = compute_correlation_loss(anonymized_data, conditions)
            mse_loss, cosine_loss = criterion(anonymized_data, target_batch)
              
            loss = 0.25*mse_loss + 0.25 cosine_loss + 0.5*correlation_loss
            print(f"batch loss:{loss}; mse_loss:{mse_loss}; cosine_sim:{1 - cosine_loss}; correlation_loss:{correlation_loss}; utt_loss_same:{utt_loss}; target_same:{cos_loss}")            
            loss.backward()
            if (batch_idx % ite == 0):
                optimizer.step()
                model.zero_grad()
            # 记录损失
            running_loss += loss.item()
            writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar('mse_Loss/train', mse_loss.item(), epoch * len(dataloader) + batch_idx)
            writer.add_scalar('cosine_Loss/train', cosine_loss.item(), epoch * len(dataloader) + batch_idx)

            if loss < best_loss_mini:
                best_loss_mini = loss
                torch.save(model.state_dict(), model_save_path.replace("linear_uniform","linear_uniform_mini"))
                print(f'Model saved with mini batch loss {loss:.4f}')
        scheduler.step()
        epoch_loss = running_loss / len(dataloader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_save_path)
            print(f'Model saved with loss {best_loss:.4f}')

    writer.close()

def load_excel_data(file_path):
    """
    从Excel文件加载数据。

    :param file_path: Excel文件的路径
    :return: 从Excel文件中提取的数据
    """
    df = pd.read_excel(file_path, header=None)
    return df.values[1,:]

import pandas as pd
# 训练设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_blocks = 1
model = U2SAnon(num_blocks=num_blocks, trainning_flag=True, condition_file_path=identity_path).to(device)
criterion = AnonymizationLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# 定义路径
original_root = '/home4T_0/cyguo_data/zyliu/project/VQMIVC/train-tts100/spkembs'
target_root = '/home4T_0/cyguo_data/zyliu/project/VQMIVC/train-tts100/spkembs'

# 加载数据路径
original_data_paths, target_data_paths = load_data(original_root, target_root)

# 创建数据集和DataLoader
dataset = CustomDataset(original_data_paths, target_data_paths)
batch_size = 256
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 设置训练参数
num_epochs = 200
model_save_path = f'/home4T_0/cyguo_data/zyliu/project/VQMIVC/utils_feature/logs_uniform_lada/best_anonymizer_uniform_3loss_{seed}_360_blocks{num_blocks}_0.2.pth'
log_dir = '/home4T_0/cyguo_data/zyliu/project/VQMIVC/utils_feature/logs_bsevec_uniform/log'

# 开始训练
train_anonymizer(model, dataloader, criterion, optimizer, scheduler, num_epochs, model_save_path, log_dir)

