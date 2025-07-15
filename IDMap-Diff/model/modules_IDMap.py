import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.base import BaseModule
from einops import rearrange

class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x):
        # x: [B], 标量时间步
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = x.unsqueeze(1) * emb.unsqueeze(0) * 1000.0
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # [B, dim]
        return emb

class RefBlock(BaseModule):
    """
    原本 RefBlock 依赖多层2D卷积和InstanceNorm。这里我们要简化为只对 ref 特征进行简单的条件融合。
    我们仍保留 time_emb 对 ref 的调制,但改为1D卷积或线性变换。
    """
    def __init__(self, out_dim, time_emb_dim):
        super(RefBlock, self).__init__()
        self.out_dim = out_dim
        base_dim = out_dim // 4

        # 使用简单的线性层+Mish进行映射
        self.mlp_time_1 = nn.Sequential(Mish(), nn.Linear(time_emb_dim, base_dim))
        self.mlp_time_2 = nn.Sequential(Mish(), nn.Linear(time_emb_dim, base_dim))

        # 假设 ref 是 [B, out_dim], 将其看作 [B, 1, out_dim], 使用1D卷积模拟处理
        # 这里其实 ref 已经是一个向量了，可以不卷积，直接线性变换即可
        self.ref_fc_1 = nn.Sequential(
            nn.Linear(out_dim, 2*base_dim),
            Mish(),
            nn.Linear(2*base_dim, base_dim)
        )
        self.ref_fc_2 = nn.Sequential(
            nn.Linear(base_dim, 2*base_dim),
            Mish(),
            nn.Linear(2*base_dim, base_dim)
        )

        self.final_fc = nn.Linear(base_dim, out_dim)

    def forward(self, ref, mask, time_emb):
        # ref: [B, out_dim]
        # mask: [B, 1, T=1 or T=512]? 对于ref向量，本例可能不需要mask，或mask全1
        # time_emb: [B, time_emb_dim]

        # 将ref映射到base_dim
        y = self.ref_fc_1(ref)   # [B, base_dim]
        y += self.mlp_time_1(time_emb)  # 融合time emb
        y = self.ref_fc_2(y) 
        y += self.mlp_time_2(time_emb)  # 再次融合time emb
        y = self.final_fc(y)  # [B, out_dim]
        return y


class Block1D(BaseModule):
    def __init__(self, dim=512, dim_out=512):
        super(Block1D, self).__init__()
        self.linear = nn.Linear(dim, dim_out)
        self.batchnorm = nn.BatchNorm1d(dim_out)
        # self.batchnorm = nn.BatchNorm1d(1)
        
        self.mish = Mish()

    def forward(self, x, mask):
        # x: [B, C, T]
        # mask: [B, 1, T]
        if len(mask.shape) == 3:
            mask = mask.squeeze()
        x = x 
        x = self.linear(x) 
        x = self.batchnorm(x)
        output = self.mish(x)
        return output 

class ResnetBlock1D(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock1D, self).__init__()
        self.mlp = nn.Sequential(Mish(), nn.Linear(time_emb_dim, dim_out))
        self.block1 = Block1D(dim, dim_out)
        self.block2 = Block1D(dim_out, dim_out)
        if dim != dim_out:
            self.res_conv = nn.Linear(dim, dim_out)
        else:
            self.res_conv = nn.Identity()

    def forward(self, x, mask, time_emb):
        # x: [B, C], mask: [B,1,T], time_emb: [B, time_emb_dim]
        h = self.block1(x, mask)
        # 将time_emb映射后加到特征上，需要先扩展time_emb维度
        # time_emb: [B, dim_out], 将其扩维与h对齐 [B, dim_out, 1]
        t_emb = self.mlp(time_emb)
        h = h + t_emb
        h = self.block2(h, mask)
        return h + self.res_conv(x)

class GradLogPEstimator_linear(BaseModule):
    def __init__(self, dim_base=512, dim_cond=512, use_ref_t=False, depth=4):
        super(GradLogPEstimator_linear, self).__init__()
        self.use_ref_t = use_ref_t

        # 时间嵌入
        self.time_pos_emb = SinusoidalPosEmb(dim_base)
        self.mlp_t = nn.Sequential(
            nn.Linear(dim_base, dim_base * 4),
            Mish(),
            nn.Linear(dim_base * 4, dim_base)
        )

        cond_total = dim_base + 512  # 加上c的维度
        if use_ref_t:
            self.ref_block = RefBlock(out_dim=dim_cond, time_emb_dim=dim_base)
            cond_total += dim_cond

        self.cond_block = nn.Sequential(
            nn.Linear(cond_total, 4 * dim_cond),
            Mish(),
            nn.Linear(4 * dim_cond, dim_cond)
        )

        # 将输入 x 与 condition 拼在一起后进行处理
        # x: [B, 512] -> 转为 [B, C=1, T=512]
        # condition: [B, dim_cond] 需要扩展为 [B, dim_cond, T=512] 再拼接
        # 最终输入到ResnetBlock的维度: [B, 1+dim_cond, 512]

        self.dim_cond = dim_cond
        self.initial_fc = nn.Linear(1024 + dim_cond, dim_cond)  # 将拼接后的向量降维到dim_cond维的通道
        # 转1D输入格式为[B, Channel, 512]
        
        # 构建多个ResnetBlock1D进行特征提取
        self.resblocks = nn.ModuleList([
            ResnetBlock1D(dim_cond, dim_cond, time_emb_dim=dim_base) for _ in range(depth)
        ])

        # 最终输出层，将dim_cond维映射回1维输出(或512维输出)
        # 如果我们想最终输出仍为512维，可以再添加一层映射
        self.final_fc = nn.Linear(dim_cond, 512)

    def forward(self, x, g, x_mask, mean, t, ref):
        # 假设 x, mean, ref, c 都是 [B, 512]， x_mask为[B, 1, 512]的mask，t为[B]
        # 将 mean 与 x合并: 原始代码将 x stack [mean, x] => [B, 2, ...]
        # 这里我们已经决定不做多尺度结构，只要处理 x 本身即可。
        # 如果仍需 mean，可考虑将mean与x在特征维拼接。
        # 本示例直接用 x 和 mean 的平均值拼，或保留x即可依据需求修改。

        if len(g.shape) >= 3:
            g = g.squeeze()  # [B, 512]
        
        if len(x.shape) >= 3:
            x = x.squeeze()  # [B, 512]
        
        if len(mean.shape) >= 3:
            mean = mean.squeeze() # [B, 512])
            
            
        # t时间嵌入
        condition = self.time_pos_emb(t)       # [B, dim_base]
        t_emb = self.mlp_t(condition)          # [B, dim_base]

        # 如果使用ref
        if self.use_ref_t:
            # ref: [B,512], ref_mask: [B,1,512]
            # 将ref通过ref_block获取额外条件
            ref_cond = self.ref_block(ref, ref_mask, t_emb)  # [B, dim_cond]
            condition = torch.cat([condition, ref_cond], dim=1)  # [B, dim_base+dim_cond]

        # 拼上c: [B, dim_base(+dim_cond)+512]
        condition = torch.cat([condition, g], dim=1)
        condition = self.cond_block(condition)  # [B, dim_cond]

        cond_ = condition

        # cat
        x_seq = torch.cat([x, mean, cond_], dim=1)  # [B, 2*512 + dim_cond]
        
        x_seq =  self.initial_fc(x_seq)

        # 对序列应用若干 ResnetBlock1D
        for rb in self.resblocks:
            x_seq = rb(x_seq, x_mask, t_emb)

        # 最终映射回512维特征，这里将x_seq平均池化时序维度或直接Linear映射

        # 或可选：通过final_fc再映射一次（如果需要）
        out = self.final_fc(x_seq)  # [B,512]
        if len(out.shape) == 2:
            out = out.unsqueeze(1)
        return out
