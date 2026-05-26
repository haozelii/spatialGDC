import torch
from torch import nn
import math
import torch.nn.functional as F

class DisentangleLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z1_s, z2_s, z1_p, z2_p):

        # ① shared 对齐（跨视图一致）
        loss_align = F.mse_loss(z1_s, z2_s)

        # ② 正交（核心）
        loss_orth = (
            (z1_s * z1_p).sum(dim=1).pow(2).mean() +
            (z2_s * z2_p).sum(dim=1).pow(2).mean()
        )

        return loss_align + 0.1 * loss_orth

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, x, xbar, eps=1e-8):
        posScores = torch.exp((x * xbar).sum(dim=1) / self.temperature)
        negScores = torch.exp((x @ xbar.T) / self.temperature).sum(dim=1)
        return -torch.log(posScores / (negScores + eps)).mean()
        
#这是去置信度高的正样本 
# class IntersectionContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.2) -> None:
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, x, xbar, adj_spatial_dense, adj_expr_dense, eps=1e-8):
#         # 0. 🚨 极其重要：对特征进行 L2 归一化，将点积转化为余弦相似度 [-1, 1]
#         x = F.normalize(x, p=2, dim=1)
#         xbar = F.normalize(xbar, p=2, dim=1)
        
#         # 1. 确定“绝对正样本池”：交集邻居 + 对角线（自己）
#         intersection_mask = (adj_spatial_dense > 0) & (adj_expr_dense > 0)
#         pos_mask = intersection_mask | torch.eye(x.size(0), dtype=torch.bool, device=x.device)
        
#         # 2. 计算全局的指数相似度矩阵 (N x N)
#         # 因为做了归一化，(x @ xbar.T) 的最大值就是 1，除以 0.2 后最大值是 5
#         # exp(5) 约等于 148，数值非常稳定，绝对不会爆显存或出现 NaN
#         sim_matrix_exp = torch.exp((x @ xbar.T) / self.temperature)
        
#         # 3. 计算真正的正样本得分 (分子)
#         # posScores = (sim_matrix_exp * pos_mask).sum(dim=1)
#         # 修改后 (Mean): 无论有多少邻居，正样本的能量尺度都和原来一致
#         # pos_mask.sum(dim=1) 算出了每个 spot 有多少个正样本（包含自己）
#         posScores = (sim_matrix_exp * pos_mask).sum(dim=1) / pos_mask.sum(dim=1)
        
#         # 4. 计算所有样本的总得分 (分母 = 正样本 + 负样本)
#         totalScores = sim_matrix_exp.sum(dim=1)
        
#         # 5. 计算最终 Loss
#         return -torch.log(posScores / (totalScores + eps)).mean()
#剔除假负样本
# class IntersectionContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.2) -> None:
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, x, xbar, adj_spatial_dense, adj_expr_dense, eps=1e-8):
#         """
#         基于交集邻域的对比损失（修复版）
        
#         Args:
#             x: [N, d] - 视图1的表示
#             xbar: [N, d] - 视图2的表示
#             adj_spatial_dense: [N, N] - 空间邻接矩阵（稠密）
#             adj_expr_dense: [N, N] - 表达邻接矩阵（稠密）
#             eps: 数值稳定性参数
        
#         Returns:
#             loss: 标量损失值
#         """
#         # 1. 特征 L2 归一化
#         x = F.normalize(x, p=2, dim=1)  # [N, d]
#         xbar = F.normalize(xbar, p=2, dim=1)  # [N, d]
        
#         device = x.device
#         N = x.shape[0]
        
#         # 2. 确保图矩阵是稠密的且维度正确
#         if adj_spatial_dense.shape != (N, N):
#             raise ValueError(
#                 f"adj_spatial_dense shape {adj_spatial_dense.shape} != ({N}, {N})"
#             )
#         if adj_expr_dense.shape != (N, N):
#             raise ValueError(
#                 f"adj_expr_dense shape {adj_expr_dense.shape} != ({N}, {N})"
#             )
        
#         # 3. 找到"假负样本"：空间和基因的交集邻域
#         # 交集邻域 = (在空间图中相邻) AND (在表达图中相邻)
#         intersection_mask = (adj_spatial_dense > 0) & (adj_expr_dense > 0)  # [N, N]
        
#         # 🚨 核心：绝对不能把对角线（自己/正样本）当成负样本
#         intersection_mask.fill_diagonal_(False)
        
#         # 4. 计算相似度矩阵（避免数值溢出，使用 log-sum-exp 技巧）
#         sim_matrix = (x @ xbar.T) / self.temperature  # [N, N]
        
#         # 5. 计算 InfoNCE 损失
#         # 分子：正样本的相似度（自己对自己）
#         pos_sim = (x * xbar).sum(dim=1) / self.temperature  # [N]
#         pos_scores = torch.exp(pos_sim)  # [N]
        
#         # 分母：所有样本 - 假负样本（交集邻域）
#         # 这里使用 logsumexp 的稳定形式
#         sim_matrix_masked = sim_matrix.clone()
#         sim_matrix_masked[intersection_mask] = float('-inf')  # 🌟 改进：使用 -inf 而不是 0
        
#         # 计算分母的对数（logsumexp）
#         neg_scores_logsumexp = torch.logsumexp(sim_matrix_masked, dim=1)  # [N]
        
#         # 最终 loss：使用数值稳定的对数形式
#         # log(pos / (pos + neg_without_hard)) = log(pos) - log(pos + neg)
#         pos_log = torch.log(pos_scores + eps)
#         denom_log = torch.logsumexp(
#             torch.stack([pos_sim, neg_scores_logsumexp], dim=1), 
#             dim=1
#         )  # [N]
        
#         loss = -(pos_log - denom_log).mean()
        
#         return loss

#假负样本排斥了减弱
class IntersectionContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.2, fn_penalty=2.0) -> None:
        super().__init__()
        self.temperature = temperature
        # 🌟 新增：软惩罚系数。值越大，对假负样本的排斥力越弱
        # 如果 fn_penalty = 0，退化为普通 InfoNCE；如果你之前用 -inf，相当于 fn_penalty = 无穷大
        self.fn_penalty = fn_penalty 

    def forward(self, x, xbar, adj_spatial_dense, adj_expr_dense, eps=1e-8):
        x = F.normalize(x, p=2, dim=1)  
        xbar = F.normalize(xbar, p=2, dim=1)  
        
        N = x.shape[0]
        
        # 1. 寻找交集邻域（潜在的假负样本）
        intersection_mask = (adj_spatial_dense > 0) & (adj_expr_dense > 0) 
        intersection_mask.fill_diagonal_(False)
        
        # 2. 计算基础相似度矩阵
        sim_matrix = (x @ xbar.T) / self.temperature  # [N, N]
        
        # 3. 计算正样本得分
        pos_sim = (x * xbar).sum(dim=1) / self.temperature  
        
        # 4. 🌟 核心修改：软权重惩罚 (Soft Masking)
        # 不再使用 -inf，而是将假负样本的相似度强行降低 fn_penalty 个 logit
        # 这意味着在 exp() 之后，它们在分母中的比重会被缩小 exp(fn_penalty) 倍，但不会彻底消失！
        sim_matrix_masked = sim_matrix.clone()
        if self.fn_penalty > 0:
            sim_matrix_masked[intersection_mask] = sim_matrix_masked[intersection_mask] - self.fn_penalty
        
        # 排除掉对角线（自己不是自己的负样本）
        sim_matrix_masked.fill_diagonal_(float('-inf'))
        
        # 5. 计算分母的 logsumexp
        neg_scores_logsumexp = torch.logsumexp(sim_matrix_masked, dim=1)  
        
        # 6. 计算最终 Loss
        pos_log = pos_sim # log(exp(pos_sim)) 就是 pos_sim
        denom_log = torch.logsumexp(
            torch.stack([pos_sim, neg_scores_logsumexp], dim=1), 
            dim=1
        )  
        
        loss = -(pos_log - denom_log).mean()
        
        return loss
class ClusterLoss(nn.Module):
    def __init__(
            self,
            n_classes,
            device=torch.device("cuda:0"),
            temperature=0.2,
    ):
        super(ClusterLoss, self).__init__()
        self.n_classes = n_classes
        self.temperature = temperature
        self.device = device

        self.mask = self.mask_correlated_clusters(n_classes)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, n_classes):
        N = 2 * n_classes
        mask = torch.ones(N, N)
        mask.fill_diagonal_(0)
        for i in range(n_classes):
            mask[i, i + n_classes] = 0
            mask[i + n_classes, i] = 0
        mask = mask.bool()
        return mask

    def normalizeLabel(self, c_i, c_j):
        c_i = torch.square(c_i)
        c_j = torch.square(c_j)
        p_i = c_i.sum(dim=0).view(-1)
        c_i /= p_i
        p_i = c_i.sum(dim=1).view(-1)
        c_i /= p_i.unsqueeze(1)
        p_j = c_j.sum(dim=0).view(-1)
        c_j /= p_j
        p_j = c_j.sum(dim=1).view(-1)
        c_j /= p_j.unsqueeze(1)
        return c_i, c_j

    def forward(self, c_i, c_j):
        p_i = c_i.sum(dim=0).view(-1)
        p_i /= p_i.sum()
        neg_entropy_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        neg_entropy_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        neg_entropy_loss = neg_entropy_i + neg_entropy_j

        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.n_classes
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1),
                                c.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.n_classes)
        sim_j_i = torch.diag(sim, -self.n_classes)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=-1)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss + 1. * neg_entropy_loss

class GraphConsis(nn.Module):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, emb, graphWeight):
        dist1 = torch.cdist(emb, emb, p=2)
        dist1 = torch.div(dist1, torch.max(dist1))
        return torch.mean((1 - dist1) * graphWeight)


class GraphRecLoss(nn.Module):
    def __init__(self, norm_val, pos_weight) -> None:
        super().__init__()
        self.norm_val = norm_val
        self.pos_weight = pos_weight

    def forward(self, emb, target):
        # emb = F.normalize(emb, p=2, dim=1)
        input = emb @ emb.T
        logits = F.binary_cross_entropy_with_logits(input,
                                                    target,
                                                    pos_weight=self.pos_weight)
        return self.norm_val * logits


class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, xbar):
        return torch.square(x - xbar).mean(dim=1).mean()


class ZINBLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, mean, disp, pi=0, scale_factor=1.0, ridge_lambda=0.0):
        '''
        args: x, raw count, [N, hvgs]
              scale_factor, [n,]
        '''
        eps = 1e-10
        mean = (mean.T * scale_factor).T
        if pi == 0:
            pi = torch.tensor(0.0)

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(
            x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (
            x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result