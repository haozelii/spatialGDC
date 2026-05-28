import torch
from torch import nn
import math
import torch.nn.functional as F

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
    def __init__(self, temperature=0.2, fn_penalty=2.0, use_union=False,
                 adaptive_eta=True, adaptive_mode="degree", eta_floor=0.5) -> None:
        """
        adaptive_mode:
          - "degree": binary intersection deg_H, formula 2
          - "floor":  same as degree but clamped to [eta_floor * fn_penalty, inf)
          - "soft":   soft intersection: deg_H = sum(min(w_spa, w_expr))
        """
        super().__init__()
        self.temperature = temperature
        self.fn_penalty = fn_penalty
        self.use_union = use_union
        self.adaptive_eta = adaptive_eta
        self.adaptive_mode = adaptive_mode
        self.eta_floor = eta_floor

    def forward(self, x, xbar, adj_spatial_dense, adj_expr_dense, eps=1e-8):
        x = F.normalize(x, p=2, dim=1)  
        xbar = F.normalize(xbar, p=2, dim=1)  
        
        N = x.shape[0]
        
        # 假负样本池：union → 更大池子，intersection → 更精准
        if self.use_union:
            fn_mask = (adj_spatial_dense > 0) | (adj_expr_dense > 0)
        else:
            fn_mask = (adj_spatial_dense > 0) & (adj_expr_dense > 0)
        fn_mask.fill_diagonal_(False)
        
        # 2. 计算基础相似度矩阵
        sim_matrix = (x @ xbar.T) / self.temperature  # [N, N]
        
        # 3. 计算正样本得分
        pos_sim = (x * xbar).sum(dim=1) / self.temperature  
        
        # 4. 🌟 软权重惩罚
        sim_matrix_masked = sim_matrix.clone()
        if self.fn_penalty > 0:
            if self.adaptive_eta:
                if self.adaptive_mode == "soft":
                    # 方案 B: soft intersection — min(adj_spatial, adj_expr) weights
                    soft_weights = torch.min(adj_spatial_dense, adj_expr_dense)  # [N,N]
                    soft_weights.fill_diagonal_(0)
                    deg_H = soft_weights.sum(dim=1)  # weighted degree
                else:
                    # binary intersection degree
                    deg_H = fn_mask.float().sum(dim=1)

                nonzero_mask = deg_H > 0
                if nonzero_mask.any():
                    avg_deg_H = deg_H[nonzero_mask].mean()
                    deg_row = deg_H.unsqueeze(1)  # [N, 1]
                    deg_col = deg_H.unsqueeze(0)  # [1, N]
                    eta_raw = self.fn_penalty * (deg_row + deg_col) / (2.0 * avg_deg_H + 1e-8)
                    if self.adaptive_mode == "floor":
                        # 方案 A: clamp to [eta_floor * fn_penalty, inf)
                        eta_matrix = torch.clamp(eta_raw, min=self.eta_floor * self.fn_penalty)
                    else:
                        eta_matrix = eta_raw
                    sim_matrix_masked[fn_mask] = sim_matrix_masked[fn_mask] - eta_matrix[fn_mask]
            else:
                # Flat penalty: uniform scalar η for all intersection neighbors
                sim_matrix_masked[fn_mask] = sim_matrix_masked[fn_mask] - self.fn_penalty
        
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

class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, xbar):
        return torch.square(x - xbar).mean(dim=1).mean()