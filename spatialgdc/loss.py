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
        
        if self.use_union:
            fn_mask = (adj_spatial_dense > 0) | (adj_expr_dense > 0)
        else:
            fn_mask = (adj_spatial_dense > 0) & (adj_expr_dense > 0)
        fn_mask.fill_diagonal_(False)
        
        sim_matrix = (x @ xbar.T) / self.temperature
        
        pos_sim = (x * xbar).sum(dim=1) / self.temperature  
        
        sim_matrix_masked = sim_matrix.clone()
        if self.fn_penalty > 0:
            if self.adaptive_eta:
                if self.adaptive_mode == "soft":
                    soft_weights = torch.min(adj_spatial_dense, adj_expr_dense)
                    soft_weights.fill_diagonal_(0)
                    deg_H = soft_weights.sum(dim=1)
                else:
                    deg_H = fn_mask.float().sum(dim=1)

                nonzero_mask = deg_H > 0
                if nonzero_mask.any():
                    avg_deg_H = deg_H[nonzero_mask].mean()
                    deg_row = deg_H.unsqueeze(1)
                    deg_col = deg_H.unsqueeze(0)
                    eta_raw = self.fn_penalty * (deg_row + deg_col) / (2.0 * avg_deg_H + 1e-8)
                    if self.adaptive_mode == "floor":
                        eta_matrix = torch.clamp(eta_raw, min=self.eta_floor * self.fn_penalty)
                    else:
                        eta_matrix = eta_raw
                    sim_matrix_masked[fn_mask] = sim_matrix_masked[fn_mask] - eta_matrix[fn_mask]
            else:
                sim_matrix_masked[fn_mask] = sim_matrix_masked[fn_mask] - self.fn_penalty
        
        sim_matrix_masked.fill_diagonal_(float('-inf'))
        
        neg_scores_logsumexp = torch.logsumexp(sim_matrix_masked, dim=1)  
        
        pos_log = pos_sim
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