"""
4-View 2-Orig + 2-Aug with switchable fusion
fusion_type: "attention" (original) | "ctcf" (Cross-Topology Complementary Fusion)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.functional import normalize


class NoiseLayer(nn.Module):
    def __init__(self, alpha=0.01, dropout=0.5) -> None:
        super().__init__()
        self.alpha = alpha
        self.drop = dropout

    def forward(self, x):
        gauss_x = x + self.alpha * torch.randn_like(x)
        return F.dropout(gauss_x, self.drop, training=self.training)


class AttentionBlock(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class ConcatProjectionFusion(nn.Module):
    """
    Concat + Projection Fusion
    
    Simplest cross-view interaction: concatenate all 4 views,
    project back to d with a small MLP. Captures all pairwise
    and higher-order interactions implicitly.
    
    Innovation angle: "Multi-View Interactive Projection"
    — concat allows all cross-view correlations to be learned
    — LayerNorm stabilizes training
    — Skip connection preserves baseline
    """
    def __init__(self, d=24, n_views=4):
        super().__init__()
        self.n_views = n_views
        self.proj = nn.Sequential(
            nn.Linear(n_views * d, d),
            nn.LayerNorm(d),
        )
        self.view_weights = nn.Parameter(torch.ones(n_views) / n_views)
        
    def forward(self, *views):
        # Linear baseline (weighted average)
        w = F.softmax(self.view_weights, dim=0)
        linear_fuse = sum(w[i] * views[i] for i in range(self.n_views))
        # Cross-view interaction via concat+proj
        concat = torch.cat(views, dim=-1)
        interaction = self.proj(concat)
        # Residual: keep linear baseline, add interaction
        fused = linear_fuse + interaction
        return fused


class PerDimGatedFusion(nn.Module):
    """
    Per-Dimension Gated Fusion
    
    Improvement over scalar attention: each dimension gets its own
    per-view weight, allowing the model to select which view is
    most informative for each latent dimension.
    
    Example: dim 5 might come from spatial (location info),
    while dim 12 might come from expression (cell type).
    Scalar attention can't express this; per-dim gating can.
    """
    def __init__(self, d=24, n_views=4, hidden=32):
        super().__init__()
        self.n_views = n_views
        # Gating network: concat → per-dim per-view weights
        self.gate = nn.Sequential(
            nn.Linear(n_views * d, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_views * d),
        )
        self.view_weights = nn.Parameter(torch.ones(n_views) / n_views)
        
    def forward(self, *views):
        # Linear baseline
        w = F.softmax(self.view_weights, dim=0)
        linear_fuse = sum(w[i] * views[i] for i in range(self.n_views))
        
        # Per-dim gating
        concat = torch.cat(views, dim=-1)  # (N, n_views*d)
        gate_logits = self.gate(concat).view(-1, self.n_views, len(views[0].T))  # (N, V, d)
        gate_weights = F.softmax(gate_logits, dim=1)  # softmax over views
        stacked = torch.stack(views, dim=1)  # (N, V, d)
        gated_fuse = (gate_weights * stacked).sum(dim=1)  # (N, d)
        
        # Mix gated and linear (learnable balance)
        fused = gated_fuse
        return fused


class TwoLevelFusion(nn.Module):
    """
    2-Level Hierarchical Fusion — Innovation 3
    ===========================================
    Level 1 (within-topology): ConcatProj fusion for same-topology pairs
      V1+V3 (both spatial) → Z_spa  —  ConcatProjection(d, n_views=2)
      V2+V4 (both expr)   → Z_expr  —  ConcatProjection(d, n_views=2)
    
    Level 2 (cross-topology): ConcatProj fusion for cross-topology
      Z_spa + Z_expr → Z_fuse       —  ConcatProjection(d, n_views=2)
    
    Design rationale:
    - L1: same-topology views share GCN weights but differ in augmentation.
      ConcatProj allows learning which augmentation matters per dimension.
    - L2: genuinely different topological views → ConcatProj with residual.
    
    Both levels use proven ConcatProjection (residual connection + LayerNorm).
    The hierarchy is in WHAT gets fused when, not in reducing fusion complexity.
    """
    def __init__(self, d=24):
        super().__init__()
        # Level 1: within-topology fusion (2 views each)
        self.spa_fusion = ConcatProjectionFusion(d=d, n_views=2)
        self.expr_fusion = ConcatProjectionFusion(d=d, n_views=2)
        
        # Level 2: cross-topology fusion (2 fused views)
        self.cross_fusion = ConcatProjectionFusion(d=d, n_views=2)
        
    def forward(self, z1, z2, z3, z4):
        # ── Level 1: within-topology fusion ──
        z_spa = self.spa_fusion(z1, z3)  # spatial: orig + aug
        z_expr = self.expr_fusion(z2, z4)  # expression: orig + aug
        
        # ── Level 2: cross-topology fusion ──
        z_fuse = self.cross_fusion(z_spa, z_expr)
        
        return z_fuse, z_spa, z_expr


class SpatialGDCNet(Module):

    def __init__(self, dims_list, n_clusters, graph_corr=0.4, dropout=0.5, fusion_type="attention", num_views=4) -> None:
        super(SpatialGDCNet, self).__init__()
        self.input_dim = dims_list[0]
        self.hidden_dim = dims_list[1]
        self.z_dim = dims_list[2]
        self.dropout = dropout
        self.n_clusters = n_clusters
        self.graph_corr = graph_corr
        self.fusion_type = fusion_type
        self.num_views = num_views  # 4=full, 2=ablation (no aug)

        self.noiseLayer = NoiseLayer(dropout=self.dropout)
        self.Transform1 = TransForm_W(self.input_dim, self.hidden_dim, self.dropout)
        self.Transform2 = TransForm_W(self.hidden_dim, self.z_dim, self.dropout)

        self.act = nn.ELU()
        self.relu = nn.ReLU()
        
        if fusion_type == "concat_proj":
            self.fusion = ConcatProjectionFusion(d=self.z_dim, n_views=num_views)
        elif fusion_type == "perdim_gate":
            self.fusion = PerDimGatedFusion(d=self.z_dim, n_views=num_views)
        elif fusion_type == "two_level":
            self.fusion = TwoLevelFusion(d=self.z_dim)
        elif fusion_type == "ctcf":
            self.fusion = ConcatProjectionFusion(d=self.z_dim, n_views=num_views)  # fallback
        else:
            self.attention = AttentionBlock(self.z_dim)

        self.projectInsHead = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
        )

        self.projectClsHead = nn.Sequential(
            nn.Linear(self.z_dim, self.z_dim),
            nn.ReLU(),
            nn.Linear(self.z_dim, self.n_clusters),
            nn.Softmax(dim=1),
        )

    def get_dropped_adj(self, adj, custom_keep_prob=None):
        if not self.training:
            return adj
        if custom_keep_prob is not None:
            mask = torch.bernoulli(custom_keep_prob).to(adj.device)
            scaled_values = adj._values() * mask / (custom_keep_prob + 1e-8)
            return torch.sparse_coo_tensor(adj._indices(), scaled_values, size=adj.size())
        else:
            dropped_values = F.dropout(adj._values(), p=self.graph_corr, training=True)
            return torch.sparse_coo_tensor(adj._indices(), dropped_values, size=adj.size())

    def encoder(self, data, adj, custom_keep_prob=None, drop_edge=True):
        feature = self.noiseLayer(data)
        if drop_edge:
            adj1_dropped = self.get_dropped_adj(adj, custom_keep_prob)
        else:
            adj1_dropped = adj
        feature = self.act(torch.spmm(adj1_dropped, self.Transform1(feature)))
        if drop_edge:
            adj2_dropped = self.get_dropped_adj(adj, custom_keep_prob)
        else:
            adj2_dropped = adj
        feature = self.act(torch.spmm(adj2_dropped, self.Transform2(feature)))
        return feature

    def getCluster(self, embed):
        labels = self.projectClsHead(embed)
        return torch.argmax(labels, dim=1)

    def _fuse_pair(self, va, vb):
        """Fuse exactly 2 views (used in both Level 1 and Level 2)."""
        if self.fusion_type == "attention":
            z = torch.stack([va, vb], dim=1)  # (N, 2, d)
            z_fuse, _ = self.attention(z)
        elif self.fusion_type == "two_level":
            z_fuse, _, _ = self.fusion(va, vb, va, vb)
        else:
            z_fuse = self.fusion(va, vb)
        return z_fuse

    def _fuse(self, z1_norm, z2_norm, z3_norm, z4_norm):
        if self.fusion_type == "attention":
            z = torch.stack([z1_norm, z2_norm, z3_norm, z4_norm], dim=1)
            z_fuse, _ = self.attention(z)
        elif self.fusion_type == "two_level":
            z_fuse, _, _ = self.fusion(z1_norm, z2_norm, z3_norm, z4_norm)
        else:
            z_fuse = self.fusion(z1_norm, z2_norm, z3_norm, z4_norm)
        return z_fuse

    def forward(self, data, adj1, adj2, adj2_keep_prob=None, use_spatial_drop=True, batch_onehot=None):
        """
        2-Level Hierarchical Fusion + Simplified Contrastive Learning

        Level 1: same-topo fusion (V1+V3→z_spa, V2+V4→z_expr)
                 → Instance CL on (h_spa, h_expr)  — 1 pair
                 → Cluster CL on (label_spa, label_expr) — 1 pair
        Level 2: cross-topo fusion (z_spa+z_expr→z_fuse)
                 → Reconstruction from z_fuse

        Returns: (h_spa, h_expr, z_fuse, label_spa, label_expr, x_rec)
        """
        actual_keep_prob = adj2_keep_prob if use_spatial_drop else None

        if self.num_views == 2:
            # 2-view: no augmentation, z_spa=z1, z_expr=z2
            z1 = self.encoder(data, adj1, drop_edge=False)
            z2 = self.encoder(data, adj2, drop_edge=False)

            z_spa = normalize(z1, p=2, dim=1)
            z_expr = normalize(z2, p=2, dim=1)

            h_spa = normalize(self.projectInsHead(z_spa), p=2, dim=1)
            h_expr = normalize(self.projectInsHead(z_expr), p=2, dim=1)

            label_spa = self.projectClsHead(z_spa)
            label_expr = self.projectClsHead(z_expr)

            z_fuse = self._fuse_pair(z_spa, z_expr)
            x_rec = self.relu(z_fuse @ self.Transform2.W.data.T) @ self.Transform1.W.data.T

            return h_spa, h_expr, z_fuse, label_spa, label_expr, x_rec

        # ── 4-View Encoding ──
        z1 = self.encoder(data, adj1, drop_edge=False)   # V1: orig spa
        z2 = self.encoder(data, adj2, drop_edge=False)   # V2: orig expr
        z3 = self.encoder(data, adj1, drop_edge=True)    # V3: aug spa (DropEdge)
        z4 = self.encoder(data, adj2, custom_keep_prob=actual_keep_prob, drop_edge=True)  # V4: aug expr (SP-DropEdge)

        z1_norm = normalize(z1, p=2, dim=1)
        z2_norm = normalize(z2, p=2, dim=1)
        z3_norm = normalize(z3, p=2, dim=1)
        z4_norm = normalize(z4, p=2, dim=1)

        # ── Level 1: Same-Topology Fusion ──
        # Fuse orig+aug within each topology → 消除增强差异
        z_spa = self._fuse_pair(z1_norm, z3_norm)   # spatial: V1 + V3
        z_expr = self._fuse_pair(z2_norm, z4_norm)  # expression: V2 + V4

        # ── Projection Heads (on fused representations) ──
        h_spa = normalize(self.projectInsHead(z_spa), p=2, dim=1)
        h_expr = normalize(self.projectInsHead(z_expr), p=2, dim=1)

        label_spa = self.projectClsHead(z_spa)
        label_expr = self.projectClsHead(z_expr)

        # ── Level 2: Cross-Topology Fusion ──
        # Fuse spatial+expression → 融合互补信息
        z_fuse = self._fuse_pair(z_spa, z_expr)

        # ── Reconstruction ──
        x_rec = self.relu(z_fuse @ self.Transform2.W.data.T) @ self.Transform1.W.data.T

        return h_spa, h_expr, z_fuse, label_spa, label_expr, x_rec


class TransForm_W(nn.Module):
    def __init__(self, input_dim, out_dim, dropout=0.5, act=None) -> None:
        super().__init__()
        self.dropout = dropout
        self.W = nn.Parameter(
            nn.init.xavier_uniform_(torch.empty(input_dim, out_dim))
        )

    def forward(self, x):
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x @ self.W
