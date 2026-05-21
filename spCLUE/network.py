"""
4-View 2-Original + 2-Augmented Dual-Graph CGCN
=================================================
V1: spatial 原始 (NoiseLayer YES, DropEdge NO)  — preserve topology
V2: expr   原始 (NoiseLayer YES, DropEdge NO)  — preserve topology
V3: spatial 增强 (NoiseLayer YES, DropEdge 0.4) — perturb topology
V4: expr   增强 (NoiseLayer YES, SP-DropEdge)   — Innovation 1

CL Pair1 (cross-topo): V1(orig_spa) ↔ V4(aug_expr_SP)  — Innovation pairing
CL Pair2 (cross-topo): V3(aug_spa)  ↔ V2(orig_expr)    — Standard pairing
CL Pair3 (same-topo):  V1(orig_spa) ↔ V3(aug_spa)      — Aug invariance
CL Pair4 (same-topo):  V2(orig_expr)↔ V4(aug_expr)     — Aug invariance

kappa budget: cross-topo 0.03×2 + same-topo 0.02×2 = 0.10 total
4-way Attention fusion
Reconstruction: relu(z@W2.T) @ W1.T (2-step)
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


class CCGCN(Module):

    def __init__(self, dims_list, n_clusters, graph_corr=0.4, dropout=0.5) -> None:
        super(CCGCN, self).__init__()
        self.input_dim = dims_list[0]
        self.hidden_dim = dims_list[1]
        self.z_dim = dims_list[2]
        self.dropout = dropout
        self.n_clusters = n_clusters
        self.graph_corr = graph_corr

        self.noiseLayer = NoiseLayer(dropout=self.dropout)
        self.Transform1 = TransForm_W(self.input_dim, self.hidden_dim, self.dropout)
        self.Transform2 = TransForm_W(self.hidden_dim, self.z_dim, self.dropout)

        self.act = nn.ELU()
        self.relu = nn.ReLU()
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
            scaled_values = adj._values() * mask / custom_keep_prob
            return torch.sparse_coo_tensor(adj._indices(), scaled_values, size=adj.size())
        else:
            dropped_values = F.dropout(adj._values(), p=self.graph_corr, training=True)
            return torch.sparse_coo_tensor(adj._indices(), dropped_values, size=adj.size())

    def encoder(self, data, adj, custom_keep_prob=None, drop_edge=True):
        """
        drop_edge=True:  NoiseLayer + DropEdge (增强view)
        drop_edge=False: NoiseLayer only, no DropEdge (原始view, preserve topology)
        """
        # Always apply NoiseLayer (feature noise is essential for CL)
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

    def forward(self, data, adj1, adj2, adj2_keep_prob=None, use_spatial_drop=True, batch_onehot=None):
        actual_keep_prob = adj2_keep_prob if use_spatial_drop else None

        # ── 2 原始 view (NoiseLayer YES, DropEdge NO) ──
        z1 = self.encoder(data, adj1, drop_edge=False)          # V1: spatial 原始
        z2 = self.encoder(data, adj2, drop_edge=False)          # V2: expr 原始

        # ── 2 增强 view (NoiseLayer YES, DropEdge YES) ──
        z3 = self.encoder(data, adj1, drop_edge=True)           # V3: spatial 增强
        z4 = self.encoder(data, adj2, custom_keep_prob=actual_keep_prob, drop_edge=True)  # V4: expr 增强(SP)

        z1_norm = normalize(z1, p=2, dim=1)
        z2_norm = normalize(z2, p=2, dim=1)
        z3_norm = normalize(z3, p=2, dim=1)
        z4_norm = normalize(z4, p=2, dim=1)

        h1_norm = normalize(self.projectInsHead(z1_norm), p=2, dim=1)
        h2_norm = normalize(self.projectInsHead(z2_norm), p=2, dim=1)
        h3_norm = normalize(self.projectInsHead(z3_norm), p=2, dim=1)
        h4_norm = normalize(self.projectInsHead(z4_norm), p=2, dim=1)

        label1 = self.projectClsHead(z1_norm)
        label2 = self.projectClsHead(z2_norm)
        label3 = self.projectClsHead(z3_norm)
        label4 = self.projectClsHead(z4_norm)

        z = torch.stack([z1_norm, z2_norm, z3_norm, z4_norm], dim=1)
        z_fuse, _ = self.attention(z)

        # 2-step reconstruction with intermediate ReLU (CRITICAL)
        x_Rec = self.relu(z_fuse @ self.Transform2.W.data.T) @ self.Transform1.W.data.T

        return h1_norm, h2_norm, h3_norm, h4_norm, z_fuse, label1, label2, label3, label4, x_Rec


class CCGCNs(Module):
    """Batch-corrected version."""

    def __init__(
        self,
        dims_list,
        n_clusters,
        n_batches,
        graph_corr=0.4,
        dropout=0.5,
        device="cuda:0"
    ) -> None:
        super(CCGCNs, self).__init__()
        self.input_dim = dims_list[0]
        self.hidden_dim = dims_list[1]
        self.z_dim = dims_list[2]
        self.dropout = dropout
        self.n_clusters = n_clusters
        self.n_batches = n_batches
        self.graph_corr = graph_corr

        self.noiseLayer = NoiseLayer()
        self.Transform1 = TransForm_W(self.input_dim, self.hidden_dim, self.dropout)
        self.Transform2 = TransForm_W(self.hidden_dim, self.z_dim, self.dropout)

        self.batchPortion = torch.eye(self.n_batches).to(device)
        self.weightBatch = 0.01

        if self.weightBatch != 0:
            self.batchEmbed = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(self.n_batches, self.z_dim))
            )
            self.batchPCA = nn.Parameter(
                nn.init.xavier_normal_(torch.empty(self.n_batches, self.input_dim))
            )
        else:
            self.batchEmbed = torch.ones(self.n_batches, self.z_dim).to(device)
            self.batchPCA = torch.ones(self.n_batches, self.input_dim).to(device)

        self.act = nn.ELU()
        self.relu = nn.ReLU()
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

    def forward(self, data, adj1, adj2, adj2_keep_prob=None, batch_onehot=None):
        batch_list = batch_onehot
        batch_noise = self.batchPortion[batch_list] @ self.batchPCA
        data = data - self.weightBatch * batch_noise

        # 2 original + 2 augmented
        z1 = self.encoder(data, adj1, drop_edge=False)
        z2 = self.encoder(data, adj2, drop_edge=False)
        z3 = self.encoder(data, adj1, drop_edge=True)
        z4 = self.encoder(data, adj2, custom_keep_prob=adj2_keep_prob, drop_edge=True)

        z1_norm = normalize(z1, p=2, dim=1)
        z2_norm = normalize(z2, p=2, dim=1)
        z3_norm = normalize(z3, p=2, dim=1)
        z4_norm = normalize(z4, p=2, dim=1)

        h1_norm = normalize(self.projectInsHead(z1_norm), p=2, dim=1)
        h2_norm = normalize(self.projectInsHead(z2_norm), p=2, dim=1)
        h3_norm = normalize(self.projectInsHead(z3_norm), p=2, dim=1)
        h4_norm = normalize(self.projectInsHead(z4_norm), p=2, dim=1)

        label1 = self.projectClsHead(z1_norm)
        label2 = self.projectClsHead(z2_norm)
        label3 = self.projectClsHead(z3_norm)
        label4 = self.projectClsHead(z4_norm)

        z = torch.stack([z1_norm, z2_norm, z3_norm, z4_norm], dim=1)
        z_fuse, _ = self.attention(z)

        norm_batch_embed = normalize(self.batchEmbed, p=2, dim=1)
        z_dec = z_fuse + self.weightBatch * self.batchPortion[batch_list] @ norm_batch_embed

        x_Rec = self.relu(z_dec @ self.Transform2.W.data.T) @ self.Transform1.W.data.T

        return h1_norm, h2_norm, h3_norm, h4_norm, z_fuse, label1, label2, label3, label4, x_Rec


class InnerProductDec(nn.Module):

    def __init__(self, dropout=0.2) -> None:
        super().__init__()
        self.dropout = dropout

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj_rec = z @ z.T
        return adj_rec


class IdentityMap(nn.Module):

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x


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
