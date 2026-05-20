"""
4-View R7 Dual-Graph Contrastive GCN — PROVEN CONFIG
======================================================
V1: spatial + NoiseLayer + DropEdge(0.4) — independent random
V2: spatial + NoiseLayer + DropEdge(0.4) — independent random  
V3: expr   + NoiseLayer + Spatial-Prior DropEdge — Innovation 1
V4: expr   + NoiseLayer + DropEdge(0.4) — independent random

CL Pair 1: V1 ↔ V3 (primary cross-topo)
CL Pair 2: V2 ↔ V4 (auxiliary cross-topo)
ICL soft penalty on all pairs ← Innovation 2
4-way Attention fusion
Reconstruction: relu(z@W2.T) @ W1.T (2-step with intermediate activation)

CRITICAL: kappa=0.05 per pair (total=0.1, matches 2-view)
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

    def encoder(self, data, adj, custom_keep_prob=None):
        feature = self.noiseLayer(data)
        adj1_dropped = self.get_dropped_adj(adj, custom_keep_prob)
        feature = self.act(torch.spmm(adj1_dropped, self.Transform1(feature)))
        adj2_dropped = self.get_dropped_adj(adj, custom_keep_prob)
        feature = self.act(torch.spmm(adj2_dropped, self.Transform2(feature)))
        return feature

    def getCluster(self, embed):
        labels = self.projectClsHead(embed)
        return torch.argmax(labels, dim=1)

    def forward(self, data, adj1, adj2, adj2_keep_prob=None, use_spatial_drop=True, batch_onehot=None):
        actual_keep_prob = adj2_keep_prob if use_spatial_drop else None

        # 4 independent encoder passes
        z1 = self.encoder(data, adj1, custom_keep_prob=None)
        z2 = self.encoder(data, adj1, custom_keep_prob=None)
        z3 = self.encoder(data, adj2, custom_keep_prob=actual_keep_prob)
        z4 = self.encoder(data, adj2, custom_keep_prob=None)

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

        return h1_norm, h2_norm, h3_norm, h4_norm, z_fuse, label1, label3, label2, label4, x_Rec


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

    def encoder(self, data, adj, custom_keep_prob=None):
        feature = self.noiseLayer(data)
        adj1_dropped = self.get_dropped_adj(adj, custom_keep_prob)
        feature = self.act(torch.spmm(adj1_dropped, self.Transform1(feature)))
        adj2_dropped = self.get_dropped_adj(adj, custom_keep_prob)
        feature = self.act(torch.spmm(adj2_dropped, self.Transform2(feature)))
        return feature

    def getCluster(self, embed):
        labels = self.projectClsHead(embed)
        return torch.argmax(labels, dim=1)

    def forward(self, data, adj1, adj2, adj2_keep_prob=None, batch_onehot=None):
        batch_list = batch_onehot
        batch_noise = self.batchPortion[batch_list] @ self.batchPCA
        data = data - self.weightBatch * batch_noise

        z1 = self.encoder(data, adj1, custom_keep_prob=None)
        z2 = self.encoder(data, adj1, custom_keep_prob=None)
        z3 = self.encoder(data, adj2, custom_keep_prob=adj2_keep_prob)
        z4 = self.encoder(data, adj2, custom_keep_prob=None)

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

        return h1_norm, h2_norm, h3_norm, h4_norm, z_fuse, label1, label3, label2, label4, x_Rec


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
