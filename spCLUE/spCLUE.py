"""
2-Level Hierarchical Fusion Training Loop
===========================================
Level 1 (same-topo): V1+V3 -> z_spa, V2+V4 -> z_expr  (within-topology fuse)
Level 2 (cross-topo): z_spa+z_expr -> z_fuse           (cross-topology fuse)

Contrastive Learning (simplified: 1 pair each):
  Instance CL:  ICL(h_spa, h_expr)          — 1 pair, cross-topology alignment
  Cluster CL:   CCL(label_spa, label_expr)   — 1 pair, cross-topology alignment
  Reconstruction: MSE(x_rec, input)
"""

import torch
import numpy as np

from .network import CCGCN, CCGCNs
from tqdm import tqdm
from .loss import ContrastiveLoss, ClusterLoss, MSELoss, IntersectionContrastiveLoss
from .utils import sparse_mx_to_torch_sparse_tensor, adjust_learning_rate, fix_seed

from sklearn.metrics import adjusted_rand_score


class spCLUE:
    def __init__(
        self,
        input_data,
        graph_dict,
        n_clusters=12,
        batch_list=None,
        epochs=500,
        random_seed=0,
        device=torch.device("cuda:0"),
        learning_rate=0.001,
        weight_decay=0.001,
        dim_input=200,
        dim_hidden=64,
        dim_embed=24,
        graph_corr=0.4,
        dropout=0.5,
        gamma=1,
        beta=1,
        kappa=0.1,
        batch_train=False,
        expr_keep_prob=None,
        use_instance_cl=True,
        fusion_type="attention",
        num_views=4,
        use_intersection_cl=True,
        use_spatial_drop=True,
        fn_penalty=2.0,
        use_union=False,
    ):
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs

        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self.graph_corr = graph_corr
        self.gamma = gamma
        self.beta = beta
        self.kappa = kappa
        self.dims_list = [dim_input, dim_hidden, dim_embed]
        self.n_spot = input_data.shape[0]
        self.use_instance_cl = use_instance_cl
        self.fusion_type = fusion_type
        self.num_views = num_views
        self.use_intersection_cl = use_intersection_cl
        self.use_spatial_drop = use_spatial_drop
        self.fn_penalty = fn_penalty
        self.use_union = use_union
        
        fix_seed(self.random_seed)
        self.input_data = torch.FloatTensor(input_data).to(self.device)
        self.g_spatial = sparse_mx_to_torch_sparse_tensor(graph_dict["spatial"]).to(self.device)
        self.g_expr = sparse_mx_to_torch_sparse_tensor(graph_dict["expr"]).to(self.device)
        
        if expr_keep_prob is not None:
            self.expr_keep_prob = torch.FloatTensor(expr_keep_prob).to(self.device)
        else:
            self.expr_keep_prob = None

        if batch_list is None:
            self.model = CCGCN(
                self.dims_list, self.n_clusters, self.graph_corr, dropout,
                fusion_type=self.fusion_type, num_views=self.num_views
            ).to(self.device)
        else:
            self.n_batches = len(set(batch_list))
            self.epochs = 500
            self.batchList = torch.LongTensor(batch_list).to(self.device)
            self.batch_train = batch_train
            self.model = CCGCNs(
                self.dims_list, self.n_clusters, self.n_batches, self.graph_corr,
                fusion_type=self.fusion_type
            ).to(self.device)

    def loss_idx(self):
        tmp = np.arange(self.n_spot)
        if self.batch_train:
            np.random.shuffle(tmp)
            return tmp[:20000]
        return tmp

    def updateResult(self, batch_case=False):
        with torch.no_grad():
            self.model.eval()
            if batch_case:
                _, _, features_fuse, _, _, x_rec = self.model(
                    self.input_data, self.g_spatial, self.g_expr,
                    adj2_keep_prob=self.expr_keep_prob,
                    batch_onehot=self.batchList,
                    use_spatial_drop=self.use_spatial_drop
                )
                features_fuse = features_fuse.detach().cpu().numpy()
                return features_fuse

            _, _, features_fuse, _, _, x_rec = self.model(
                self.input_data, self.g_spatial, self.g_expr,
                adj2_keep_prob=self.expr_keep_prob,
                use_spatial_drop=self.use_spatial_drop
            )
            predLabel = self.model.getCluster(features_fuse)
            features_fuse = features_fuse.detach().cpu().numpy()
            predLabel = predLabel.detach().cpu().numpy()
            x_rec = x_rec.detach().cpu().numpy()
            return predLabel, features_fuse, x_rec

    def train(self):
        if self.use_intersection_cl:
            self.instance_crit = IntersectionContrastiveLoss(fn_penalty=self.fn_penalty, use_union=self.use_union)
        else:
            self.instance_crit = ContrastiveLoss()
        self.cluster_crit = ClusterLoss(self.n_clusters, self.device)
        self.rec_crit = MSELoss()

        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        max_ari = 0.3 if self.n_spot <= 10000 else 1.1
        
        adj_spatial_dense = self.g_spatial.to_dense()
        adj_expr_dense = self.g_expr.to_dense()
        
        print("Training Start =========================>")
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            adjust_learning_rate(self.optimizer, epoch, self.learning_rate)
            self.optimizer.zero_grad()

            h_spa, h_expr, z_fuse, label_spa, label_expr, x_rec = self.model(
                self.input_data, self.g_spatial, self.g_expr,
                adj2_keep_prob=self.expr_keep_prob,
                use_spatial_drop=self.use_spatial_drop
            )
            
            if self.use_instance_cl:
                if self.use_intersection_cl:
                    ic_loss = (
                        self.instance_crit(h_spa, h_expr, adj_spatial_dense, adj_expr_dense)
                        + self.instance_crit(h_expr, h_spa, adj_spatial_dense, adj_expr_dense)
                    ) / 2
                else:
                    ic_loss = (
                        self.instance_crit(h_spa, h_expr)
                        + self.instance_crit(h_expr, h_spa)
                    ) / 2
                cur_contrastive_loss = self.kappa * ic_loss
            else:
                cur_contrastive_loss = torch.tensor(0.0).to(self.device)
            
            cur_cluster_loss = self.cluster_crit(label_spa, label_expr)
            cur_rec_expr_loss = self.rec_crit(x_rec, self.input_data)

            cur_batch_loss = (
                cur_contrastive_loss
                + self.beta * cur_cluster_loss
                + self.gamma * cur_rec_expr_loss
            )
            cur_batch_loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                predLabel_spa = label_spa.detach().cpu().numpy().argmax(axis=1)
                predLabel_expr = label_expr.detach().cpu().numpy().argmax(axis=1)
                cur_ari = adjusted_rand_score(predLabel_spa, predLabel_expr)
                print(f"epoch {epoch + 1}: cross-topo ARI = {cur_ari:.4f}")
                if cur_ari >= max_ari:
                    predLabel, features_fuse, x_rec = self.updateResult()
                    return predLabel, features_fuse, x_rec

        print("Training Finished =================<")
        with torch.no_grad():
            self.model.eval()
            _, _, features_fuse, _, _, x_rec = self.model(
                self.input_data, self.g_spatial, self.g_expr,
                adj2_keep_prob=self.expr_keep_prob,
                use_spatial_drop=self.use_spatial_drop
            )
            predLabel = self.model.getCluster(features_fuse)
            features_fuse = features_fuse.detach().cpu().numpy()
            predLabel = predLabel.detach().cpu().numpy()
            x_rec = x_rec.detach().cpu().numpy()

        return predLabel, features_fuse, x_rec

    def trainBatch(self):
        self.instance_crit = ContrastiveLoss()
        self.rec_crit = MSELoss()
        self.cluster_crit = ClusterLoss(self.n_clusters, self.device)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        max_ari = 0.5
        print("Training Start =========================>")
        for epoch in tqdm(range(self.epochs)):
            self.model.train()
            adjust_learning_rate(self.optimizer, epoch, self.learning_rate)
            self.optimizer.zero_grad()

            h_spa, h_expr, z_fuse, label_spa, label_expr, x_rec = self.model(
                self.input_data, self.g_spatial, self.g_expr,
                adj2_keep_prob=self.expr_keep_prob,
                batch_onehot=self.batchList,
                use_spatial_drop=self.use_spatial_drop
            )

            cur_loss_id = self.loss_idx()
            
            cur_contrastive_loss = self.kappa * (
                self.instance_crit(h_spa[cur_loss_id], h_expr[cur_loss_id])
                + self.instance_crit(h_expr[cur_loss_id], h_spa[cur_loss_id])
            ) / 2
            
            cur_cluster_loss = self.cluster_crit(
                label_spa[cur_loss_id], label_expr[cur_loss_id]
            )
            cur_rec_expr_loss = self.rec_crit(x_rec, self.input_data)

            cur_batch_loss = (
                cur_contrastive_loss
                + self.gamma * cur_rec_expr_loss
                + self.beta * cur_cluster_loss
            )
            cur_batch_loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 100 == 0:
                predLabel_spa = label_spa.detach().cpu().numpy().argmax(axis=1)
                predLabel_expr = label_expr.detach().cpu().numpy().argmax(axis=1)
                cur_ari = round(adjusted_rand_score(predLabel_spa, predLabel_expr), 2)
                print(f"epoch {epoch + 1}: {cur_ari}")
                if epoch + 1 == 100:
                    self.kappa = 0.0
                if cur_ari >= max_ari:
                    features_fuse = self.updateResult(batch_case=True)
                    return "hello", features_fuse

        print("Training Finished =================<")
        with torch.no_grad():
            self.model.eval()
            _, _, features_fuse, _, _, _ = self.model(
                self.input_data, self.g_spatial, self.g_expr,
                adj2_keep_prob=self.expr_keep_prob,
                batch_onehot=self.batchList,
                use_spatial_drop=self.use_spatial_drop
            )
            features_fuse = features_fuse.detach().cpu().numpy()
        return "hello", features_fuse
