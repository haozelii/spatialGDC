"""
4-View R7 Training Loop — PROVEN CONFIG (kappa=0.05/pair)
==========================================================
V1(spatial) <-> V3(expr+sp_prior) : primary cross-topo
V2(spatial) <-> V4(expr+uniform) : auxiliary cross-topo
ICL soft penalty on both pairs <- Innovation 2
CRITICAL: kappa=0.05 per pair (total=0.1, matches 2-view)
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
        self.kappa = kappa / 2.0  # CRITICAL: halved for 2 CL pairs
        self.dims_list = [dim_input, dim_hidden, dim_embed]
        self.n_spot = input_data.shape[0]
        self.use_instance_cl = use_instance_cl
        
        fix_seed(self.random_seed)
        self.input_data = torch.FloatTensor(input_data).to(self.device)
        self.g_spatial = sparse_mx_to_torch_sparse_tensor(graph_dict["spatial"]).to(
            self.device
        )
        self.g_expr = sparse_mx_to_torch_sparse_tensor(graph_dict["expr"]).to(
            self.device
        )
        
        if expr_keep_prob is not None:
            self.expr_keep_prob = torch.FloatTensor(expr_keep_prob).to(self.device)
        else:
            self.expr_keep_prob = None

        if batch_list is None:
            self.model = CCGCN(
                self.dims_list, self.n_clusters, self.graph_corr, dropout
            ).to(self.device)
        else:
            self.n_batches = len(set(batch_list))
            self.epochs = 500
            self.batchList = torch.LongTensor(batch_list).to(self.device)
            self.batch_train = batch_train
            self.model = CCGCNs(
                self.dims_list, self.n_clusters, self.n_batches, self.graph_corr
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
                _, _, _, _, features_fuse, _, _, _, _, x_rec = self.model(
                    self.input_data, 
                    self.g_spatial, 
                    self.g_expr, 
                    adj2_keep_prob=self.expr_keep_prob,
                    batch_onehot=self.batchList
                )
                features_fuse = features_fuse.detach().cpu().numpy()
                return features_fuse

            _, _, _, _, features_fuse, _, _, _, _, x_rec = self.model(
                self.input_data, 
                self.g_spatial, 
                self.g_expr,
                adj2_keep_prob=self.expr_keep_prob 
            )
            predLabel = self.model.getCluster(features_fuse)
            features_fuse = features_fuse.detach().cpu().numpy()
            predLabel = predLabel.detach().cpu().numpy()
            x_rec = x_rec.detach().cpu().numpy()
            
            return predLabel, features_fuse, x_rec

    def train(self):
        self.instance_crit = IntersectionContrastiveLoss()
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

            (
                h1, h2, h3, h4,
                output_fuse,
                label1, label3, label2, label4,
                x_rec,
            ) = self.model(
                self.input_data, 
                self.g_spatial, 
                self.g_expr,
                adj2_keep_prob=self.expr_keep_prob
            )
            
            if self.use_instance_cl:
                ic_13 = (
                    self.instance_crit(h1, h3, adj_spatial_dense, adj_expr_dense)
                    + self.instance_crit(h3, h1, adj_spatial_dense, adj_expr_dense)
                ) / 2
                ic_24 = (
                    self.instance_crit(h2, h4, adj_spatial_dense, adj_expr_dense)
                    + self.instance_crit(h4, h2, adj_spatial_dense, adj_expr_dense)
                ) / 2
                cur_contrastive_loss = ic_13 + ic_24
            else:
                cur_contrastive_loss = torch.tensor(0.0).to(self.device)
            
            cur_cluster_loss = self.cluster_crit(label1, label3) + self.cluster_crit(label2, label4)
            cur_rec_expr_loss = self.rec_crit(x_rec, self.input_data)

            cur_batch_loss = (
                self.kappa * cur_contrastive_loss
                + self.beta * cur_cluster_loss
                + self.gamma * cur_rec_expr_loss
            )
            cur_batch_loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                predLabel1_np = label1.detach().cpu().numpy().argmax(axis=1)
                predLabel3_np = label3.detach().cpu().numpy().argmax(axis=1)
                cur_ari = adjusted_rand_score(predLabel1_np, predLabel3_np)
                print(f"epoch {epoch + 1}: cross-topo ARI = {cur_ari:.4f}")
                if cur_ari >= max_ari:
                    predLabel, features_fuse, x_rec = self.updateResult()
                    return predLabel, features_fuse, x_rec

        print("Training Finished =================<")
        with torch.no_grad():
            self.model.eval()
            _, _, _, _, features_fuse, _, _, _, _, x_rec = self.model(
                self.input_data, 
                self.g_spatial, 
                self.g_expr,
                adj2_keep_prob=self.expr_keep_prob
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

            (
                h1, h2, h3, h4,
                output_fuse,
                label1, label3, label2, label4,
                x_rec,
            ) = self.model(
                self.input_data, 
                self.g_spatial, 
                self.g_expr, 
                adj2_keep_prob=self.expr_keep_prob,
                batch_onehot=self.batchList
            )

            cur_loss_id = self.loss_idx()
            
            cur_contrastive_loss = (
                self.instance_crit(h1[cur_loss_id], h3[cur_loss_id])
                + self.instance_crit(h3[cur_loss_id], h1[cur_loss_id])
                + self.instance_crit(h2[cur_loss_id], h4[cur_loss_id])
                + self.instance_crit(h4[cur_loss_id], h2[cur_loss_id])
            ) / 4
            
            cur_cluster_loss = self.cluster_crit(
                label1[cur_loss_id], label3[cur_loss_id]
            ) + self.cluster_crit(
                label2[cur_loss_id], label4[cur_loss_id]
            )
            
            cur_rec_expr_loss = self.rec_crit(x_rec, self.input_data)

            cur_batch_loss = (
                self.kappa * cur_contrastive_loss
                + self.gamma * cur_rec_expr_loss
                + self.beta * cur_cluster_loss
            )

            cur_batch_loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 100 == 0:
                predLabel1_np = label1.detach().cpu().numpy().argmax(axis=1)
                predLabel3_np = label3.detach().cpu().numpy().argmax(axis=1)
                cur_ari = round(adjusted_rand_score(predLabel1_np, predLabel3_np), 2)
                print(f"epoch {epoch + 1}: {cur_ari}")

                if epoch + 1 == 100:
                    self.kappa, self.beta = 0.0, 1.0

                if cur_ari >= max_ari:
                    features_fuse = self.updateResult(batch_case=True)
                    return "hello", features_fuse
        print("Training Finished =================<")

        with torch.no_grad():
            self.model.eval()
            _, _, _, _, features_fuse, _, _, _, _, _ = self.model(
                self.input_data, 
                self.g_spatial, 
                self.g_expr, 
                adj2_keep_prob=self.expr_keep_prob,
                batch_onehot=self.batchList
            )
            features_fuse = features_fuse.detach().cpu().numpy()

        return "hello", features_fuse
