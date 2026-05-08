import scanpy as sc
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import numpy as np
import scipy.sparse as sp
import os
import pandas as pd 

import scanpy as sc
import pandas as pd
import os
from sklearn.decomposition import PCA

def load_and_preprocess_st(data_path, meta_filename="metadata.tsv"):
    """
    完全复现原作者 0.76 ARI 结果的数据预处理函数 (全量基因 + sklearn PCA)
    """
    print(f"--- 正在从 {data_path} 加载数据 ---")
    adata = sc.read_visium(data_path)
    adata.var_names_make_unique()
    print(f"原始数据维度: {adata.shape[0]} spots, {adata.shape[1]} genes")

    # ==========================================
    # 1. 加载 Metadata 标签 (原样保留)
    # ==========================================
    meta_path = os.path.join(data_path, meta_filename)
    if os.path.exists(meta_path):
        meta_df = pd.read_csv(meta_path, sep='\t', index_col=0)
        adata.obs = adata.obs.join(meta_df, how='left')
        if 'layer_guess' in adata.obs.columns:
            adata.obs.rename(columns={'layer_guess': 'Region'}, inplace=True)
        print(f"✅ 标签合并成功！")

    # ==========================================
    # 2. 1:1 复现原作者的预处理数学逻辑
    #    (不选 HVG，不扔基因，保留全部 33538 个特征)
    # ==========================================
    print("--- 执行过滤、归一化与缩放 ---")
    sc.pp.filter_genes(adata, min_counts=1)
    sc.pp.filter_cells(adata, min_counts=1)
    
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # 按照原代码，不使用 max_value 截断
    sc.pp.scale(adata)

    # ==========================================
    # 3. 1:1 复现原作者的 sklearn PCA
    # ==========================================
    print("--- 使用 sklearn 进行 200 维全量基因 PCA 降维 (可能需要十几秒) ---")
    # 必须指定 random_state=0 保证特征矩阵绝对一致
    adata.obsm["X_pca"] = PCA(n_components=200, random_state=0).fit_transform(adata.X)
    
    print(f"✅ 预处理完成！用于训练的 X_pca 维度: {adata.obsm['X_pca'].shape}")
    
    return adata


def calcGAEParams(graph, n_samples):
    '''graph is a bipartite graph, return pos_weight and norm_val
    '''
    non_zero_cnt = graph.sum()
    norm_val = (n_samples * n_samples) / (2 * (n_samples * n_samples - non_zero_cnt))
    pos_weight = (n_samples * n_samples - non_zero_cnt) / non_zero_cnt
    return norm_val, pos_weight


def calcGraphWeight(coor, eps=1e-6):
    dist = cdist(coor, coor, "euclidean")
    dist = dist / (np.max(dist) + eps)
    return dist


def correlation_graph(A, B):
    '''calculate correlation between A and B.
    Args:
        A (np.ndarray): sample matrix, shape: [samples, features].
        B (np.ndarray): sample matrix, shape: [samples, features].
    Returns: 
        corr (np.ndarray): correlation matrix of features, shape: [features, features].
    '''
    am = A - np.mean(A, axis=0, keepdims=True)
    bm = B - np.mean(B, axis=0, keepdims=True)
    return am.T @ bm / (np.sqrt(np.sum(am**2, axis=0, keepdims=True)).T * np.sqrt(np.sum(bm**2, axis=0, keepdims=True)))


def prepare_graph(adata, key="spatial", n_neighbors=12, n_comps=50, eps=1e-8, svd_solver="randomized", self_weight=0.3):
    n_spots = adata.shape[0]
    assert key in ["spatial", "expr"], "case should be [spatial] or [expr]"
    if key == "spatial":
        print("create adjacent matrix from spatial idx --------------->")
        expr = adata.obsm[key]
        weights = 1. / (cdist(expr, expr, "euclidean") + eps)
    else:
        print("create adjacent matrix from pca expr --------------->")
        expr = PCA(n_components=n_comps, random_state=0, svd_solver=svd_solver).fit_transform(adata.X)
        weights = correlation_graph(expr.T, expr.T)

    print("create knn graph ---->")
    threshold = np.sort(weights)[:, -n_neighbors - 1:-n_neighbors]
    weights[weights < threshold] = 0
    weights = (weights + weights.T) / 2
    weights = weights * (1 - np.eye(n_spots))  # drop the diag

    adjFilter = 0. if key == "spatial" else 0.1
    # convert to bipartite case
    adjBip = np.where(weights > adjFilter, 1, 0)
    print(f"{key} knn graph created ----<")

    return sp.coo_matrix(symm_norm(adjBip, weightDiag=self_weight))

def symm_norm(adj, weightDiag=.3, eps=1e-8):
    '''
    args: adjacent matrix with diag = 0
    return: D^{-1/2} (A + I) D^{-1 / 2}
    '''
    n_spot = adj.shape[0]
    adj_self = (1 - weightDiag) * adj + np.eye(n_spot) * weightDiag  
    degrees = 1. / np.sqrt((np.sum(adj_self, axis=1) + eps))
    adj_self *= degrees
    adj_self *= degrees[:, None]
    return adj_self.astype(np.float32)

# 将这个函数添加到 preprocess.py 的最下方
# def compute_spatial_keep_prob(adj_coo, spatial_coords, sigma=0.5):
#     """
#     在模型外部（预处理阶段）计算特征图中每条边的保留概率
#     Args:
#         adj_coo (scipy.sparse.coo_matrix): 特征图的稀疏矩阵
#         spatial_coords (np.ndarray): 细胞的物理空间坐标，通常是 adata.obsm['spatial']
#         sigma (float): 高斯核带宽，控制距离衰减惩罚力度 (建议取值 0.1 ~ 0.5)
#     Returns:
#         np.ndarray: 每条边对应的保留概率数组
#     """
#     print(f"--- 计算特征图的空间先验保留概率 (sigma={sigma}) ---")
#     # scipy 的 coo_matrix 使用 .row 和 .col 获取边索引
#     src = adj_coo.row
#     dst = adj_coo.col
    
#     # 获取每条边两端的物理坐标
#     src_coords = spatial_coords[src]
#     dst_coords = spatial_coords[dst]
    
#     # 计算物理距离的平方 (Numpy 计算)
#     dist_sq = np.sum((src_coords - dst_coords) ** 2, axis=1)
    
#     # 归一化距离，避免不同切片的坐标尺度影响超参数 sigma
#     if dist_sq.max() > 0:
#         dist_sq = dist_sq / dist_sq.max()
        
#     # 计算保留概率 P_keep：距离越远，概率越低
#     keep_prob = np.exp(-dist_sq / (2 * sigma ** 2))
    
#     # 设定一个保底概率，即使极其远，也保留 10% 的机会
#     keep_prob = np.clip(keep_prob, 0.1, 1.0) #0.2
    
#     return keep_prob.astype(np.float32)
def compute_spatial_keep_prob(adj_coo, spatial_coords, sigma=0.5):
    src = adj_coo.row
    dst = adj_coo.col
    
    src_coords = spatial_coords[src]
    dst_coords = spatial_coords[dst]
    
    # 物理距离平方
    dist_sq = np.sum((src_coords - dst_coords) ** 2, axis=1)
    
    # 内部做一次 Min-Max 归一化，确保所有切片尺度统一
    if dist_sq.max() > 0:
        dist_sq = (dist_sq - dist_sq.min()) / (dist_sq.max() - dist_sq.min() + 1e-8)
        
    # 🌟 核心修复：温柔衰减 (Soft Decay)
    # 将原来的 [0~1] 衰减，平滑映射到 [0.5 ~ 1.0] 区间
    # 这样长程边最多被砍掉一半，绝不破坏图的整体连通性！
    base_prob = np.exp(-dist_sq / (2 * sigma ** 2))
    keep_prob = 0.5 + 0.5 * base_prob 
    
    return keep_prob.astype(np.float32)