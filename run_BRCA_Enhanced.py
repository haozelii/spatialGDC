"""
BRCA 基因增强版：用最优参数重新跑一次，保存增强基因表达
"""
import os
import sys
import warnings
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch
import gc

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"
warnings.filterwarnings("ignore")

spCLUE_ROOT_PATH = "/home/bio/lhz/spatialGDC"
if spCLUE_ROOT_PATH not in sys.path:
    sys.path.append(spCLUE_ROOT_PATH)
import spCLUE

print("🚀 BRCA 基因增强版 (最优参数: sigma=0.7, gamma=2.0, kappa=0.1)")

# 最优参数
BEST_SIGMA = 0.7
BEST_GAMMA = 2.0
BEST_KAPPA = 0.1

# 加载数据
brca_dir = "./dataset/BRCA1/V1_Human_Breast_Cancer_Block_A_Section_1"
meta_path = "./dataset/BRCA1/metadata.tsv"

adata = sc.read_visium(brca_dir)
adata.var_names_make_unique()

meta_df = pd.read_csv(meta_path, sep='\t', index_col=0)
adata.obs = adata.obs.join(meta_df, how='left')

# 预处理
sc.pp.filter_genes(adata, min_counts=1)
sc.pp.filter_cells(adata, min_counts=1)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata)

spCLUE.fix_seed(0)

# PCA 保留 pc_matrix 用于基因逆变换
pca_model = PCA(n_components=200, random_state=0)
adata.obsm["X_pca"] = pca_model.fit_transform(adata.X)
pc_matrix = pca_model.components_.T  # (genes, 200)

# 构建图
g_spatial = spCLUE.prepare_graph(adata, "spatial")
g_expr = spCLUE.prepare_graph(adata, "expr")
graph_dict = {"spatial": g_spatial, "expr": g_expr}

spatial_coords = adata.obsm["spatial"].copy()
spatial_coords_norm = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))
expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords_norm, sigma=BEST_SIGMA)

n_clusters = adata.obs['fine_annot_type'].dropna().nunique()

# 训练
model = spCLUE.spCLUE(
    input_data=adata.obsm["X_pca"],
    graph_dict=graph_dict,
    n_clusters=n_clusters,
    expr_keep_prob=expr_keep_prob,
    gamma=BEST_GAMMA,
    kappa=BEST_KAPPA
)

_, adata.obsm["SpatialGDC_emb"], best_x_rec_pca = model.train()

# 基因增强逆变换
if torch.is_tensor(best_x_rec_pca):
    best_x_rec_pca = best_x_rec_pca.detach().cpu().numpy()
x_rec_gene = np.dot(best_x_rec_pca, pc_matrix.T)
adata.layers["SpatialGDC_enhanced"] = x_rec_gene
print(f"✨ 基因增强完成: {x_rec_gene.shape}")

# 聚类
spCLUE.clustering(adata, n_clusters, key="SpatialGDC_emb", refinement=True, cluster_methods="mclust")
cluster_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
adata.obs['domain'] = adata.obs[cluster_col].astype('category')

# 评估
gt_col = 'fine_annot_type'
y_true = adata.obs[gt_col].astype(str)
y_pred = adata.obs['domain'].astype(str)
ari = adjusted_rand_score(y_true, y_pred)
nmi = normalized_mutual_info_score(y_true, y_pred)
print(f"🏆 ARI={ari:.4f}, NMI={nmi:.4f}")

# 保存 (带增强基因层)
adata.write_h5ad("./benchmarking/baselines/results/BRCA_our.h5ad")
adata.write_h5ad("./results_best/BRCA_BEST.h5ad")
print("💾 已保存带 SpatialGDC_enhanced 层的 h5ad")
