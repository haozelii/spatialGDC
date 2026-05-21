import os
import sys
import warnings
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import gc
import torch

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"
warnings.filterwarnings("ignore")

spCLUE_ROOT_PATH = "/home/bio/lhz/spatialGDC"
if spCLUE_ROOT_PATH not in sys.path:
    sys.path.append(spCLUE_ROOT_PATH)
import spCLUE

sample_name = "151507"
spCLUE.fix_seed(0)

data_path = f"./dataset/DLPFC/{sample_name}/"
adata = spCLUE.load_and_preprocess_st(data_path=data_path)

n_clusters = 7
print(f"Spots: {adata.shape[0]}, n_clusters: {n_clusters}")

if 'X_pca' not in adata.obsm.keys() or 'PCs' not in adata.varm.keys():
    print("Running PCA...")
    if 'log1p' not in adata.uns_keys():
        try:
            sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)
        except Exception:
            pass
    sc.tl.pca(adata, svd_solver='arpack', n_comps=200)

pc_matrix = adata.varm['PCs']

g_spatia = spCLUE.prepare_graph(adata, "spatial")
g_expr = spCLUE.prepare_graph(adata, "expr")
graph_dict = {"spatial": g_spatia, "expr": g_expr}

spatial_coords = adata.obsm["spatial"].copy()
spatial_coords = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))
expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords, sigma=0.5)

spCLUE_model = spCLUE.spCLUE(
    input_data=adata.obsm["X_pca"].copy(),
    graph_dict=graph_dict,
    n_clusters=n_clusters,
    expr_keep_prob=expr_keep_prob
)

# 4-view returns: predLabel, features_fuse, x_rec_spa, x_rec_expr
_, adata.obsm["SpatialGDC_emb"], x_rec_spa, x_rec_expr = spCLUE_model.train()

# Use average of dual reconstruction for gene enhancement
best_x_rec_pca = (x_rec_spa + x_rec_expr) / 2

if torch.is_tensor(best_x_rec_pca):
    best_x_rec_pca = best_x_rec_pca.detach().cpu().numpy()

x_rec_gene = np.dot(best_x_rec_pca, pc_matrix.T)
adata.layers["SpatialGDC_enhanced"] = x_rec_gene
print("Gene enhancement done.")

try:
    pred = spCLUE.clustering(adata, n_clusters, key="SpatialGDC_emb", refinement=True, cluster_methods="mclust")
except Exception as e:
    print(f"mclust refinement failed ({e}), trying without refinement...")
    spCLUE.clustering(adata, n_clusters, key="SpatialGDC_emb", refinement=False, cluster_methods="mclust")

cluster_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
adata.obs['domain'] = adata.obs[cluster_col].astype('category')

adata_eval = adata[adata.obs.Region.notna()].copy()
ARI = adjusted_rand_score(adata_eval.obs["Region"], adata_eval.obs["domain"])
NMI = normalized_mutual_info_score(adata_eval.obs["Region"], adata_eval.obs["domain"])

print(f"\n{'='*50}")
print(f"Sample {sample_name}: ARI = {ARI:.4f} | NMI = {NMI:.4f}")
print(f"{'='*50}")
