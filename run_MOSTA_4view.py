"""
MOSTA 4-View — Locked R7_K05 Config
=====================================
Dataset: Mouse Embryo E9.5 (12 organ annotations, ~14k spots)
Framework: 4 views → Attention fusion → Recon + ICL + ClusterCL
"""

import os, sys, warnings, gc
import pandas as pd, numpy as np, scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
import torch

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"
warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/bio/lhz/spatialGDC")
import spCLUE

print("=" * 60)
print("MOSTA 4-View: Mouse Embryo E9.5")
print("=" * 60)

# Load
adata = sc.read_h5ad("./dataset/Mouse_Embryo/E9.5_E1S1.MOSTA.h5ad")
adata.var_names_make_unique()
print(f"Raw: {adata.shape}")

# Filter: keep annotated spots only
adata = adata[adata.obs['annotation'].notna()].copy()
adata.obs['Region'] = adata.obs['annotation'].astype(str)
n_clusters = adata.obs['annotation'].nunique()
print(f"Annotated spots: {adata.shape[0]}, classes: {n_clusters}")

# Preprocess
sc.pp.filter_genes(adata, min_counts=1)
sc.pp.filter_cells(adata, min_counts=1)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata)

adata.obsm["X_pca"] = PCA(n_components=200, random_state=0).fit_transform(adata.X)
print(f"Post-PCA: {adata.obsm['X_pca'].shape}")

# Graphs
g_spatial = spCLUE.prepare_graph(adata, "spatial")
g_expr = spCLUE.prepare_graph(adata, "expr")
graph_dict = {"spatial": g_spatial, "expr": g_expr}

spatial_coords = adata.obsm["spatial"].copy()
spatial_coords = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))
expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords, sigma=0.5)

spCLUE.fix_seed(0)
model = spCLUE.spCLUE(
    input_data=adata.obsm["X_pca"].copy(),
    graph_dict=graph_dict,
    n_clusters=n_clusters,
    expr_keep_prob=expr_keep_prob,
    kappa=0.1,
    fusion_type="attention",
)

_, adata.obsm["emb"], best_x_rec = model.train()

# Convert reconstructed PCA back to gene space for enhanced expression
if torch.is_tensor(best_x_rec):
    best_x_rec = best_x_rec.detach().cpu().numpy()
# (skip gene-space conversion for MOSTA since we don't have PC matrix stored)

# Clustering
try:
    spCLUE.clustering(adata, n_clusters, key="emb", refinement=True, cluster_methods="mclust")
except Exception as e:
    print(f"mclust refinement failed: {e}")
    spCLUE.clustering(adata, n_clusters, key="emb", refinement=False, cluster_methods="mclust")

cluster_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
adata_eval = adata[adata.obs.Region.notna()].copy()
ARI = adjusted_rand_score(adata_eval.obs["Region"], adata_eval.obs[cluster_col])
NMI = normalized_mutual_info_score(adata_eval.obs["Region"], adata_eval.obs[cluster_col])

print(f"\nMOSTA: ARI = {ARI:.4f} | NMI = {NMI:.4f}")

pd.DataFrame([{"Dataset": "MOSTA", "ARI": ARI, "NMI": NMI, "Spots": adata.shape[0], "Clusters": n_clusters}]).to_csv(
    "4view_MOSTA_Results.csv", index=False)
print("Saved: 4view_MOSTA_Results.csv")

del model, adata; gc.collect()
torch.cuda.empty_cache()
