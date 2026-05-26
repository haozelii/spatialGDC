"""
BRCA 4-View — Locked R7_K05 Config
====================================
Dataset: BRCA1 (20 fine_annot_type classes, ~3798 spots)
Framework: 4 views → Attention fusion → Recon + ICL + ClusterCL
"""

import os, sys, warnings, gc
import pandas as pd, numpy as np, scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"
warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/bio/lhz/spatialGDC")
import spCLUE

BRCA_DIR = "/home/bio/lhz/spatialGDC/dataset/BRCA1"
SAMPLE = "V1_Human_Breast_Cancer_Block_A_Section_1"

print("=" * 60)
print(f"BRCA 4-View: {SAMPLE}")
print("=" * 60)

# Load
data_path = os.path.join(BRCA_DIR, SAMPLE)
adata = sc.read_visium(data_path)
adata.var_names_make_unique()
print(f"Raw: {adata.shape}")

# Metadata with fine_annot_type
meta = pd.read_csv(os.path.join(BRCA_DIR, "metadata.tsv"), sep='\t', index_col=0)
adata.obs = adata.obs.join(meta, how='left')
adata.obs['Region'] = adata.obs['fine_annot_type']
n_clusters = adata.obs['Region'].nunique()
print(f"Classes: {n_clusters}")

# Preprocess (same as DLPFC)
sc.pp.filter_genes(adata, min_counts=1)
sc.pp.filter_cells(adata, min_counts=1)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.scale(adata)

from sklearn.decomposition import PCA
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

_, adata.obsm["emb"], _ = model.train()

# Clustering
try:
    spCLUE.clustering(adata, n_clusters, key="emb", refinement=True, cluster_methods="mclust")
except:
    spCLUE.clustering(adata, n_clusters, key="emb", refinement=False, cluster_methods="mclust")

cluster_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
adata_eval = adata[adata.obs.Region.notna()].copy()
ARI = adjusted_rand_score(adata_eval.obs["Region"], adata_eval.obs[cluster_col])
NMI = normalized_mutual_info_score(adata_eval.obs["Region"], adata_eval.obs[cluster_col])

print(f"\nBRCA: ARI = {ARI:.4f} | NMI = {NMI:.4f}")

pd.DataFrame([{"Dataset": "BRCA", "ARI": ARI, "NMI": NMI, "Spots": adata.shape[0], "Clusters": n_clusters}]).to_csv(
    "4view_BRCA_Results.csv", index=False)
print("Saved: 4view_BRCA_Results.csv")

del model, adata; gc.collect()
torch.cuda.empty_cache()
