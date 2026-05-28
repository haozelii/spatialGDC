"""BRCA V3: SpatialGDC experiment with best params from grid search"""
import os, sys, warnings, gc
import numpy as np, pandas as pd, scanpy as sc
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import PCA
import torch

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/bio/lhz/spatialGDC")
import spCLUE

# V3 best params
SIGMA, GAMMA, KAPPA, BETA = 0.6, 5.0, 0.05, 2.0
SEED = 0

BRCA_PATH = "./dataset/BRCA1/V1_Human_Breast_Cancer_Block_A_Section_1"

print("Loading BRCA data...")
adata = sc.read_visium(BRCA_PATH)
adata.var_names_make_unique()
adata.obs_names = adata.obs_names.str.replace('-1', '', regex=False).str.strip()

# Preprocessing (match grid search exactly)
sc.pp.filter_genes(adata, min_counts=1)
sc.pp.filter_cells(adata, min_counts=1)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
adata = adata[:, adata.var.highly_variable].copy()
sc.pp.scale(adata)

from sklearn.decomposition import PCA
pca = PCA(n_components=200, random_state=SEED)
adata.obsm["X_pca"] = pca.fit_transform(adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X)

# Build graphs
g_spatial = spCLUE.prepare_graph(adata, "spatial")
g_expr = spCLUE.prepare_graph(adata, "expr")
graph_dict = {"spatial": g_spatial, "expr": g_expr}

spatial_coords = adata.obsm["spatial"].copy()
spatial_coords = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))
expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords, sigma=SIGMA)

# Load ground truth
meta = pd.read_csv("./dataset/BRCA1/metadata.tsv", sep='\t', index_col=0)
meta.index = meta.index.astype(str).str.replace('-1', '', regex=False).str.strip()
gt_col = 'fine_annot_type'
n_clusters = meta[gt_col].nunique()
print(f"n_clusters = {n_clusters}")

spCLUE.fix_seed(SEED)
model = spCLUE.spCLUE(
    input_data=adata.obsm["X_pca"].copy(),
    graph_dict=graph_dict, n_clusters=n_clusters,
    expr_keep_prob=expr_keep_prob,
    gamma=GAMMA, kappa=KAPPA, beta=BETA,
    use_spatial_drop=True, use_intersection_cl=True,
)

print("Training...")
predLabel, features_fuse, x_rec = model.train()

adata.obsm["emb"] = features_fuse

# mclust refinement
try:
    spCLUE.clustering(adata, n_clusters, key="emb", refinement=True, cluster_methods="mclust")
except:
    spCLUE.clustering(adata, n_clusters, key="emb", refinement=False, cluster_methods="mclust")

cluster_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
adata.obs['domain'] = adata.obs[cluster_col].astype('category')

# Evaluate
common = meta.index.intersection(adata.obs_names)
y_true = meta.loc[common, gt_col]
y_pred = adata.obs.loc[common, 'domain']
ari = adjusted_rand_score(y_true, y_pred)
print(f"BRCA SpatialGDC V3: ARI = {ari:.4f}")

# Save
out_path = "./benchmarking/baselines/results/BRCA_our.h5ad"
adata.write_h5ad(out_path)
print(f"Saved to {out_path}")
