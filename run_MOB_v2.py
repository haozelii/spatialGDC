"""MOB Silhouette Coefficient — run model on preprocessed h5ad"""
import os, sys, warnings
import numpy as np
import scanpy as sc
from sklearn.metrics import silhouette_score

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"
warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/bio/lhz/spatialGDC")
import spCLUE

adata_raw = sc.read_h5ad("./results/Mouse_OB_our.h5ad")
n_clusters = 7

g_spatial = spCLUE.prepare_graph(adata_raw, "spatial")
g_expr = spCLUE.prepare_graph(adata_raw, "expr")
graph_dict = {"spatial": g_spatial, "expr": g_expr}
spatial_coords = adata_raw.obsm["spatial"].copy()
spatial_coords = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))
expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords, sigma=0.5)

spCLUE.fix_seed(0)
model = spCLUE.spCLUE(
    input_data=adata_raw.obsm["X_pca"].copy(),
    graph_dict=graph_dict, n_clusters=n_clusters,
    expr_keep_prob=expr_keep_prob, gamma=1.0, kappa=0.1,
)
_, emb, _ = model.train()
adata_raw.obsm["emb"] = emb

spCLUE.clustering(adata_raw, n_clusters, key="emb", refinement=True, cluster_methods="mclust")
cc = 'mclust_refined' if 'mclust_refined' in adata_raw.obs.columns else 'mclust'
labels = adata_raw.obs[cc].values.astype(int)

# Subsample for SC computation (19527 spots too large for full matrix)
n_sample = min(5000, len(labels))
idx = np.random.RandomState(0).choice(len(labels), n_sample, replace=False)
sc = silhouette_score(emb[idx], labels[idx])
print(f"MOB Silhouette Coefficient (n={n_sample}): {sc:.4f}")
with open("SpatialGDC_MOB_SC_v2.txt", "w") as f:
    f.write(f"MOB Silhouette Coefficient (sample n={n_sample}): {sc:.4f}\n")
