"""MOB Grid Search — optimize Silhouette Coefficient"""
import os, sys, warnings, gc
import numpy as np
import scanpy as sc
from sklearn.metrics import silhouette_score
import torch

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"
warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/bio/lhz/spatialGDC")
import spCLUE

adata_raw = sc.read_h5ad("./results/Mouse_OB_our.h5ad")
n_clusters = 7
print(f"MOB spots: {adata_raw.shape[0]}, clusters: {n_clusters}")

g_spatial = spCLUE.prepare_graph(adata_raw, "spatial")
g_expr = spCLUE.prepare_graph(adata_raw, "expr")
graph_dict = {"spatial": g_spatial, "expr": g_expr}
spatial_coords = adata_raw.obsm["spatial"].copy()
spatial_coords = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))

# Fixed subsample for consistent SC comparison across runs
N_SAMPLE = 5000
rng = np.random.RandomState(42)
sample_idx = rng.choice(adata_raw.shape[0], N_SAMPLE, replace=False)

grid = {
    'sigma': [0.3, 0.4, 0.5, 0.6, 0.7],
    'gamma': [1.0, 2.0, 3.0, 5.0],
    'kappa': [0.02, 0.05, 0.1, 0.2],
    'beta': [0.5, 1.0, 2.0],
}
total = len(grid['sigma'])*len(grid['gamma'])*len(grid['kappa'])*len(grid['beta'])
print(f"Grid: {total} combos")

best_sc = -1.0
best_info = {}
combo = 0

for s in grid['sigma']:
    for g in grid['gamma']:
        for k in grid['kappa']:
            for b in grid['beta']:
                combo += 1
                print(f"\n[{combo}/{total}] sigma:{s} gamma:{g} kappa:{k} beta:{b}")
                try:
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    spCLUE.fix_seed(0)
                    adata = adata_raw.copy()
                    expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords, sigma=s)
                    model = spCLUE.spCLUE(
                        input_data=adata.obsm["X_pca"].copy(),
                        graph_dict=graph_dict, n_clusters=n_clusters,
                        expr_keep_prob=expr_keep_prob, gamma=g, kappa=k, beta=b,
                    )
                    _, emb, _ = model.train()
                    adata.obsm["emb"] = emb
                    spCLUE.clustering(adata, n_clusters, key="emb", refinement=True, cluster_methods="mclust")
                    cc = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
                    labels = adata.obs[cc].values.astype(int)
                    sc_val = silhouette_score(emb[sample_idx], labels[sample_idx])
                    print(f"  SC={sc_val:.4f}")
                    if sc_val > best_sc:
                        best_sc = sc_val
                        best_info = {"sigma": s, "gamma": g, "kappa": k, "beta": b, "SC": round(sc_val,4)}
                        print(f"  >>> NEW BEST! SC={sc_val:.4f}")
                except Exception as e:
                    print(f"  FAIL: {e}")

print(f"\nMOB Best: SC={best_info['SC']:.4f} (sigma={best_info['sigma']}, gamma={best_info['gamma']}, kappa={best_info['kappa']}, beta={best_info['beta']})")
with open("SpatialGDC_MOB_SC_v2.txt", "w") as f:
    f.write(f"MOB Best SC: {best_info['SC']:.4f} (sigma={best_info['sigma']}, gamma={best_info['gamma']}, kappa={best_info['kappa']}, beta={best_info['beta']})\n")
