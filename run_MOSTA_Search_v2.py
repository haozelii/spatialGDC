"""MOSTA Grid Search — h5ad loading + manual preprocessing like old script"""
import os, sys, warnings, gc
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"
warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/bio/lhz/spatialGDC")
import spCLUE

mosta_path = "./dataset/Mouse_Embryo/E9.5_E1S1.MOSTA.h5ad"
print(f"Loading MOSTA: {mosta_path}")
adata_raw = sc.read_h5ad(mosta_path)
adata_raw.var_names_make_unique()

gt_col = 'annotation'
adata_raw = adata_raw[adata_raw.obs[gt_col].notna()].copy()

sc.pp.filter_genes(adata_raw, min_counts=1)
sc.pp.filter_cells(adata_raw, min_counts=1)
sc.pp.normalize_total(adata_raw, target_sum=1e4)
sc.pp.log1p(adata_raw)
sc.pp.highly_variable_genes(adata_raw, flavor="seurat_v3", n_top_genes=3000)
adata_raw = adata_raw[:, adata_raw.var.highly_variable].copy()
sc.pp.scale(adata_raw)
sc.tl.pca(adata_raw, svd_solver='arpack', n_comps=200)

adata_raw.obs['Region'] = adata_raw.obs[gt_col]
n_clusters = adata_raw.obs['Region'].nunique()
print(f"Spots: {adata_raw.shape[0]}, clusters: {n_clusters}")

g_spatial = spCLUE.prepare_graph(adata_raw, "spatial")
g_expr = spCLUE.prepare_graph(adata_raw, "expr")
graph_dict = {"spatial": g_spatial, "expr": g_expr}
spatial_coords = adata_raw.obsm["spatial"].copy()
spatial_coords = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))

param_grid = {'sigma': [0.4, 0.5, 0.6], 'gamma': [1.0, 1.5, 2.0], 'kappa': [0.05, 0.1, 0.5]}
best_ari = -1.0
best_info = {}

print(f"Grid: {len(param_grid['sigma'])*len(param_grid['gamma'])*len(param_grid['kappa'])} combos")
for s in param_grid['sigma']:
    for g in param_grid['gamma']:
        for k in param_grid['kappa']:
            print(f"\n[MOSTA] sigma:{s} gamma:{g} kappa:{k}")
            try:
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                spCLUE.fix_seed(0)
                adata = adata_raw.copy()
                expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords, sigma=s)
                model = spCLUE.spCLUE(
                    input_data=adata.obsm["X_pca"].copy(),
                    graph_dict=graph_dict, n_clusters=n_clusters,
                    expr_keep_prob=expr_keep_prob, gamma=g, kappa=k, beta=1.0,
                )
                _, adata.obsm["emb"], _ = model.train()
                spCLUE.clustering(adata, n_clusters, key="emb", refinement=True, cluster_methods="mclust")
                cc = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
                ae = adata[adata.obs.Region.notna()]
                ari = adjusted_rand_score(ae.obs["Region"], ae.obs[cc])
                nmi = normalized_mutual_info_score(ae.obs["Region"], ae.obs[cc])
                print(f"  ARI={ari:.4f} NMI={nmi:.4f}")
                if ari > best_ari:
                    best_ari = ari
                    best_info = {"sigma": s, "gamma": g, "kappa": k, "ARI": round(ari,4), "NMI": round(nmi,4)}
            except Exception as e:
                print(f"  FAIL: {e}")

print(f"\nMOSTA Best: ARI={best_info['ARI']:.4f} NMI={best_info['NMI']:.4f}")
pd.DataFrame([best_info]).to_csv("SpatialGDC_MOSTA_GridSearch_v2.csv", index=False)
