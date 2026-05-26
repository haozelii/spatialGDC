"""BRCA Enhanced Grid Search v2 — wider ranges based on v1 best (sigma=0.5, gamma=2.0, kappa=0.1)"""
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

brca_dir = "./dataset/BRCA1/V1_Human_Breast_Cancer_Block_A_Section_1"
print(f"Loading BRCA: {brca_dir}")
adata_raw = sc.read_visium(brca_dir)
adata_raw.var_names_make_unique()

sc.pp.filter_genes(adata_raw, min_counts=1)
sc.pp.normalize_total(adata_raw, target_sum=1e4)
sc.pp.log1p(adata_raw)
sc.pp.highly_variable_genes(adata_raw, flavor="seurat_v3", n_top_genes=3000)
adata_raw = adata_raw[:, adata_raw.var.highly_variable].copy()
sc.pp.scale(adata_raw)
sc.tl.pca(adata_raw, svd_solver='arpack', n_comps=200)

df_meta = pd.read_csv(os.path.join("./dataset/BRCA1", 'metadata.tsv'), sep='\t', index_col=0)
adata_raw.obs['Region'] = df_meta.loc[adata_raw.obs_names, 'fine_annot_type']
n_clusters = adata_raw.obs['Region'].nunique()
print(f"Spots: {adata_raw.shape[0]}, clusters: {n_clusters}")

# Round 1: Explore graph construction (n_neighbors, self_weight)
# Round 2: Explore finer kappa and gamma
# Round 3: Explore beta

g_spatial_base = spCLUE.prepare_graph(adata_raw, "spatial")
g_expr_base = spCLUE.prepare_graph(adata_raw, "expr")
graph_dict = {"spatial": g_spatial_base, "expr": g_expr_base}
spatial_coords = adata_raw.obsm["spatial"].copy()
spatial_coords = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))

best_ari = 0.622  # from v1
best_info = {}

# Grid: sigma, gamma, kappa, beta — fine-grained around v1 best
grid = {
    'sigma': [0.3, 0.4, 0.5, 0.6, 0.7],
    'gamma': [1.5, 2.0, 3.0, 5.0],
    'kappa': [0.02, 0.05, 0.1, 0.2],
    'beta': [0.5, 1.0, 2.0],
}

total = len(grid['sigma'])*len(grid['gamma'])*len(grid['kappa'])*len(grid['beta'])
print(f"Grid: {total} combos (sigma×gamma×kappa×beta)")

combo = 0
for s in grid['sigma']:
    for g in grid['gamma']:
        for k in grid['kappa']:
            for b in grid['beta']:
                combo += 1
                print(f"\n[{combo}/{total}] BRCA sigma:{s} gamma:{g} kappa:{k} beta:{b}")
                try:
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    spCLUE.fix_seed(0)
                    adata = adata_raw.copy()
                    expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr_base, spatial_coords, sigma=s)
                    model = spCLUE.spCLUE(
                        input_data=adata.obsm["X_pca"].copy(),
                        graph_dict=graph_dict, n_clusters=n_clusters,
                        expr_keep_prob=expr_keep_prob, gamma=g, kappa=k, beta=b,
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
                        best_info = {"sigma": s, "gamma": g, "kappa": k, "beta": b, "ARI": round(ari,4), "NMI": round(nmi,4)}
                        print(f"  >>> NEW BEST! ARI={ari:.4f}")
                except Exception as e:
                    print(f"  FAIL: {e}")

print(f"\nBRCA Best: ARI={best_info['ARI']:.4f} NMI={best_info['NMI']:.4f} "
      f"(sigma={best_info['sigma']}, gamma={best_info['gamma']}, kappa={best_info['kappa']}, beta={best_info['beta']})")
pd.DataFrame([best_info]).to_csv("SpatialGDC_BRCA_GridSearch_v2.csv", index=False)
