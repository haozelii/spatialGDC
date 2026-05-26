"""BRCA Ablation — Full vs w/o SP-DropEdge vs w/o SoftPenalty"""
import os, sys, warnings, gc
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score
import torch

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/bio/lhz/spatialGDC")
import spCLUE

brca_dir = "./dataset/BRCA1/V1_Human_Breast_Cancer_Block_A_Section_1"
adata_raw = sc.read_visium(brca_dir)
adata_raw.var_names_make_unique()
sc.pp.filter_genes(adata_raw, min_counts=1)
sc.pp.normalize_total(adata_raw, target_sum=1e4)
sc.pp.log1p(adata_raw)
sc.pp.highly_variable_genes(adata_raw, flavor="seurat_v3", n_top_genes=3000)
adata_raw = adata_raw[:, adata_raw.var.highly_variable].copy()
sc.pp.scale(adata_raw)
sc.tl.pca(adata_raw, svd_solver='arpack', n_comps=200)
meta = pd.read_csv("./dataset/BRCA1/metadata.tsv", sep='\t', index_col=0)
adata_raw.obs['Region'] = meta.loc[adata_raw.obs_names, 'fine_annot_type']
n_clusters = adata_raw.obs['Region'].nunique()
print(f"BRCA: {adata_raw.shape[0]} spots, {n_clusters} clusters")

gs = spCLUE.prepare_graph(adata_raw, "spatial")
ge = spCLUE.prepare_graph(adata_raw, "expr")
gd = {"spatial": gs, "expr": ge}
sc_ = adata_raw.obsm["spatial"].copy()
sc_ = (sc_ - sc_.min(axis=0)) / (sc_.max(axis=0) - sc_.min(axis=0))

# Best BRCA: sigma=0.6, gamma=5.0, kappa=0.05, beta=2.0
BEST = {"sigma": 0.6, "gamma": 5.0, "kappa": 0.05, "beta": 2.0}
SEEDS = [0, 42, 123, 999, 2024]
results = []

for variant, use_sp, use_intcl in [
    ("Full", True, True),
    ("w/o SP-DropEdge", False, True),
    ("w/o SoftPenalty", True, False),
]:
    ari_list = []
    for seed in SEEDS:
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        spCLUE.fix_seed(seed)
        adata = adata_raw.copy()
        ekp = spCLUE.compute_spatial_keep_prob(ge, sc_, sigma=BEST["sigma"])
        m = spCLUE.spCLUE(
            input_data=adata.obsm["X_pca"].copy(), graph_dict=gd,
            n_clusters=n_clusters, expr_keep_prob=ekp,
            gamma=BEST["gamma"], kappa=BEST["kappa"], beta=BEST["beta"],
            use_spatial_drop=use_sp, use_intersection_cl=use_intcl,
        )
        _, emb, _ = m.train()
        adata.obsm["emb"] = emb
        spCLUE.clustering(adata, n_clusters, key="emb", refinement=True, cluster_methods="mclust")
        cc = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
        ae = adata[adata.obs.Region.notna()]
        ari = adjusted_rand_score(ae.obs["Region"], ae.obs[cc])
        ari_list.append(ari)
        print(f"  {variant} seed={seed}: ARI={ari:.4f}")
    m_ari = np.mean(ari_list)
    results.append({"Variant": variant, "Mean_ARI": round(m_ari, 4), "Std_ARI": round(np.std(ari_list), 4)})
    print(f"  >>> {variant}: mean ARI={m_ari:.4f}")

df = pd.DataFrame(results)
df.to_csv("SpatialGDC_BRCA_Ablation_v2.csv", index=False)
print("\nDone!", df.to_string())
