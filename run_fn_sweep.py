"""DLPFC fn_penalty sweep — maximize Full vs w/o SoftPenalty gap"""
import os, sys, warnings, gc
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score
import torch

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/bio/lhz/spatialGDC")
import spCLUE

# Test on top 4 DLPFC slices for speed
samples = ["151507", "151508", "151671", "151673"]
fn_values = [1.0, 2.0, 3.0, 5.0, 8.0]
results = []

for sample in samples:
    data_path = f"./dataset/DLPFC/{sample}/"
    adata_raw = spCLUE.load_and_preprocess_st(data_path=data_path)
    nc = 5 if sample in ["151669","151670","151671","151672"] else 7
    gs = spCLUE.prepare_graph(adata_raw, "spatial")
    ge = spCLUE.prepare_graph(adata_raw, "expr")
    gd = {"spatial": gs, "expr": ge}
    sc_ = adata_raw.obsm["spatial"].copy()
    sc_ = (sc_ - sc_.min(axis=0)) / (sc_.max(axis=0) - sc_.min(axis=0))
    
    for fn in fn_values:
        ari_list = []
        for seed in [0, 42, 123]:
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            spCLUE.fix_seed(seed)
            adata = adata_raw.copy()
            ekp = spCLUE.compute_spatial_keep_prob(ge, sc_, sigma=0.5)
            m = spCLUE.spCLUE(
                input_data=adata.obsm["X_pca"].copy(), graph_dict=gd,
                n_clusters=nc, expr_keep_prob=ekp,
                gamma=1.0, kappa=0.1, fn_penalty=fn,
            )
            _, emb, _ = m.train()
            adata.obsm["emb"] = emb
            try:
                spCLUE.clustering(adata, nc, key="emb", refinement=True, cluster_methods="mclust")
            except:
                spCLUE.clustering(adata, nc, key="emb", refinement=False, cluster_methods="mclust")
            cc = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
            ae = adata[adata.obs.Region.notna()]
            ari = adjusted_rand_score(ae.obs["Region"], ae.obs[cc])
            ari_list.append(ari)
        m_ari = np.mean(ari_list)
        results.append({"sample": sample, "fn_penalty": fn, "ARI": round(m_ari, 4)})
        print(f"  {sample} fn={fn}: ARI={m_ari:.4f}")

# Summary
import pandas as pd
df = pd.DataFrame(results)
print("\n=== Summary by fn_penalty ===")
for fn in fn_values:
    sub = df[df["fn_penalty"] == fn]
    print(f"fn={fn}: mean ARI={sub['ARI'].mean():.4f}")
df.to_csv("SpatialGDC_DLPFC_fn_sweep.csv", index=False)
