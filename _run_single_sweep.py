#!/usr/bin/env python3
"""Standalone runner for a single sweep configuration. 
Usage: python _run_single_sweep.py <label> <csv_suffix>
"""
import os, sys, warnings, gc
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"
warnings.filterwarnings("ignore")

ROOT = "/home/bio/lhz/spatialGDC"
if ROOT not in sys.path:
    sys.path.append(ROOT)
import spCLUE

label = sys.argv[1]
suffix = sys.argv[2]

samples = ["151507","151508","151509","151510","151669","151670","151671","151672","151673","151674","151675","151676"]
results = []
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

for sample in samples:
    print(f"\n{'='*50}\n{label}: {sample}\n{'='*50}")
    try:
        spCLUE.fix_seed(0)
        data_path = f"./dataset/DLPFC/{sample}/"
        adata = spCLUE.load_and_preprocess_st(data_path=data_path)
        n_clusters = 5 if sample in ["151669","151670","151671","151672"] else 7
        print(f"Spots: {adata.shape[0]}, n_clusters: {n_clusters}")

        if 'X_pca' not in adata.obsm.keys() or 'PCs' not in adata.varm.keys():
            print("PCA...")
            try:
                sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
                sc.pp.normalize_total(adata, target_sum=1e4)
                sc.pp.log1p(adata)
            except:
                pass
            sc.tl.pca(adata, svd_solver='arpack', n_comps=200)

        g_spatia = spCLUE.prepare_graph(adata, "spatial")
        g_expr = spCLUE.prepare_graph(adata, "expr")
        graph_dict = {"spatial": g_spatia, "expr": g_expr}

        spatial_coords = adata.obsm["spatial"].copy()
        spatial_coords = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))
        expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords, sigma=0.5)

        model = spCLUE.spCLUE(
            input_data=adata.obsm["X_pca"].copy(),
            graph_dict=graph_dict,
            n_clusters=n_clusters,
            expr_keep_prob=expr_keep_prob,
            kappa=0.1,
        )
        
        _, emb, rec = model.train()
        if torch.is_tensor(emb):
            emb = emb.detach().cpu().numpy()
        adata.obsm["SpatialGDC_emb"] = emb

        try:
            spCLUE.clustering(adata, n_clusters, key="SpatialGDC_emb", refinement=True, cluster_methods="mclust")
        except Exception as e:
            print(f"Refine failed: {e}")
            spCLUE.clustering(adata, n_clusters, key="SpatialGDC_emb", refinement=False, cluster_methods="mclust")

        col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
        adata.obs['domain'] = adata.obs[col].astype('category')
        adata_eval = adata[adata.obs.Region.notna()].copy()
        ari = adjusted_rand_score(adata_eval.obs["Region"], adata_eval.obs["domain"])
        nmi = normalized_mutual_info_score(adata_eval.obs["Region"], adata_eval.obs["domain"])
        print(f"{sample}: ARI={ari:.4f} NMI={nmi:.4f}")
        results.append({"Sample": sample, "ARI": ari, "NMI": nmi, "Spots": adata.shape[0], "Clusters": n_clusters})
    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append({"Sample": sample, "ARI": "Error", "NMI": "Error", "Spots": "N/A", "Clusters": "N/A"})
    finally:
        for v in ['model','adata','adata_eval','g_spatia','g_expr','expr_keep_prob','spatial_coords']:
            if v in locals():
                del locals()[v]
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

df = pd.DataFrame(results)
valid = df[df["ARI"] != "Error"]
if not valid.empty:
    print(f"\n{label}: Mean ARI={valid['ARI'].mean():.4f} NMI={valid['NMI'].mean():.4f}")
csv_path = os.path.join(ROOT, f"{suffix}.csv")
df.to_csv(csv_path, index=False)
print(f"Saved {csv_path}")
