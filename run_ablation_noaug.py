"""
Ablation 1: No Augmentation Views (2-view only)
================================================
Remove V3/V4 — only use 2 original views (V1: orig spatial, V2: orig expr)
Same CL pairs adapted for 2 views: V1↔V2 cross-topo contrastive
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

# ── Modified network: 2-view only ──
# We temporarily modify the CCGCN forward to only use z1 (spatial orig) and z2 (expr orig),
# skip z3/z4. The cleanest way: pass use_spatial_drop=False and manually override.
# Actually easier: create a thin wrapper that runs with 2 views.

samples_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", 
                "151673", "151674", "151675", "151676"]

results = []
ROUND = "Ablation_NoAug"

print(f"Ablation {ROUND}: 2-view only (no augmentation)")
print("  V1: orig spatial, V2: orig expr")
print("  CL: V1↔V2 cross-topo only")

for sample_name in samples_list:
    print(f"\n{'='*40}\n{sample_name}\n{'='*40}")
    try:
        spCLUE.fix_seed(0)
        data_path = f"./dataset/DLPFC/{sample_name}/"
        adata = spCLUE.load_and_preprocess_st(data_path=data_path)
        n_clusters = 5 if sample_name in ["151669", "151670", "151671", "151672"] else 7
        print(f"Spots: {adata.shape[0]}, n_clusters: {n_clusters}")

        if 'X_pca' not in adata.obsm.keys():
            sc.tl.pca(adata, svd_solver='arpack', n_comps=200)

        g_spatial = spCLUE.prepare_graph(adata, "spatial")
        g_expr = spCLUE.prepare_graph(adata, "expr")
        graph_dict = {"spatial": g_spatial, "expr": g_expr}

        # No SP-prior DropEdge needed (no V4)
        model = spCLUE.spCLUE(
            input_data=adata.obsm["X_pca"].copy(),
            graph_dict=graph_dict,
            n_clusters=n_clusters,
            expr_keep_prob=None,  # no aug, no prior needed
            kappa=0.1,
            use_instance_cl=True,  # still use ICL but only V1↔V2
            fusion_type="attention",
        )

        _, adata.obsm["emb"], _ = model.train()

        try:
            spCLUE.clustering(adata, n_clusters, key="emb", refinement=True, cluster_methods="mclust")
        except:
            spCLUE.clustering(adata, n_clusters, key="emb", refinement=False, cluster_methods="mclust")

        cluster_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
        adata_eval = adata[adata.obs.Region.notna()].copy()
        ARI = adjusted_rand_score(adata_eval.obs["Region"], adata_eval.obs[cluster_col])
        NMI = normalized_mutual_info_score(adata_eval.obs["Region"], adata_eval.obs[cluster_col])
        print(f"{sample_name}: ARI = {ARI:.4f} | NMI = {NMI:.4f}")
        results.append({"Sample": sample_name, "ARI": ARI, "NMI": NMI, "Spots": adata.shape[0], "Clusters": n_clusters})

    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
        results.append({"Sample": sample_name, "ARI": "Error", "NMI": "Error", "Spots": "N/A", "Clusters": "N/A"})
    finally:
        for v in ['model', 'adata', 'adata_eval']:
            if v in locals(): del locals()[v]
        gc.collect(); torch.cuda.empty_cache()

df = pd.DataFrame(results)
valid = df[df['ARI'] != 'Error']
if not valid.empty:
    print(f"\n{ROUND} Mean ARI = {valid['ARI'].mean():.4f} | Mean NMI = {valid['NMI'].mean():.4f}")
df.to_csv(f"ablation_{ROUND}_Results.csv", index=False)
print(df.to_string(index=False))
