"""
Unified Ablation Runner — DLPFC 12 Slices
=========================================
ablation: "no_aug" | "no_intersection_cl" | "no_sp_prior"

All use same locked config: attention fusion, kappa=0.10 total, gamma=1, beta=1
"""

import os, sys, warnings, gc, argparse
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

parser = argparse.ArgumentParser()
parser.add_argument("--ablation", required=True, choices=["no_aug", "no_intersection_cl", "no_sp_prior", "full"])
args = parser.parse_args()

ABLATION = args.ablation
NAME_MAP = {
    "no_aug": "NoAug (2-view only)",
    "no_intersection_cl": "NoIntersectionCL (std InfoNCE)",
    "no_sp_prior": "NoSPprior (uniform DropEdge for V4)",
    "full": "Full 4-view (kappa=0.1, attention fusion)",
}

samples_list = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672",
                "151673", "151674", "151675", "151676"]

results = []
print(f"Ablation: {NAME_MAP[ABLATION]}")

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

        # SP-prior DropEdge
        if ABLATION == "no_sp_prior":
            expr_keep_prob = None  # uniform DropEdge
        else:
            spatial_coords = adata.obsm["spatial"].copy()
            spatial_coords = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))
            expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords, sigma=0.5)

        # Ablation-specific params
        kwargs = dict(
            input_data=adata.obsm["X_pca"].copy(),
            graph_dict=graph_dict,
            n_clusters=n_clusters,
            expr_keep_prob=expr_keep_prob,
            kappa=0.1,
            fusion_type="attention",
        )
        if ABLATION == "no_aug":
            kwargs["num_views"] = 2
        if ABLATION == "no_intersection_cl":
            kwargs["use_intersection_cl"] = False

        model = spCLUE.spCLUE(**kwargs)
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
    print(f"\n{ABLATION} Mean ARI = {valid['ARI'].mean():.4f} | Mean NMI = {valid['NMI'].mean():.4f}")
df.to_csv(f"ablation_{ABLATION}_Results.csv", index=False)
print(df.to_string(index=False))
