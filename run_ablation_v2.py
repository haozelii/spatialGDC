"""DLPFC Ablation: Full vs w/o SP-DropEdge vs w/o Soft Penalty
Uses best hyperparams from Full model grid search per slice.
Runs 5 seeds each, reports mean ARI.
"""
import os, sys, warnings, gc
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"
warnings.filterwarnings("ignore")

spCLUE_ROOT_PATH = "/home/bio/lhz/spatialGDC"
sys.path.insert(0, spCLUE_ROOT_PATH)
import spCLUE

# Load best params from grid search
df_best = pd.read_csv("SpatialGDC_DLPFC_GridSearch_Summary.csv")
best_params = {}
for _, row in df_best.iterrows():
    best_params[str(int(row["Sample"]))] = {
        "sigma": row["sigma"], "gamma": row["gamma"], "kappa": row["kappa"]
    }

samples_list = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676"
]

SEEDS = [0, 42, 123, 999, 2024]

results = []

for sample_name in samples_list:
    print(f"\n{'='*60}\n  Slice: {sample_name}\n{'='*60}")
    params = best_params[sample_name]
    
    # Load data once per slice
    data_path = f"./dataset/DLPFC/{sample_name}/"
    adata_raw = spCLUE.load_and_preprocess_st(data_path=data_path)
    n_clusters = 5 if sample_name in ["151669", "151670", "151671", "151672"] else 7
    
    if 'Region' not in adata_raw.obs.columns:
        df_meta = pd.read_csv(os.path.join(data_path, 'metadata.tsv'), sep='\t', index_col=0)
        adata_raw.obs['Region'] = df_meta.loc[adata_raw.obs_names, 'layer_guess']
    
    if 'X_pca' not in adata_raw.obsm:
        sc.tl.pca(adata_raw, svd_solver='arpack', n_comps=200)
    
    g_spatial = spCLUE.prepare_graph(adata_raw, "spatial")
    g_expr = spCLUE.prepare_graph(adata_raw, "expr")
    graph_dict = {"spatial": g_spatial, "expr": g_expr}
    
    spatial_coords = adata_raw.obsm["spatial"].copy()
    spatial_coords = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))
    
    s = params["sigma"]
    g = params["gamma"]
    k = params["kappa"]
    
    # Three ablation configs
    configs = [
        ("Full", True, True),
        ("w/o SP-DropEdge", False, True),   # use_spatial_drop=False -> V4 gets uniform DropEdge
        ("w/o SoftPenalty", True, False),    # use_intersection_cl=False -> standard InfoNCE
    ]
    
    for variant, use_sp, use_intcl in configs:
        ari_list, nmi_list = [], []
        
        for seed in SEEDS:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            spCLUE.fix_seed(seed)
            adata = adata_raw.copy()
            
            expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords, sigma=s)
            
            model = spCLUE.spCLUE(
                input_data=adata.obsm["X_pca"].copy(),
                graph_dict=graph_dict,
                n_clusters=n_clusters,
                expr_keep_prob=expr_keep_prob,
                gamma=g, kappa=k,
                use_spatial_drop=use_sp,
                use_intersection_cl=use_intcl,
            )
            
            _, features_fuse, _ = model.train()
            adata.obsm["emb"] = features_fuse
            
            try:
                spCLUE.clustering(adata, n_clusters, key="emb", refinement=True, cluster_methods="mclust")
            except Exception as e:
                print(f"  mclust refinement failed ({e}), trying without...")
                spCLUE.clustering(adata, n_clusters, key="emb", refinement=False, cluster_methods="mclust")
            
            cluster_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
            adata_eval = adata[adata.obs.Region.notna()].copy()
            ari = adjusted_rand_score(adata_eval.obs["Region"], adata_eval.obs[cluster_col])
            nmi = normalized_mutual_info_score(adata_eval.obs["Region"], adata_eval.obs[cluster_col])
            ari_list.append(ari)
            nmi_list.append(nmi)
            print(f"  {variant} seed={seed}: ARI={ari:.4f} NMI={nmi:.4f}")
        
        mean_ari = np.mean(ari_list)
        std_ari = np.std(ari_list)
        mean_nmi = np.mean(nmi_list)
        results.append({
            "Sample": sample_name,
            "Variant": variant,
            "Mean_ARI": round(mean_ari, 4),
            "Std_ARI": round(std_ari, 4),
            "Mean_NMI": round(mean_nmi, 4),
        })
        print(f"  >>> {variant} mean ARI: {mean_ari:.4f} ± {std_ari:.4f}")

# Summary
df = pd.DataFrame(results)
df.to_csv("SpatialGDC_DLPFC_Ablation_v2.csv", index=False)
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for v in ["Full", "w/o SP-DropEdge", "w/o SoftPenalty"]:
    sub = df[df["Variant"] == v]
    print(f"  {v}: mean ARI = {sub['Mean_ARI'].mean():.4f}")

print("\nDone. Results saved to SpatialGDC_DLPFC_Ablation_v2.csv")
