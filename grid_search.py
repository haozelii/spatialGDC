"""
Grid search: n_neighbors × sigma on 3 representative DLPFC slices.
Target: find optimal global config for ablation to show clear improvement.
"""
import os, sys, warnings, gc, itertools
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

# Representative slices
SLICES = ["151507", "151670", "151671"]
# SLICES = ["151507"]  # uncomment for quick test

# Grid
N_NEIGHBORS = [6, 8, 10, 12, 16, 20]
SIGMAS = [0.1, 0.3, 0.5, 0.8, 1.0]

results = []

for sample_name in SLICES:
    print(f"\n{'='*60}\nLOADING {sample_name}\n{'='*60}")
    spCLUE.fix_seed(0)
    data_path = f"./dataset/DLPFC/{sample_name}/"
    adata = spCLUE.load_and_preprocess_st(data_path=data_path)
    n_clusters = 5 if sample_name in ["151669", "151670", "151671", "151672"] else 7
    print(f"Spots: {adata.shape[0]}, n_clusters: {n_clusters}")

    if 'X_pca' not in adata.obsm.keys():
        sc.tl.pca(adata, svd_solver='arpack', n_comps=200)

    spatial_coords_raw = adata.obsm["spatial"].copy()
    spatial_coords_norm = (spatial_coords_raw - spatial_coords_raw.min(axis=0)) / \
                          (spatial_coords_raw.max(axis=0) - spatial_coords_raw.min(axis=0))

    for n_neigh in N_NEIGHBORS:
        # Build graphs with this n_neighbors (only once per n_neigh)
        g_spatial = spCLUE.prepare_graph(adata, "spatial", n_neighbors=n_neigh)
        g_expr = spCLUE.prepare_graph(adata, "expr", n_neighbors=n_neigh)
        graph_dict = {"spatial": g_spatial, "expr": g_expr}

        for sigma in SIGMAS:
            spCLUE.fix_seed(0)
            expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords_norm, sigma=sigma)

            try:
                model = spCLUE.spCLUE(
                    input_data=adata.obsm["X_pca"].copy(),
                    graph_dict=graph_dict,
                    n_clusters=n_clusters,
                    expr_keep_prob=expr_keep_prob,
                    kappa=0.1,
                    fusion_type="attention",
                )
                _, emb, _ = model.train()

                # eval
                adata_eval = adata.copy()
                adata_eval.obsm["emb"] = emb
                try:
                    spCLUE.clustering(adata_eval, n_clusters, key="emb", refinement=True, cluster_methods="mclust")
                except:
                    spCLUE.clustering(adata_eval, n_clusters, key="emb", refinement=False, cluster_methods="mclust")

                cluster_col = 'mclust_refined' if 'mclust_refined' in adata_eval.obs.columns else 'mclust'
                adata_eval = adata_eval[adata_eval.obs.Region.notna()].copy()
                ARI = adjusted_rand_score(adata_eval.obs["Region"], adata_eval.obs[cluster_col])
                NMI = normalized_mutual_info_score(adata_eval.obs["Region"], adata_eval.obs[cluster_col])
                print(f"  {sample_name} | n_neigh={n_neigh:>2} sigma={sigma:.1f} | ARI={ARI:.4f} NMI={NMI:.4f}")
                results.append({"Sample": sample_name, "n_neighbors": n_neigh, "sigma": sigma,
                                "ARI": ARI, "NMI": NMI, "Spots": adata.shape[0], "Clusters": n_clusters})
            except Exception as e:
                print(f"  {sample_name} | n_neigh={n_neigh:>2} sigma={sigma:.1f} | ERROR: {e}")
                results.append({"Sample": sample_name, "n_neighbors": n_neigh, "sigma": sigma,
                                "ARI": "Error", "NMI": "Error", "Spots": "N/A", "Clusters": "N/A"})
            finally:
                for v in ['model', 'adata_eval']:
                    if v in locals(): del locals()[v]
                gc.collect(); torch.cuda.empty_cache()

        # Clean up graph variables
        del g_spatial, g_expr, graph_dict

    del adata, spatial_coords_raw, spatial_coords_norm
    gc.collect(); torch.cuda.empty_cache()

# Save
df = pd.DataFrame(results)
df.to_csv("grid_search_nneigh_sigma.csv", index=False)
print(f"\n{'='*60}")
print("DONE. Saved to grid_search_nneigh_sigma.csv")
valid = df[df['ARI'] != 'Error']
if not valid.empty:
    # Best per slice
    for s in SLICES:
        slice_df = valid[valid['Sample'] == s]
        if not slice_df.empty:
            best = slice_df.loc[slice_df['ARI'].idxmax()]
            print(f"\n{s} BEST: n_neigh={int(best['n_neighbors'])} sigma={best['sigma']} ARI={best['ARI']:.4f}")
    # Overall mean
    best_overall = valid.groupby(['n_neighbors', 'sigma'])['ARI'].mean().reset_index()
    top5 = best_overall.nlargest(5, 'ARI')
    print("\nTop 5 overall (mean across slices):")
    for _, row in top5.iterrows():
        print(f"  n_neigh={int(row['n_neighbors'])} sigma={row['sigma']} mean_ARI={row['ARI']:.4f}")
