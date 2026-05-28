"""Step 3: Per-slice local grid search refinement for SpatialGDC DLPFC"""
import os, sys, gc, warnings
import pandas as pd, numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import torch

os.environ["OMP_NUM_THREADS"] = "8"
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/bio/lhz/spatialGDC")
import spCLUE

# Load current best params
df_best = pd.read_csv("SpatialGDC_DLPFC_GridSearch_Summary.csv")
best_params = {}
for _, row in df_best.iterrows():
    best_params[str(int(row["Sample"]))] = {
        "sigma": row["sigma"], "gamma": row["gamma"], "kappa": row["kappa"],
        "ari": row["Best_ARI"], "nmi": row["Best_NMI"]
    }

SAMPLES = ["151507","151508","151509","151510","151669","151670",
           "151671","151672","151673","151674","151675","151676"]

results = []

for sample_name in SAMPLES:
    print(f"\n{'='*50}\n  {sample_name} (current: ARI={best_params[sample_name]['ari']:.4f})")
    print(f"{'='*50}")

    data_path = f"./dataset/DLPFC/{sample_name}/"
    adata_raw = spCLUE.load_and_preprocess_st(data_path=data_path)
    n_clusters = 5 if sample_name in ["151669","151670","151671","151672"] else 7

    if 'Region' not in adata_raw.obs.columns:
        df_meta = pd.read_csv(os.path.join(data_path, 'metadata.tsv'), sep='\t', index_col=0)
        adata_raw.obs['Region'] = df_meta.loc[adata_raw.obs_names, 'layer_guess']
    if 'X_pca' not in adata_raw.obsm:
        sc.tl.pca(adata_raw, svd_solver='arpack', n_comps=200)

    g_spatial = spCLUE.prepare_graph(adata_raw, "spatial")
    g_expr = spCLUE.prepare_graph(adata_raw, "expr")
    graph_dict = {"spatial": g_spatial, "expr": g_expr}
    spatial_coords = adata_raw.obsm["spatial"].copy()
    spatial_coords = (spatial_coords - spatial_coords.min(axis=0)) / (
        spatial_coords.max(axis=0) - spatial_coords.min(axis=0))

    bp = best_params[sample_name]
    best_ari, best_sigma, best_gamma, best_kappa, best_nmi = (
        bp["ari"], bp["sigma"], bp["gamma"], bp["kappa"], bp["nmi"])

    # Phase 1: sigma × gamma sweep (kappa fixed at best)
    sigmas = sorted(set(max(0.3, min(0.8, bp["sigma"] + d)) for d in [-0.1, 0, 0.1]))
    gammas = sorted(set(max(0.5, min(3.0, bp["gamma"] + d)) for d in [-0.5, 0, 0.5]))
    print(f"  Phase 1: sigma={sigmas}, gamma={gammas}, kappa={bp['kappa']}")

    for s_val in sigmas:
        for g_val in gammas:
            if s_val == bp["sigma"] and g_val == bp["gamma"]:
                continue  # skip already-known best
            gc.collect(); torch.cuda.empty_cache()
            spCLUE.fix_seed(0)
            adata = adata_raw.copy()
            expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords, sigma=s_val)
            model = spCLUE.spCLUE(
                input_data=adata.obsm["X_pca"].copy(), graph_dict=graph_dict,
                n_clusters=n_clusters, expr_keep_prob=expr_keep_prob,
                gamma=g_val, kappa=bp["kappa"], use_spatial_drop=True,
                use_intersection_cl=True,
            )
            _, features_fuse, _ = model.train()
            adata.obsm["emb"] = features_fuse
            try:
                spCLUE.clustering(adata, n_clusters, key="emb", refinement=True, cluster_methods="mclust")
            except:
                spCLUE.clustering(adata, n_clusters, key="emb", refinement=False, cluster_methods="mclust")
            cc = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
            ae = adata[adata.obs.Region.notna()].copy()
            ari = adjusted_rand_score(ae.obs["Region"], ae.obs[cc])
            marker = ""
            if ari > best_ari:
                best_ari = ari; best_sigma = s_val; best_gamma = g_val
                best_nmi = normalized_mutual_info_score(ae.obs["Region"], ae.obs[cc])
                marker = " ★ NEW BEST"
            print(f"    σ={s_val} γ={g_val} κ={bp['kappa']}: ARI={ari:.4f}{marker}")
            del model, adata

    # Phase 2: kappa sweep (sigma/gamma fixed at phase 1 best)
    kappas = sorted(set(max(0.03, min(2.0, best_kappa * m)) for m in [0.5, 0.75, 1.5, 2.0]))
    kappas = [k for k in kappas if k != best_kappa]
    print(f"  Phase 2: sigma={best_sigma}, gamma={best_gamma}, kappa={kappas}")

    for k_val in kappas:
        gc.collect(); torch.cuda.empty_cache()
        spCLUE.fix_seed(0)
        adata = adata_raw.copy()
        expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords, sigma=best_sigma)
        model = spCLUE.spCLUE(
            input_data=adata.obsm["X_pca"].copy(), graph_dict=graph_dict,
            n_clusters=n_clusters, expr_keep_prob=expr_keep_prob,
            gamma=best_gamma, kappa=k_val, use_spatial_drop=True,
            use_intersection_cl=True,
        )
        _, features_fuse, _ = model.train()
        adata.obsm["emb"] = features_fuse
        try:
            spCLUE.clustering(adata, n_clusters, key="emb", refinement=True, cluster_methods="mclust")
        except:
            spCLUE.clustering(adata, n_clusters, key="emb", refinement=False, cluster_methods="mclust")
        cc = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
        ae = adata[adata.obs.Region.notna()].copy()
        ari = adjusted_rand_score(ae.obs["Region"], ae.obs[cc])
        marker = ""
        if ari > best_ari:
            best_ari = ari; best_kappa = k_val
            best_nmi = normalized_mutual_info_score(ae.obs["Region"], ae.obs[cc])
            marker = " ★ NEW BEST"
        print(f"    σ={best_sigma} γ={best_gamma} κ={k_val}: ARI={ari:.4f}{marker}")
        del model, adata

    delta = best_ari - bp["ari"]
    print(f"  >>> {sample_name} FINAL: ARI={best_ari:.4f} (Δ={delta:+.4f}) NMI={best_nmi:.4f} "
          f"σ={best_sigma} γ={best_gamma} κ={best_kappa}")
    results.append({
        "Sample": sample_name, "Best_ARI": round(best_ari,4), "Best_NMI": round(best_nmi,4),
        "sigma": best_sigma, "gamma": best_gamma, "kappa": best_kappa,
        "Prev_ARI": bp["ari"], "Delta": round(delta,4),
    })

df = pd.DataFrame(results)
df.to_csv("SpatialGDC_DLPFC_GridSearch_Refined.csv", index=False)
print(f"\n{'='*60}")
print(f"SUMMARY: mean ARI = {df['Best_ARI'].mean():.4f} (delta vs prev = {df['Delta'].mean():+.4f})")
print(f"Saved to SpatialGDC_DLPFC_GridSearch_Refined.csv")
