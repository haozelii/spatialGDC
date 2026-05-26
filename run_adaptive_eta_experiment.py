"""DLPFC + BRCA: Adaptive η vs Flat η comparison
Compares per-edge adaptive penalty (formula 2) against uniform scalar penalty.
Same hyperparams as baseline Full model. 5 seeds each.
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

# ── Load best params from grid search ──
df_best = pd.read_csv("SpatialGDC_DLPFC_GridSearch_Summary.csv")
best_params = {}
for _, row in df_best.iterrows():
    best_params[str(int(row["Sample"]))] = {
        "sigma": row["sigma"], "gamma": row["gamma"], "kappa": row["kappa"]
    }

DLPFC_SAMPLES = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676"
]
SEEDS = [0, 42, 123, 999, 2024]

# ── Experiment configs ──
CONFIGS = [
    ("Flat η",     True, 2.0, False),   # adaptive_eta=False → uniform scalar
    ("Adaptive η", True, 2.0, True),    # adaptive_eta=True  → per-edge formula 2
]

# ============================================================
# DLPFC
# ============================================================
results = []

for sample_name in DLPFC_SAMPLES:
    print(f"\n{'='*60}\n  DLPFC Slice: {sample_name}\n{'='*60}")
    params = best_params[sample_name]

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
    spatial_coords = (spatial_coords - spatial_coords.min(axis=0)) / (
        spatial_coords.max(axis=0) - spatial_coords.min(axis=0))

    s, g, k = params["sigma"], params["gamma"], params["kappa"]

    for variant, use_intcl, fn_penalty, use_adaptive in CONFIGS:
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
                fn_penalty=fn_penalty,
                use_intersection_cl=use_intcl,
                adaptive_eta=use_adaptive,
                use_spatial_drop=True,
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
            "Sample": sample_name, "Variant": variant,
            "Mean_ARI": round(mean_ari, 4), "Std_ARI": round(std_ari, 4),
            "Mean_NMI": round(mean_nmi, 4),
        })
        print(f"  >>> {variant} mean ARI: {mean_ari:.4f} ± {std_ari:.4f}")

# Save intermediate results after DLPFC
df_dlpfc = pd.DataFrame(results)
df_dlpfc.to_csv("SpatialGDC_Adaptive_vs_Flat_Eta_DLPFC.csv", index=False)
print("DLPFC results saved to SpatialGDC_Adaptive_vs_Flat_Eta_DLPFC.csv")

# ============================================================
# BRCA
# ============================================================
print(f"\n{'='*60}\n  BRCA\n{'='*60}")

BRCA_PATH = "./dataset/BRCA1/V1_Human_Breast_Cancer_Block_A_Section_1"
adata_brca_raw = sc.read_visium(BRCA_PATH)
adata_brca_raw.var_names_make_unique()

brca_params = {"sigma": 0.6, "gamma": 5.0, "kappa": 0.05, "beta": 2.0, "fn_penalty": 2.0}

# Preprocessing
sc.pp.filter_genes(adata_brca_raw, min_counts=1)
sc.pp.filter_cells(adata_brca_raw, min_counts=1)
sc.pp.normalize_total(adata_brca_raw, target_sum=1e4)
sc.pp.log1p(adata_brca_raw)
sc.pp.scale(adata_brca_raw)

from sklearn.decomposition import PCA
pca_brca = PCA(n_components=200, random_state=0)
adata_brca_raw.obsm["X_pca"] = pca_brca.fit_transform(adata_brca_raw.X.toarray() if hasattr(adata_brca_raw.X, 'toarray') else adata_brca_raw.X)

g_spatial_brca = spCLUE.prepare_graph(adata_brca_raw, "spatial")
g_expr_brca = spCLUE.prepare_graph(adata_brca_raw, "expr")
graph_dict_brca = {"spatial": g_spatial_brca, "expr": g_expr_brca}

spatial_coords_brca = adata_brca_raw.obsm["spatial"].copy()
spatial_coords_brca = (spatial_coords_brca - spatial_coords_brca.min(axis=0)) / (
    spatial_coords_brca.max(axis=0) - spatial_coords_brca.min(axis=0))

expr_keep_prob_brca = spCLUE.compute_spatial_keep_prob(g_expr_brca, spatial_coords_brca, sigma=brca_params["sigma"])

# Load ground truth
meta_brca = pd.read_csv("./dataset/BRCA1/metadata.tsv", sep='\t', index_col=0)
adata_brca_raw.obs['Region'] = meta_brca.loc[adata_brca_raw.obs_names, 'fine_annot_type']
n_clusters_brca = adata_brca_raw.obs['Region'].nunique()

s_b, g_b, k_b = brca_params["sigma"], brca_params["gamma"], brca_params["kappa"]

for variant, use_intcl, fn_penalty, use_adaptive in CONFIGS:
    ari_list, nmi_list = [], []

    for seed in SEEDS:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        spCLUE.fix_seed(seed)
        adata = adata_brca_raw.copy()

        model = spCLUE.spCLUE(
            input_data=adata.obsm["X_pca"].copy(),
            graph_dict=graph_dict_brca,
            n_clusters=n_clusters_brca,
            expr_keep_prob=expr_keep_prob_brca,
            gamma=g_b, kappa=k_b, beta=brca_params["beta"],
            fn_penalty=fn_penalty,
            use_intersection_cl=use_intcl,
            adaptive_eta=use_adaptive,
            use_spatial_drop=True,
        )

        _, features_fuse, _ = model.train()
        adata.obsm["emb"] = features_fuse

        try:
            spCLUE.clustering(adata, n_clusters_brca, key="emb", refinement=True, cluster_methods="mclust")
        except Exception as e:
            print(f"  mclust refinement failed ({e}), trying without...")
            spCLUE.clustering(adata, n_clusters_brca, key="emb", refinement=False, cluster_methods="mclust")

        cluster_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
        adata_eval = adata[adata.obs.Region.notna()].copy()
        ari = adjusted_rand_score(adata_eval.obs["Region"], adata_eval.obs[cluster_col])
        nmi = normalized_mutual_info_score(adata_eval.obs["Region"], adata_eval.obs[cluster_col])
        ari_list.append(ari)
        nmi_list.append(nmi)
        print(f"  BRCA {variant} seed={seed}: ARI={ari:.4f} NMI={nmi:.4f}")

    mean_ari = np.mean(ari_list)
    std_ari = np.std(ari_list)
    mean_nmi = np.mean(nmi_list)
    results.append({
        "Sample": "BRCA", "Variant": variant,
        "Mean_ARI": round(mean_ari, 4), "Std_ARI": round(std_ari, 4),
        "Mean_NMI": round(mean_nmi, 4),
    })
    print(f"  >>> BRCA {variant} mean ARI: {mean_ari:.4f} ± {std_ari:.4f}")

# ── Save results ──
df = pd.DataFrame(results)
df.to_csv("SpatialGDC_Adaptive_vs_Flat_Eta.csv", index=False)
print(f"\n{'='*60}")
print("SUMMARY")
print(f"{'='*60}")
for v in ["Flat η", "Adaptive η"]:
    sub = df[df["Variant"] == v]
    for ds in DLPFC_SAMPLES + ["BRCA"]:
        sub_ds = sub[sub["Sample"] == ds]
        if len(sub_ds) > 0:
            print(f"  {ds:8s} {v:12s}: ARI={sub_ds['Mean_ARI'].values[0]:.4f}")
    sub_all = sub[sub["Sample"] != "BRCA"]
    print(f"  {'DLPFC':8s} {v:12s}: mean ARI = {sub_all['Mean_ARI'].mean():.4f}")
    brca_row = sub[sub["Sample"] == "BRCA"]
    if len(brca_row) > 0:
        print(f"  {'BRCA':8s} {v:12s}: ARI = {brca_row['Mean_ARI'].values[0]:.4f}")

print("\nDone. Results saved to SpatialGDC_Adaptive_vs_Flat_Eta.csv")
