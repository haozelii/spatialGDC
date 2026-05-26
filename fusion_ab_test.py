#!/usr/bin/env python3
"""
Fusion AB Test: Attention vs ConcatProj vs PerDimGate
On 12-slice DLPFC dataset — using spCLUE data pipeline.
"""
import os
import sys
import gc
import numpy as np
import torch
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"
sys.path.insert(0, "/home/bio/lhz/spatialGDC")
import spCLUE

# ============ CONFIG ============
DATA_ROOT = "/home/bio/lhz/spatialGDC/dataset/DLPFC"
SLICES = [
    ("151507", 7), ("151508", 7), ("151509", 7), ("151510", 7),
    ("151669", 5), ("151670", 5), ("151671", 5), ("151672", 5),
    ("151673", 7), ("151674", 7), ("151675", 7), ("151676", 7),
]
FUSION_TYPES = ["attention", "concat_proj", "perdim_gate"]
SEED = 42
EPOCHS = 500

# ============ RUN ============
results = {}
for fusion_type in FUSION_TYPES:
    print(f"\n{'='*60}")
    print(f"  FUSION: {fusion_type}")
    print(f"{'='*60}")
    
    slice_results = {}
    for slice_id, n_clusters in SLICES:
        data_path = os.path.join(DATA_ROOT, slice_id)
        print(f"\n--- {slice_id} [{fusion_type}] ---")
        try:
            spCLUE.fix_seed(SEED)
            adata = spCLUE.load_and_preprocess_st(data_path=data_path)
            
            if 'X_pca' not in adata.obsm.keys():
                sc.tl.pca(adata, svd_solver='arpack', n_comps=200)
            
            g_spatial = spCLUE.prepare_graph(adata, "spatial")
            g_expr = spCLUE.prepare_graph(adata, "expr")
            graph_dict = {"spatial": g_spatial, "expr": g_expr}
            
            spatial_coords = adata.obsm["spatial"].copy()
            spatial_coords = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0) + 1e-8)
            expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords, sigma=0.5)
            
            spCLUE.fix_seed(SEED)
            model = spCLUE.spCLUE(
                input_data=adata.obsm["X_pca"],
                graph_dict=graph_dict,
                n_clusters=n_clusters,
                expr_keep_prob=expr_keep_prob,
                fusion_type=fusion_type,
                random_seed=SEED,
            )
            _, emb, _ = model.train()
            adata.obsm["SpatialGDC_emb"] = emb if not torch.is_tensor(emb) else emb.detach().cpu().numpy()
            
            # Clustering
            try:
                pred = spCLUE.clustering(adata, n_clusters, key="SpatialGDC_emb", refinement=True, cluster_methods="mclust")
            except:
                spCLUE.clustering(adata, n_clusters, key="SpatialGDC_emb", refinement=False, cluster_methods="mclust")
            
            cluster_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
            adata.obs['domain'] = adata.obs[cluster_col].astype('category')
            
            adata_eval = adata[adata.obs.Region.notna()].copy()
            ari = adjusted_rand_score(adata_eval.obs["Region"], adata_eval.obs["domain"])
            nmi = normalized_mutual_info_score(adata_eval.obs["Region"], adata_eval.obs["domain"])
            
            slice_results[slice_id] = {"ARI": ari, "NMI": nmi}
            print(f"  {slice_id} [{fusion_type}]: ARI={ari:.4f} | NMI={nmi:.4f}")
            
        except Exception as e:
            print(f"  {slice_id} [{fusion_type}]: ERROR - {e}")
            import traceback; traceback.print_exc()
            slice_results[slice_id] = {"ARI": 0.0, "NMI": 0.0}
        
        finally:
            if 'model' in dir(): del model
            if 'adata' in dir(): del adata
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    results[fusion_type] = slice_results

# ============ COMPARISON TABLE ============
print(f"\n{'='*80}")
print(f"COMPARISON: {' vs '.join(FUSION_TYPES)}")
print(f"{'='*80}")

header = f"{'Slice':<12}"
for ft in FUSION_TYPES:
    header += f" {ft:>12}"
print(header)
print("-" * (12 + 13 * len(FUSION_TYPES)))

arism = {}
for slice_id, _ in SLICES:
    row = f"{slice_id:<12}"
    for ft in FUSION_TYPES:
        if slice_id in results[ft]:
            ari = results[ft][slice_id]["ARI"]
            row += f" {ari:>12.4f}"
            arism.setdefault(slice_id, {})[ft] = ari
        else:
            row += f" {'N/A':>12}"
    print(row)

print("-" * (12 + 13 * len(FUSION_TYPES)))
row = f"{'MEAN':<12}"
for ft in FUSION_TYPES:
    mean_ari = np.mean([results[ft][s]["ARI"] for s in results[ft] if s in results[ft]])
    row += f" {mean_ari:>12.4f}"
print(row)

print()
row_nmi = f"{'Mean NMI':<12}"
for ft in FUSION_TYPES:
    nmi_vals = [results[ft][s]["NMI"] for s in results[ft]]
    row_nmi += f" {np.mean(nmi_vals):>12.4f}"
print(row_nmi)

# Per-slice winner
print(f"\nPer-slice winner:")
for slice_id, _ in SLICES:
    if slice_id in arism:
        best = max(arism[slice_id], key=arism[slice_id].get)
        best_val = arism[slice_id][best]
        attn_val = arism[slice_id].get("attention", 0)
        diff = best_val - attn_val
        print(f"  {slice_id}: {best} ({best_val:.4f}, vs attn {diff:+.4f})")
