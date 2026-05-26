"""Quick comparison: Flat vs Degree vs Floor vs Soft on DLPFC 151507"""
import sys
sys.path.insert(0, '/home/bio/lhz/spatialGDC')
import torch, numpy as np, gc, scanpy as sc
from sklearn.metrics import adjusted_rand_score
import spCLUE

CONFIGS = [
    ("Flat η",       True,  False,  None,         None),
    ("Degree (η_ij)", True,  True,  "degree",     None),
    ("Floor (η_ij)",  True,  True,  "floor",      0.5),
    ("Soft (η_ij)",   True,  True,  "soft",       None),
]

print("=" * 60)
print("DLPFC 151507: Adaptive η Variant Comparison")
print("=" * 60)

# Load data once
spCLUE.fix_seed(0)
adata_raw = spCLUE.load_and_preprocess_st(data_path="./dataset/DLPFC/151507/")
adata_raw.obs_names = adata_raw.obs_names.str.replace('-1', '', regex=False).str.strip()

if 'X_pca' not in adata_raw.obsm:
    sc.tl.pca(adata_raw, svd_solver='arpack', n_comps=200)

g_spatial = spCLUE.prepare_graph(adata_raw, "spatial")
g_expr = spCLUE.prepare_graph(adata_raw, "expr")
graph_dict = {"spatial": g_spatial, "expr": g_expr}

spatial_coords = adata_raw.obsm["spatial"].copy()
spatial_coords = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))
expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords, sigma=0.4)

for label, use_intcl, use_adaptive, mode, floor in CONFIGS:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    spCLUE.fix_seed(0)
    adata = adata_raw.copy()

    kwargs = dict(
        input_data=adata.obsm["X_pca"].copy(),
        graph_dict=graph_dict,
        n_clusters=7,
        expr_keep_prob=expr_keep_prob,
        gamma=1.0, kappa=0.1, fn_penalty=2.0,
        use_intersection_cl=use_intcl,
        adaptive_eta=use_adaptive,
        use_spatial_drop=True,
    )
    if mode is not None:
        kwargs["adaptive_mode"] = mode
    if floor is not None:
        kwargs["eta_floor"] = floor

    model = spCLUE.spCLUE(**kwargs)
    _, features_fuse, _ = model.train()

    adata.obsm["emb"] = features_fuse
    try:
        spCLUE.clustering(adata, 7, key="emb", refinement=True, cluster_methods="mclust")
    except:
        spCLUE.clustering(adata, 7, key="emb", refinement=False, cluster_methods="mclust")

    cluster_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
    adata_eval = adata[adata.obs.Region.notna()].copy()
    ari = adjusted_rand_score(adata_eval.obs["Region"], adata_eval.obs[cluster_col])

    # Print η stats
    adj_s = g_spatial.to_dense()
    adj_e = g_expr.to_dense()
    if mode == "soft":
        soft_w = torch.min(adj_s, adj_e)
        soft_w.fill_diagonal_(0)
        deg = soft_w.sum(dim=1)
    else:
        fn_mask = (adj_s > 0) & (adj_e > 0)
        fn_mask.fill_diagonal_(False)
        deg = fn_mask.float().sum(dim=1)
    nonzero = deg > 0
    avg = deg[nonzero].mean().item() if nonzero.any() else 0
    eta_range = f"[{2.0*deg.min().item()/avg:.1f}, {2.0*deg.max().item()/avg:.1f}]" if avg > 0 else "N/A"

    print(f"  {label:16s}  ARI={ari:.4f}  deg_H mean={avg:.1f}  eta range={eta_range}")

print("=" * 60)
