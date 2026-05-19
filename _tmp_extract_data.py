#!/usr/bin/env python3
"""Extract MOB SC scores and BRCA marker gene data"""
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

# === MOB SC Scores ===
print("=== MOB Silhouette Scores ===")
our = sc.read_h5ad('results/Mouse_OB_our.h5ad')
print('obs cols:', list(our.obs.columns))
dom = our.obs.get('mclust_refined', our.obs.get('domain', None))
if dom is not None:
    labels = dom.astype(str).values
    emb = our.obsm.get('spCLUE_emb', our.obsm.get('X_pca', None))
    if emb is not None:
        sc_emb = silhouette_score(emb, labels)
        print(f'SpatialGDC MOB SC(embed)={sc_emb:.4f}')
    sc_spatial = silhouette_score(our.obsm['spatial'], labels)
    print(f'SpatialGDC MOB SC(spatial)={sc_spatial:.4f}')

for method in ['SEDR', 'SpaGCN']:
    try:
        bl = sc.read_h5ad(f'benchmarking/baselines/results/Mouse_OB_{method}.h5ad')
        dom_bl = bl.obs.get('domain', bl.obs.get('mclust_refined', None))
        if dom_bl is not None:
            labels_bl = dom_bl.astype(str).values
            emb_bl = bl.obsm.get('SEDR_emb', bl.obsm.get('X_pca', None))
            if emb_bl is not None:
                sc_bl = silhouette_score(emb_bl, labels_bl)
                print(f'{method} MOB SC(embed)={sc_bl:.4f}')
        del bl
    except Exception as e:
        print(f'{method}: {e}')

# === BRCA Marker Genes ===
print("\n=== BRCA Marker Gene Expression ===")
brca = sc.read_h5ad('benchmarking/baselines/results/BRCA_BEST.h5ad')
print('BRCA obs cols:', list(brca.obs.columns))

# Map to major domains
def map_major(x):
    x = str(x)
    if 'IDC' in x: return 'IDC'
    elif 'Healthy' in x: return 'Healthy'
    elif 'Tumor_edge' in x: return 'Tumor_edge'
    elif 'DCIS' in x or 'LCIS' in x: return 'DCIS/LCIS'
    else: return 'Other'

if 'fine_annot_type' in brca.obs.columns:
    brca.obs['Major_Domain'] = brca.obs['fine_annot_type'].apply(map_major)
    print('Major domains:', brca.obs['Major_Domain'].value_counts().to_dict())
    
    # Check for marker genes
    markers = ['IGFBP7', 'TIMP1', 'COL1A1', 'COL1A2', 'B2M', 'APOE', 'HLA-B']
    if 'SpatialGDC_enhanced' in brca.layers:
        print('Found SpatialGDC_enhanced layer')
        X = brca.layers['SpatialGDC_enhanced']
    else:
        print('No SpatialGDC_enhanced layer, using X')
        X = brca.X
    
    for gene in markers:
        if gene in brca.var_names:
            idx = list(brca.var_names).index(gene)
            if hasattr(X, 'toarray'):
                expr = X.toarray()[:, idx]
            else:
                expr = X[:, idx]
            for domain in ['IDC', 'Healthy', 'Tumor_edge', 'DCIS/LCIS']:
                mask = brca.obs['Major_Domain'] == domain
                if mask.sum() > 0:
                    mean_val = expr[mask].mean()
                    print(f'{gene} in {domain}: mean={mean_val:.4f}')
        else:
            print(f'{gene}: NOT found')

# === SpatialGDC BRCA ARI/NMI from best h5ad ===
print("\n=== BRCA ARI/NMI from h5ad ===")
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
if 'fine_annot_type' in brca.obs.columns and 'mclust_refined' in brca.obs.columns:
    valid = brca.obs['fine_annot_type'].notna() & brca.obs['mclust_refined'].notna()
    ari = adjusted_rand_score(brca.obs.loc[valid, 'fine_annot_type'], brca.obs.loc[valid, 'mclust_refined'])
    nmi = normalized_mutual_info_score(brca.obs.loc[valid, 'fine_annot_type'], brca.obs.loc[valid, 'mclust_refined'])
    print(f'BRCA_BEST ARI={ari:.4f} NMI={nmi:.4f}')
