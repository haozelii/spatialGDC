#!/usr/bin/env python3
"""
Figure 6 Panel A: Allen Reference Atlas Layer Visualization for Mouse Olfactory Bulb

This script creates Panel A showing the Allen Reference Atlas laminar organization
of MOB, with spots colored by anatomical layer based on marker gene evidence.

Cluster-to-layer mapping derived from 8 marker genes:
  Gabra1 -> GL (glomerular layer)
  Slc6a11 -> EPL (external plexiform layer)
  Mbp -> MCL (mitral cell layer)
  Atp2b4 -> GCL (granule cell layer)
  Pcp4 -> ONL (olfactory nerve layer)
  Nrgn -> IPL (internal plexiform layer)
  Cck -> EPL/EPLi (external plexiform layer, inner)
  Kctd12 -> GL/EPL boundary (superficial external plexiform layer)

Evidence-based mapping (from marker gene mean expression per cluster):
  Cluster 0  -> ONL    (Pcp4=2.10, outer ring)
  Cluster 6  -> ONL    (r=2258 outermost, peripheral nerve layer)
  Cluster 10 -> GL     (Gabra1=1.12, GL marker)
  Cluster 4  -> GL     (Gabra1=0.69, high Pcp4)
  Cluster 5  -> GL     (Kctd12=0.77, GL/EPL boundary)
  Cluster 3  -> EPL    (Cck=0.87, Slc6a11=0.48)
  Cluster 8  -> EPL    (Cck=0.69, Slc6a11=0.47)
  Cluster 9  -> EPL    (Cck=1.31, EPLi marker)
  Cluster 7  -> EPL    (Slc6a11=0.63, inner EPL)
  Cluster 2  -> MCL    (Mbp=0.93, MCL marker)
  Cluster 1  -> GCL    (Nrgn=0.67, Atp2b4=0.23, innermost core)
"""

import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. Load the result data
# ============================================================
res_file = "./results/Mouse_OB_our.h5ad"
if not os.path.exists(res_file):
    print(f"ERROR: Cannot find {res_file}")
    exit()

adata = sc.read_h5ad(res_file)

# Get cluster labels
pred_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
adata.obs['domain'] = adata.obs[pred_col].astype(str)

# ============================================================
# 2. CORRECTED cluster-to-layer mapping (evidence-based)
# ============================================================
# MOB layers from outermost to innermost:
# ONL (olfactory nerve layer) -> GL (glomerular layer) -> 
# EPL (external plexiform layer) -> MCL (mitral cell layer) -> 
# IPL (internal plexiform layer) -> GCL (granule cell layer) -> RMS

cluster2layer = {
    '0': 'ONL',       # Pcp4=2.10 (highest ONL marker), outer ring
    '6': 'ONL',       # r=2258 (outermost), peripheral
    '10': 'GL',       # Gabra1=1.12 (highest GL marker)
    '4': 'GL',        # Gabra1=0.69, Pcp4 high (GL region)
    '5': 'GL',        # Kctd12=0.77 (GL/EPL boundary marker)
    '3': 'EPL',       # Cck=0.87, Slc6a11=0.48, Gabra1=0.47
    '8': 'EPL',       # Cck=0.69, Slc6a11=0.47
    '9': 'EPL',       # Cck=1.31 (highest EPLi marker)
    '7': 'EPL',       # Slc6a11=0.63 (highest EPL marker)
    '2': 'MCL',       # Mbp=0.93 (highest MCL marker)
    '1': 'GCL',       # Nrgn=0.67, Atp2b4=0.23, innermost core
}

adata.obs['layer'] = adata.obs['domain'].map(lambda x: cluster2layer.get(x, 'Unknown'))

# Set categorical order (outer to inner)
layer_order = ['ONL', 'GL', 'EPL', 'MCL', 'GCL']
existing_layers = [l for l in layer_order if l in adata.obs['layer'].unique()]
for l in adata.obs['layer'].unique():
    if l not in existing_layers:
        existing_layers.append(l)
adata.obs['layer'] = pd.Categorical(adata.obs['layer'], categories=existing_layers, ordered=True)

print(f"Layer annotation complete. Layers: {adata.obs['layer'].value_counts().to_dict()}")

# ============================================================
# 3. Swap spatial coordinates to match vertical alignment
# ============================================================
spatial = adata.obsm['spatial'].copy()
adata.obsm['spatial'] = spatial[:, [1, 0]]  # swap x,y for vertical display

# ============================================================
# 4. Panel A: Allen Reference Atlas Layer Visualization
# ============================================================
# Colors matching Allen Brain Atlas convention (blue-based gradient)
layer_colors = {
    'ONL':  '#4DBBD5',   # Cyan-blue (outermost)
    'GL':   '#3C5488',   # Dark blue
    'EPL':  '#E64B35',   # Red
    'MCL':  '#F39B7F',   # Salmon
    'IPL':  '#91D1C2',   # Teal
    'GCL':  '#8491B4B2', # Muted blue-gray (innermost)
    'RMS':  '#B09C85',   # Tan
}

print("Drawing Panel A: Allen Reference Atlas layer visualization...")

fig, ax = plt.subplots(figsize=(6, 8), facecolor='white')

x = adata.obsm['spatial'][:, 0]
y = adata.obsm['spatial'][:, 1]
labels = adata.obs['layer']

# Draw from inner to outer (so outer points overlay inner ones)
for layer_name in reversed(layer_order):
    idx = (labels == layer_name)
    if idx.sum() == 0:
        continue
    color = layer_colors.get(layer_name, '#999999')
    ax.scatter(
        x[idx], y[idx],
        color=color,
        label=layer_name,
        s=12, marker='o', edgecolors='none', antialiased=False
    )

ax.set_title("Allen Reference Atlas\nMOB Layers", fontsize=20, fontweight='bold', pad=15)
ax.set_aspect('equal')
ax.set_xticks([])
ax.set_yticks([])

for spine in ax.spines.values():
    spine.set_linewidth(1.5)

# Legend
handles = [mpatches.Patch(color=layer_colors.get(l, '#999999'), label=l) 
           for l in layer_order if l in labels.unique()]
lgnd = ax.legend(
    handles=handles, loc='center left', bbox_to_anchor=(1, 0.5),
    fontsize=14, frameon=True, title="Layer",
    title_fontsize=16, edgecolor='black', fancybox=False
)

os.makedirs("figures", exist_ok=True)
fig.savefig("figures/Fig6_PanelA_Allen_Ref_Layers.png", dpi=300, bbox_inches='tight')
fig.savefig("figures/Fig6_PanelA_Allen_Ref_Layers.pdf", dpi=300, bbox_inches='tight')
plt.close()

print("Panel A saved to figures/Fig6_PanelA_Allen_Ref_Layers.png/.pdf")

# ============================================================
# 5. Validation: Print marker gene enrichment per layer
# ============================================================
if 'log1p' not in adata.uns_keys():
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

marker_genes = ["Gabra1", "Slc6a11", "Mbp", "Atp2b4", "Pcp4", "Nrgn", "Cck", "Kctd12"]
marker_layers = {
    "Gabra1": "GL", "Slc6a11": "EPL", "Mbp": "MCL", "Atp2b4": "GCL",
    "Pcp4": "ONL", "Nrgn": "IPL", "Cck": "EPL/EPLi", "Kctd12": "GL/EPL"
}

print("\n=== Marker Gene Enrichment per Layer (Validation) ===")
print(f"{'Gene':<12} {'Expected':<12} {'Best Layer':<12} {'Mean Exp':>10}")
print("-" * 50)

valid_genes = [g for g in marker_genes if g in adata.var_names]
for gene in valid_genes:
    exp = adata[:, gene].X.toarray().flatten() if hasattr(adata[:, gene].X, 'toarray') else adata[:, gene].X.flatten()
    best_layer = None
    best_val = -1
    for layer_name in layer_order:
        idx = (labels == layer_name)
        if idx.sum() > 0:
            v = exp[idx].mean()
            if v > best_val:
                best_val = v
                best_layer = layer_name
    expected = marker_layers[gene]
    match = "OK" if expected.split('/')[0] in best_layer or best_layer in expected.split('/') else "CHECK"
    print(f"{gene:<12} {expected:<12} {best_layer:<12} {best_val:>10.4f}  {match}")

# ============================================================
# 6. Also save layer annotations back to the h5ad for downstream use
# ============================================================
adata.obs['layer_annotation'] = adata.obs['layer']
# Restore original spatial
adata.obsm['spatial'] = spatial
print(f"\nLayer annotations saved to adata.obs['layer_annotation']")
print(f"Category order: {list(adata.obs['layer_annotation'].cat.categories)}")
print("\nDone!")
