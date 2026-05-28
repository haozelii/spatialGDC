# SpatialGDC

**Spatial Graph Dual Contrastive Learning for Spatial Transcriptomics Domain Identification**

SpatialGDC identifies spatial domains in transcriptomics data by jointly modeling spatial proximity and gene expression similarity through dual-graph contrastive learning. The framework constructs two complementary views — a spatial nearest-neighbor graph and an expression correlation graph — and learns spot representations via hierarchical fusion and spatially-aware contrastive objectives.

## Architecture

```
PCA Input (N×200) → 4-View Shared GCN (200→64→24)
  V1(spa_orig)  V2(expr_orig)  V3(spa_aug)  V4(expr_aug+SP)
       ↓              ↓              ↓              ↓
  Level 1 Fusion (same-topo) → z_spa, z_expr
       ↓                             ↓
  Instance CL                   Cluster CL
       ↓                             ↓
  Level 2 Fusion (cross-topo) → z_fuse → Decoder → Reconstruction
```

Key components:
- **Spatial-Prior Guided DropEdge (SP-DropEdge)**: Feature graph edges are dropped with probability inversely proportional to physical distance, injecting spatial information into the GNN message passing
- **Two-Level Hierarchical Fusion**: Level 1 resolves augmentation variance within each topology; Level 2 fuses complementary cross-modal information
- **Spatially-Aware Soft Contrastive Loss**: Spot pairs adjacent in both spatial and expression graphs receive a soft penalty rather than being forcibly repelled

## Installation

```bash
conda create -n spatialgdc python=3.9
conda activate spatialgdc
pip install torch scanpy scikit-learn pandas tqdm rpy2
```

mclust (optional, for clustering refinement):
```r
install.packages("mclust")
```

## Quick Start

```python
import scanpy as sc
from spatialgdc import (
    SpatialGDC, load_and_preprocess_st, prepare_graph,
    compute_spatial_keep_prob, clustering, fix_seed
)

# Load and preprocess Visium data
fix_seed(0)
adata = load_and_preprocess_st(data_path="./dataset/DLPFC/151507/")
adata.obs_names = adata.obs_names.str.replace('-1', '', regex=False).str.strip()

# Build dual graphs
g_spatial = prepare_graph(adata, "spatial")
g_expr = prepare_graph(adata, "expr")
graph_dict = {"spatial": g_spatial, "expr": g_expr}

# Compute spatial prior
coords = adata.obsm["spatial"].copy()
coords = (coords - coords.min(axis=0)) / (coords.max(axis=0) - coords.min(axis=0) + 1e-8)
keep_prob = compute_spatial_keep_prob(g_expr, coords, sigma=0.5)

# Train model
model = SpatialGDC(
    input_data=adata.obsm["X_pca"].copy(),
    graph_dict=graph_dict,
    n_clusters=7,
    expr_keep_prob=keep_prob,
    gamma=1.0, kappa=0.1,
)
pred_labels, embeddings, x_rec = model.train()

# Cluster and evaluate
adata.obsm["emb"] = embeddings
clustering(adata, 7, key="emb", refinement=True, cluster_methods="mclust")
```

## Results

### DLPFC (12 dorsolateral prefrontal cortex slices)

| Method | ARI | NMI |
|--------|-----|-----|
| **SpatialGDC** | **0.590** | **0.669** |
| stGRL | 0.533 | 0.666 |
| STAGATE | 0.512 | 0.657 |
| Spatial-MGCN | 0.510 | 0.656 |
| CCST | 0.465 | 0.628 |
| SEDR | 0.414 | 0.546 |
| SpaGCN | 0.343 | 0.466 |
| Seurat | 0.337 | 0.430 |

### BRCA (21-class fine annotation)

| Method | ARI | NMI |
|--------|-----|-----|
| **SpatialGDC** | **0.663** | 0.695 |
| Spatial-MGCN | 0.640 | 0.683 |
| CCST | 0.589 | **0.706** |
| stGRL | 0.548 | 0.679 |
| SpaGCN | 0.538 | 0.652 |
| STAGATE | 0.494 | 0.642 |
| Seurat | 0.482 | 0.624 |
| SEDR | 0.434 | 0.666 |

### MOSTA (12 mouse embryo organs)

| Method | ARI |
|--------|-----|
| **SpatialGDC** | **0.406** |
| stGRL | 0.328 |

## Configuration

SpatialGDC exposes key hyperparameters through the constructor:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gamma` | 1.0 | Reconstruction loss weight |
| `kappa` | 0.1 | Instance contrastive loss weight |
| `beta` | 1.0 | Cluster contrastive loss weight |
| `fn_penalty` | 2.0 | Soft penalty coefficient for spatially-adjacent pairs |
| `use_spatial_drop` | True | Enable SP-DropEdge |
| `use_intersection_cl` | True | Enable spatially-aware soft contrastive loss |

## Citation

```bibtex
@article{spatialgdc2025,
  title={SpatialGDC: Spatial Graph Dual Contrastive Learning for Spatial Transcriptomics Domain Identification},
  author={Li, Haoze},
  year={2025}
}
```
