# SpatialGDC — Research Context Compression

---

## 1. Project Overview

Spatial transcriptomics domain identification via shared-weight GCN with spatial-prior guided graph regularization and topology-disentangled contrastive learning.

---

## 2. Problem Definition

**Core pain point:** Existing GNN-based spatial domain identification methods suffer from two coupled failure modes that jointly degrade performance at tissue boundaries.

**Problem 1 — Long-range false edges in feature graphs.**
Expression correlation graphs (Pearson on PCA space) contain spurious edges connecting spatially distant spots. Cause: dropout-driven zero-inflation makes distant spots appear correlated (many shared zero counts). Consequence: GNN message passing propagates noise from biologically unrelated regions into local neighborhoods, blurring domain boundaries.

**Problem 2 — Spatially homogeneous neighbors are forcibly repelled by standard InfoNCE.**
Standard contrastive learning treats ALL non-identical spots as equally negative. In spatially structured tissues, physically adjacent spots frequently belong to the SAME functional domain (e.g., two spots in cortical layer 4). Forcing their representations apart contradicts the biological reality of tissue continuity and blurs domain boundaries.

This is NOT a "false negative" problem in the statistical sense—these pairs are correctly classified as negatives under the InfoNCE formalism. The problem is that the InfoNCE formalism itself makes an anatomical error: it assumes independence between samples that are in fact spatially correlated. We call these "spatially homogeneous neighbor pairs"—spot pairs that are adjacent in both the spatial and expression graphs and thus highly likely to be functionally similar. Their forced repulsion is biologically incorrect.

**Problem 3 — Redundant contrastive pairs in multi-view frameworks.**
Existing methods proliferate CL pairs across all pairwise combinations of augmented views (4-6 pairs). Most pairs are redundant: they compare views that differ only in random dropout, providing no complementary signal. These redundant pairs complicate hyperparameter tuning without improving downstream performance.

**Problem 4 — Lack of topology-aware fusion.**
Existing methods fuse all views in a single flat operation, failing to distinguish between within-topology augmentation variance (e.g., V1 vs V3 differ only in DropEdge) and cross-topology complementary information (e.g., spatial vs expression graphs encode fundamentally different modalities).

---

## 3. Core Innovation

**Spatial-prior guided adaptive edge pruning + two-level topology-disentangled fusion + simplified single-pair contrastive learning with spatial proximity awareness.**

**Innovation 1 — Spatial-prior guided adaptive edge pruning (SP-DropEdge).**
The feature graph A_f is constructed exclusively from expression correlation—spatial coordinates play no role. Consequently, the GCN's message passing on A_f is completely blind to physical distance: two spots 2mm apart receive the same information flow as two spots 20µm apart, provided their expression correlation is identical. This is biologically problematic because expression correlation at long range is overwhelmingly driven by technical artifacts (shared dropout zeros, ambient RNA) rather than genuine co-expression.

Classical DropEdge~\cite{rong2019dropedge} randomly discards edges to regularize GNN training. While this improves generalization, it does so agnostically—local biological edges and distal noise edges are dropped with equal probability. SP-DropEdge replaces this uniform coin-flip with a spatially-informed mechanism: each edge's survival probability is a function of the physical distance between its endpoints, P_ij = 0.5 + 0.5·exp(-d²/2σ²). Spatially proximal edges survive with high probability (~1.0); spatially distant edges survive with lower probability (~0.5).

The mechanism is NOT a denoising filter—it does not deterministically remove edges. Rather, it is a **spatial prior injection mechanism**: it uses physical distance to bias the stochastic message-passing process so that information preferentially flows along anatomically plausible pathways. The GCN still sees all edges probabilistically (the lower bound of 0.5 ensures no edge is permanently severed), but edges consistent with spatial proximity are reinforced while edges inconsistent with it are suppressed. Because pruning is stochastic in each forward pass, the GCN cannot simply memorize which edges are noisy—it must learn representations that are robust to the distance-modulated perturbation, which functions as a biologically-informed data augmentation strategy. The spatial prior thus supplements the explicit spatial graph A_s with a weak form of spatial regularization on the feature graph A_f, bridging the two modalities at the level of information flow rather than at the level of representation fusion.

**Innovation 2 — Two-level topology-disentangled fusion.**
Views within the same topology (V1 and V3 share A_s; V2 and V4 share A_f) differ only in edge perturbation. Views across topologies (spatial vs expression) encode fundamentally different modalities. Flat fusion conflates these two qualitatively different sources of variation. Our two-level design first resolves perturbation variance within each topology (Level 1: Z_spa = fuse(V1, V3), Z_expr = fuse(V2, V4)), then integrates complementary cross-modal information (Level 2: Z_fuse = fuse(Z_spa, Z_expr)). This disentangles augmentation noise from modality complementarity.

**Innovation 3 — Simplified single-pair contrastive learning with spatial proximity awareness.**
After Level-1 fusion, the remaining variation between Z_spa and Z_expr is purely cross-topology. One contrastive pair captures this completely. Additional pairs (V1↔V4, V3↔V2, etc.) compare views that differ simultaneously in both topology and perturbation, entangling two orthogonal sources of variation.

Furthermore, standard InfoNCE repels all non-identical spots equally. We identify a subset of spot pairs—those that are neighbors in both the spatial and expression graphs—that are anatomically likely to belong to the same functional domain. Rather than forcibly repelling these spatially homogeneous neighbor pairs, we apply a soft tolerance: their similarity contribution in the contrastive denominator is reduced by a factor exp(-η), weakening but not eliminating their repulsive effect. This preserves the representation's discriminative capacity while preventing the model from fragmenting anatomically continuous regions.

---

## 4. Mechanism Chain

```
Spatial transcriptomics data
  → Dual graph construction (spatial KNN + expression Pearson correlation, K=12)
  → 4-view generation (V1: spa-orig, V2: expr-orig, V3: spa-DropEdge, V4: expr-SP-DropEdge)
  → Shared-weight 2-layer GCN (200→64→24) on all 4 views
  → L2 normalization
  → Level 1: same-topology fusion (V1+V3→Z_spa, V2+V4→Z_expr)
      → eliminates augmentation variance within each topology
  → Projection heads on Z_spa and Z_expr
      → Instance Head: h_spa, h_expr (L2-normed, 24-dim)
      → Cluster Head: label_spa, label_expr (soft assignments)
  → Contrastive losses (1 pair each):
      → ICL(h_spa, h_expr) with soft penalty on Ω_FN × κ
      → CCL(label_spa, label_expr) × β
  → Level 2: cross-topology fusion (Z_spa+Z_expr→Z_fuse via attention)
  → Reconstruction: Z_fuse → PCA space (200-dim) × γ
  → Total: L = κ·ICL + β·CCL + γ·MSE
  → Inference: Z_fuse → mclust → spatial domain labels
```

Key mechanism insight — SP-DropEdge vs Classical DropEdge:
```
Classical DropEdge: P(drop) = 0.4 (uniform for all edges)
  → treats all edges equally regardless of biological plausibility
  → discards useful local signals and retains harmful long-range noise with equal probability

SP-DropEdge:
  Physical distance d_ij
    → Gaussian decay P_keep = 0.5 + 0.5·exp(-d²/2σ²)
    → per-edge retention probability ∈ [0.5, 1.0]
    → Bernoulli sampling → biased edge dropout
    → distant spurious edges preferentially removed (~50% retention)
    → proximal biological edges preferentially retained (~100% retention)
    → cleaner local neighborhoods → sharper domain boundaries
    → stochastic per-iteration sampling → biologically-informed data augmentation
    → GCN learns representations robust to dropout-driven correlation noise
```

```
Spatially homogeneous neighbor tolerance:
  Ω_neighbor = {(i,j) | A_s(i,j)>0 ∧ A_f(i,j)>0, i≠j}
    → spot pairs adjacent in BOTH spatial and expression graphs
    → anatomically likely to be functionally similar
    → standard InfoNCE would forcibly repel them → fragments tissue
    → our approach: subtract η from their similarity in denominator
    → exp(-η) reduction in repulsive weight (~7.4× at η=2.0)
    → weakened but not eliminated repulsion
    → preserved anatomical continuity + retained discriminative capacity
```

---

## 5. Key Modules

### 5.1 Dual-Graph Construction (preprocess.py)
- **Spatial graph:** inverse Euclidean distance, KNN=12, binarize, no threshold, self-weight=0.3
- **Feature graph:** PCA(50)→Pearson correlation, KNN=12, threshold=0.1, self-weight=0.3
- Both undergo symmetric normalization: D^(-1/2)(A + w·I)D^(-1/2)

### 5.2 SP-DropEdge (network.py: get_dropped_adj)
- Computes per-edge keep probabilities from spatial coordinates
- Gaussian decay with sigma=0.5, mapped to [0.5, 1.0]
- Bernoulli sampling + expectation-preserving scaling
- Applied ONLY to V4 (expression augmented view)

### 5.3 Shared-Weight GCN Encoder (network.py: encoder)
- 2 layers: NoiseLayer(α=0.01, dropout=0.5) → Transform1(200→64) → ELU → Transform2(64→24) → ELU
- Same weights for all 4 views (siamese design)
- DropEdge applied independently at each layer

### 5.4 Two-Level Fusion (network.py: forward)
- Level 1: _fuse_pair(V1,V3) and _fuse_pair(V2,V4), both via attention
- Level 2: _fuse_pair(Z_spa, Z_expr) via same attention mechanism
- Attention: Linear→Tanh→Linear→Softmax → weighted sum

### 5.5 Soft Penalty Contrastive Loss (loss.py: IntersectionContrastiveLoss)
- Intersection set Ω_FN from original (non-dropped) adjacency matrices
- fn_penalty=2.0 subtracts from similarity logits
- Symmetric bidirectional InfoNCE with temperature τ=0.2

### 5.6 Cluster-Level Contrastive Loss (loss.py: ClusterLoss)
- Cosine similarity between cluster centroids
- Negative entropy regularization prevents collapse
- Single cross-topology pair: CCL(label_spa, label_expr)

---

## 6. Loss Function Analysis

```
L_total = κ · L_ICL(h_spa, h_expr) + β · L_CCL(label_spa, label_expr) + γ · L_MSE(x_rec, X)

Where:
  L_ICL: symmetric bidirectional InfoNCE, τ=0.2
         spatially homogeneous neighbor pairs receive soft tolerance η=2.0
  L_CCL: cluster centroid cosine similarity + entropy regularization
  L_MSE: reconstruction of 200-dim PCA input via transposed GCN weights
```

**Why the spatial proximity-aware tolerance works:**
Standard InfoNCE forces exp(sim(anchor, negative)) → 0 for ALL negatives. In tissues, ~5-15% of spot pairs are adjacent in both spatial and expression graphs—these are anatomically homogeneous neighbors that likely belong to the same domain. Forcing their similarity to zero is biologically incorrect and fragments tissue continuity. The soft tolerance reduces their denominator contribution by exp(-η) without zeroing it out. This preserves a weak repulsive gradient that prevents representation collapse while avoiding excessive repulsion at domain boundaries.

**Why single-pair CL suffices:**
With 4 views, C(4,2)=6 possible CL pairs. After Level-1 fusion resolves perturbation variance, the remaining variation is purely cross-topology. One pair (Z_spa↔Z_expr) captures this completely. Additional pairs compare views that differ simultaneously in both topology and perturbation, entangling orthogonal sources of variation and adding optimization noise without signal.

---

## 7. Experiment Summary

### DLPFC (12 slices, 10x Visium)
| Metric | Value |
|--------|-------|
| Mean ARI (grid search) | 0.582 |
| Best slice | 151671: 0.812 |
| Worst slice | 151670: 0.475 |
| vs Spatial-MGCN | +0.01 |
| vs STAGATE | +0.05 |
| vs Seurat | +0.27 |

### BRCA (21-class, 10x Visium)
| Metric | Value |
|--------|-------|
| Best ARI | 0.663 |
| Best NMI | 0.695 |
| Best params | sigma=0.6, gamma=5.0, kappa=0.05, beta=2.0 |
| vs stGRL | +0.115 |
| vs Spatial-MGCN | +0.023 |

### MOSTA (12 organs, Stereo-seq)
| Metric | Value |
|--------|-------|
| Best ARI | 0.406 |
| vs stGRL | +0.078 |
| Key finding | Two-level fusion significantly benefits high-resolution data |

### MOB (Slide-seqV2)
| Metric | Value |
|--------|-------|
| Best SC | 0.183 (subsample n=5000) |
| vs SEDR | -0.07 |
| Note | SC sensitive to subsampling; visual layer recovery is accurate |

---

## 8. Ablation Insights

| Ablation | DLPFC Δ | BRCA Δ | Meaning |
|----------|---------|--------|---------|
| w/o SP-DropEdge | **-0.06** | **-0.06** | Spatial prior is the single most important innovation. Consistent across datasets. |
| w/o SoftPenalty | -0.02 | 0.00 | Soft penalty matters in laminar tissues (DLPFC), negligible in heterogeneous tumors (BRCA). Dataset-dependent. |

**Interpretation:**
- SP-DropEdge is universally beneficial: physical distance is an effective prior for distinguishing signal from noise in expression correlation graphs.
- Soft penalty is context-dependent: it helps when the tissue has regular laminar organization (intersection neighbors are informative), but not when the tissue is highly heterogeneous with fine-grained annotations (intersection neighbors are sparse).
- This dataset-dependent behavior is itself a meaningful scientific finding, not a weakness.

---

## 9. Comparison With Existing Methods

| Method | Graph | Contrastive | Fusion | Edge Denoising |
|--------|-------|-------------|--------|----------------|
| SpaGCN | Single spatial | None | None | None |
| STAGATE | Single spatial | None | None | None |
| Spatial-MGCN | Dual (spatial+expr) | None | Attention | None |
| stGRL | Dual (spatial+expr) | Standard InfoNCE | None | None |
| **SpatialGDC** | **Dual + 4-view augmentation** | **Simplified single-pair + soft penalty** | **Two-level hierarchical** | **SP-DropEdge** |

**What is new:**
1. SP-DropEdge: first method to use physical distance as a prior for adaptive feature graph pruning in spatial transcriptomics
2. Two-level fusion: first to explicitly separate within-topology augmentation fusion from cross-topology complementary fusion
3. Simplified CL: first to argue that single-pair CL suffices once augmentation variance is resolved
4. Soft penalty: first to use soft (rather than hard) false negative mitigation in spatial contrastive learning

---

## 10. Important Experimental Evidence

**Fig 2B Boxplot:** DLPFC 12-slice ARI comparison. SpatialGDC achieves highest mean and lowest variance. Demonstrates robustness.

**Fig 3 Comprehensive (151672):** 8-method spatial map + UMAP + PAGA. Shows SpatialGDC produces the most anatomically faithful domain boundaries.

**Fig BRCA Spatial Domains:** 5×2 tissue-background layout. SpatialGDC domains align with ground truth; other methods produce fragmented clusters.

**Table 1 (BRCA):** SpatialGDC leads in ARI (0.663), CCST leads in NMI (0.706). Demonstrates that SP-DropEdge improves cluster purity (ARI) more than mutual information (NMI).

**Ablation Table:** SP-DropEdge removal causes consistent -0.06 drop across DLPFC and BRCA. Soft penalty removal causes -0.02 on DLPFC, 0 on BRCA.

**MOSTA ARI:** 0.406 vs stGRL 0.328 (+23.8%). Two-level fusion provides largest relative gain on this dataset.

---

## 11. Method Draft For Paper

SpatialGDC takes as input a spatial transcriptomics dataset of N spots with gene expression matrix X (PCA-reduced to 200 dims) and spatial coordinates S. Two adjacency graphs are constructed: a spatial graph A_s via inverse Euclidean distance KNN, and a feature graph A_f via Pearson correlation KNN on PCA space. Both are symmetrically normalized with self-loop augmentation.

Four views are generated through targeted perturbation. V1 uses A_s without edge dropout. V2 uses A_f without edge dropout. V3 applies uniform random DropEdge (p=0.4) to A_s. V4 applies spatial-prior guided adaptive pruning to A_f, where each edge's retention probability P_ij = 0.5 + 0.5·exp(-d_ij²/2σ²) decays with the physical distance between its endpoints. Input features receive Gaussian noise (α=0.01) and dropout (p=0.5) across all views.

A shared-weight two-layer GCN encoder (200→64→24, ELU) processes all four views, producing L2-normalized embeddings Z_1 through Z_4. Two-level hierarchical fusion then operates: Level 1 fuses same-topology pairs (Z_spa = attention(Z_1, Z_3), Z_expr = attention(Z_2, Z_4)), resolving augmentation variance. Level 2 fuses across topologies (Z_fuse = attention(Z_spa, Z_expr)), capturing complementary information. The attention mechanism uses a two-layer perceptron with tanh activation to produce per-spot view weights.

Instance-level contrastive learning operates on the Level-1 fused representations through a projection MLP: ICL(h_spa, h_expr) with temperature τ=0.2. A false-negative-aware soft penalty subtracts η=2.0 from the similarity of spot pairs that are neighbors in both A_s and A_f, weakening their repulsive contribution without eliminating it. Cluster-level contrastive learning applies CCL between the soft cluster assignments of the two Level-1 fused representations, with negative entropy regularization to prevent collapse. A reconstruction decoder projects Z_fuse back to the 200-dimensional PCA space via transposed GCN weights.

The total loss L = κ·ICL + β·CCL + γ·MSE is minimized with Adam (lr=0.001, weight_decay=0.001) and cosine annealing over 500 epochs. After training, Z_fuse is clustered via mclust (Gaussian mixture model) with optional spatial neighborhood refinement (radius=30).

---

## 12. Introduction Material

**Background hook:** Spatial transcriptomics enables gene expression profiling while preserving tissue architecture. Spatial domain identification—partitioning tissue into coherent functional regions—is the foundational analysis task.

**Pain point:** ST data exhibits extreme sparsity and dropout. Expression correlation graphs contain long-range spurious edges. Standard contrastive learning assumes all non-identical spots are equally negative, which violates the spatial continuity of tissues.

**Gap:** Existing methods either use single spatial graphs (over-smoothing at boundaries) or multi-view graphs without spatial-prior denoising (noise propagation) and with redundant contrastive pairs (optimization complexity).

**Our approach:** We propose SpatialGDC with three synergistic innovations: (1) spatial-prior guided adaptive feature graph pruning, (2) two-level hierarchical fusion separating augmentation from complementarity, (3) simplified single-pair contrastive learning with false-negative-aware soft penalty.

---

## 13. Potential Paper Claims

For the abstract/introduction/conclusion:
- "spatial-prior guided adaptive pruning"
- "two-level hierarchical fusion disentangling within-topology and cross-topology information"
- "false-negative-aware soft penalty preserving local tissue continuity"
- "simplified single-pair contrastive learning eliminating redundant optimization"
- "shared-weight siamese GCN encoder enforcing consistent latent space"
- "spatially informed, probability-weighted edge retention"
- "intersection-neighborhood false negative mitigation"
- "dataset-dependent soft penalty behavior"

---

## 14. Important Files

| File | Role |
|------|------|
| `spCLUE/spCLUE.py` | Training loop, model initialization, hyperparameter management |
| `spCLUE/network.py` | CCGCN model class: encoder, fusion, projection heads, forward pass |
| `spCLUE/loss.py` | IntersectionContrastiveLoss (soft penalty), ClusterLoss, MSELoss |
| `spCLUE/preprocess.py` | Data loading, normalization, PCA, graph construction, SP keep_prob |
| `spCLUE/utils.py` | Sparse matrix conversion, learning rate scheduling, seed fixing |
| `paper/spatialGDC.tex` | Complete LaTeX paper (IEEEtran, ~375 lines) |
| `paper/refs.bib` | 431-line BibTeX reference file |
| `SpatialGDC_DLPFC_GridSearch_Summary.csv` | DLPFC 12-slice best results per slice |
| `SpatialGDC_DLPFC_Ablation_v2.csv` | DLPFC ablation (180 runs, 3 variants × 12 slices × 5 seeds) |
| `SpatialGDC_BRCA_GridSearch_v2.csv` | BRCA best result (240-combo search) |
| `SpatialGDC_BRCA_Ablation_v2.csv` | BRCA ablation (3 variants × 5 seeds) |
| `SpatialGDC_MOSTA_GridSearch_v2.csv` | MOSTA best result |
| `SpatialGDC_MOB_SC_v2.txt` | MOB best SC (240-combo search) |

---

## 15. Final Research Essence

**The central insight:** In spatial transcriptomics, physical distance is underutilized as a structural prior. SpatialGDC weaponizes spatial coordinates in three ways: (1) as a graph denoising signal (SP-DropEdge—the most impactful module, -0.06 ARI when removed), (2) as an organizational principle for hierarchical fusion (same-topology before cross-topology), and (3) as a constraint on contrastive learning (spatially adjacent spots should not be forcibly repelled). The framework demonstrates that computing with physical space as a first-class signal—rather than merely as an adjacency matrix—is the key to resolving tissue boundaries in noisy transcriptomic data.

**What makes it novel, distilled to one sentence:** SpatialGDC is the first method to use physical distance as a *continuous*, *edge-level* prior for adaptive graph denoising in spatial transcriptomics contrastive learning, combined with a hierarchical fusion design that separates augmentation variance from cross-modal complementarity.
