import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA


def calcGAEParams(graph, n_samples):
    """Calculate pos_weight and norm_val for a bipartite graph."""
    non_zero_cnt = graph.sum()
    norm_val = (n_samples * n_samples) / (2 * (n_samples * n_samples - non_zero_cnt))
    pos_weight = (n_samples * n_samples - non_zero_cnt) / non_zero_cnt
    return norm_val, pos_weight


def calcGraphWeight(coor, eps=1e-6):
    dist = cdist(coor, coor, "euclidean")
    dist = dist / (np.max(dist) + eps)
    return dist


def correlation_graph(A, B):
    """Calculate correlation matrix between A and B."""
    am = A - np.mean(A, axis=0, keepdims=True)
    bm = B - np.mean(B, axis=0, keepdims=True)
    return am.T @ bm / (np.sqrt(np.sum(am**2, axis=0, keepdims=True)).T * np.sqrt(np.sum(bm**2, axis=0, keepdims=True)))


def prepare_graph(adata, key="spatial", n_neighbors=12, n_comps=50, eps=1e-8,
                  svd_solver="randomized", self_weight=0.3):
    """Build spatial or expression KNN graph from AnnData."""
    n_spots = adata.shape[0]
    assert key in ["spatial", "expr"], "key must be 'spatial' or 'expr'"

    if key == "spatial":
        coords = adata.obsm[key]
        weights = 1. / (cdist(coords, coords, "euclidean") + eps)
    else:
        expr = PCA(n_components=n_comps, random_state=0,
                   svd_solver=svd_solver).fit_transform(adata.X)
        weights = correlation_graph(expr.T, expr.T)

    threshold = np.sort(weights)[:, -n_neighbors - 1:-n_neighbors]
    weights[weights < threshold] = 0
    weights = (weights + weights.T) / 2
    weights = weights * (1 - np.eye(n_spots))

    adj_filter = 0. if key == "spatial" else 0.1
    adj_bip = np.where(weights > adj_filter, 1, 0)

    return sp.coo_matrix(symm_norm(adj_bip, weightDiag=self_weight))


def symm_norm(adj, weightDiag=0.3, eps=1e-8):
    """Symmetric normalization: D^{-1/2} (A + w*I) D^{-1/2}."""
    n_spot = adj.shape[0]
    adj_self = (1 - weightDiag) * adj + np.eye(n_spot) * weightDiag
    degrees = 1. / np.sqrt((np.sum(adj_self, axis=1) + eps))
    adj_self *= degrees
    adj_self *= degrees[:, None]
    return adj_self.astype(np.float32)


def compute_spatial_keep_prob(adj_coo, spatial_coords, sigma=0.5):
    """Compute edge retention probability based on physical distance.

    Uses soft decay: P_keep = 0.5 + 0.5 * exp(-d^2 / (2*sigma^2)).
    Distant edges retain at least 50% probability.
    """
    src = adj_coo.row
    dst = adj_coo.col
    src_coords = spatial_coords[src]
    dst_coords = spatial_coords[dst]

    dist_sq = np.sum((src_coords - dst_coords) ** 2, axis=1)
    if dist_sq.max() > 0:
        dist_sq = (dist_sq - dist_sq.min()) / (dist_sq.max() - dist_sq.min() + 1e-8)

    base_prob = np.exp(-dist_sq / (2 * sigma ** 2))
    keep_prob = 0.5 + 0.5 * base_prob

    return keep_prob.astype(np.float32)
