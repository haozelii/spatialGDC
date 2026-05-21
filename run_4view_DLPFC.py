"""
4-View R7 (kappa=0.05/pair) — DLPFC 12 Slices — PROVEN CONFIG
===============================================================
V1/V2: spatial + NoiseLayer + DropEdge(0.4)
V3: expr + NoiseLayer + Spatial-Prior DropEdge (Innovation 1)
V4: expr + NoiseLayer + DropEdge(0.4)
CL: V1<->V3 + V2<->V4, ICL soft penalty (Innovation 2)
4-way Attention Fusion
kappa=0.05/pair (total=0.1, matches 2-view)
2-step reconstruction with intermediate ReLU
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm
import gc
import torch

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"
warnings.filterwarnings("ignore")

spCLUE_ROOT_PATH = "/home/bio/lhz/spatialGDC"
if spCLUE_ROOT_PATH not in sys.path:
    sys.path.append(spCLUE_ROOT_PATH)
import spCLUE

samples_list = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676"
]

results = []
os.makedirs("results", exist_ok=True)

ROUND = "R7_K05"

print(f"4-View {ROUND}: R7 with kappa=0.05/pair (PROVEN CONFIG)")
print(f"   V1/V2=spatial+uniform, V3=expr+sp_prior(Innovation1), V4=expr+uniform")
print(f"   CL: V1<->V3 + V2<->V4, ICL(Innovation2), 4-way Attention")
print(f"   Reconstruction: relu(z@W2.T)@W1.T")

for sample_name in samples_list:
    print(f"\n{'='*40}")
    print(f"Processing: {sample_name}")
    print(f"{'='*40}")
    
    try:
        spCLUE.fix_seed(0)
        
        data_path = f"./dataset/DLPFC/{sample_name}/"
        adata = spCLUE.load_and_preprocess_st(data_path=data_path)
        
        n_clusters = 5 if sample_name in ["151669", "151670", "151671", "151672"] else 7
        print(f"Spots: {adata.shape[0]}, n_clusters: {n_clusters}")

        if 'X_pca' not in adata.obsm.keys() or 'PCs' not in adata.varm.keys():
            print("Running PCA...")
            if 'log1p' not in adata.uns_keys():
                try:
                    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                except Exception:
                    pass
            sc.tl.pca(adata, svd_solver='arpack', n_comps=200)
        
        pc_matrix = adata.varm['PCs']

        g_spatia = spCLUE.prepare_graph(adata, "spatial")
        g_expr = spCLUE.prepare_graph(adata, "expr")
        graph_dict = {"spatial": g_spatia, "expr": g_expr}

        spatial_coords = adata.obsm["spatial"].copy()
        spatial_coords = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))
        expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords, sigma=0.5)

        spCLUE_model = spCLUE.spCLUE(
            input_data=adata.obsm["X_pca"].copy(),
            graph_dict=graph_dict,
            n_clusters=n_clusters,
            expr_keep_prob=expr_keep_prob,
            kappa=0.1,  # halved internally to 0.05 per pair
        )
        
        _, adata.obsm["SpatialGDC_emb"], best_x_rec_pca = spCLUE_model.train()

        if torch.is_tensor(best_x_rec_pca):
            best_x_rec_pca = best_x_rec_pca.detach().cpu().numpy()
            
        x_rec_gene = np.dot(best_x_rec_pca, pc_matrix.T)
        adata.layers["SpatialGDC_enhanced"] = x_rec_gene

        try:
            pred = spCLUE.clustering(adata, n_clusters, key="SpatialGDC_emb", refinement=True, cluster_methods="mclust")
        except Exception as e:
            print(f"mclust refinement failed ({e}), trying without refinement...")
            spCLUE.clustering(adata, n_clusters, key="SpatialGDC_emb", refinement=False, cluster_methods="mclust")

        cluster_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
        adata.obs['domain'] = adata.obs[cluster_col].astype('category')

        adata_eval = adata[adata.obs.Region.notna()].copy()
        ARI = adjusted_rand_score(adata_eval.obs["Region"], adata_eval.obs["domain"])
        NMI = normalized_mutual_info_score(adata_eval.obs["Region"], adata_eval.obs["domain"])
        
        print(f"{sample_name}: ARI = {ARI:.4f} | NMI = {NMI:.4f}")
        results.append({"Sample": sample_name, "ARI": ARI, "NMI": NMI, "Spots": adata.shape[0], "Clusters": n_clusters})

    except Exception as e:
        print(f"{sample_name} error: {e}")
        import traceback
        traceback.print_exc()
        results.append({"Sample": sample_name, "ARI": "Error", "NMI": "Error", "Spots": "N/A", "Clusters": "N/A"})
        
    finally:
        if 'spCLUE_model' in locals(): del spCLUE_model
        if 'adata' in locals(): del adata
        if 'adata_eval' in locals(): del adata_eval
        if 'g_spatia' in locals(): del g_spatia
        if 'g_expr' in locals(): del g_expr
        if 'expr_keep_prob' in locals(): del expr_keep_prob
        if 'spatial_coords' in locals(): del spatial_coords
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

df_results = pd.DataFrame(results)
valid_results = df_results[df_results['ARI'] != 'Error']
if not valid_results.empty:
    avg_ari = valid_results['ARI'].mean()
    avg_nmi = valid_results['NMI'].mean()
    print(f"\n{ROUND} Mean ARI = {avg_ari:.4f} | Mean NMI = {avg_nmi:.4f}")

csv_filename = f"4view_{ROUND}_SpatialGDC_DLPFC_Results.csv"
df_results.to_csv(csv_filename, index=False)
print(f"\nResults saved to: {csv_filename}")
print(df_results.to_string(index=False))
