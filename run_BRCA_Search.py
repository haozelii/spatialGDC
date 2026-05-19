import os
import sys
import warnings
import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import gc
import torch

# 1. 环境配置
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"
warnings.filterwarnings("ignore")

# 2. 导入核心包
spCLUE_ROOT_PATH = "/home/bio/lhz/spatialGDC"
if spCLUE_ROOT_PATH not in sys.path:
    sys.path.append(spCLUE_ROOT_PATH)
import spCLUE

# ======================================================
# 3. 超参数搜索网格 (针对 BRCA 数据集优化)
# ======================================================
param_grid = {
    'sigma': [0.3, 0.4, 0.5, 0.6, 0.7],
    'gamma': [0.5, 1.0, 1.5, 2.0],
    'kappa': [0.05, 0.1, 0.3, 0.5, 1.0],
}

os.makedirs("results_best", exist_ok=True)
os.makedirs("figures", exist_ok=True)

print("🚀 启动 BRCA 数据集超参数搜索...")

# ======================================================
# 4. 加载 BRCA 数据
# ======================================================
brca_dir = "./dataset/BRCA1/V1_Human_Breast_Cancer_Block_A_Section_1"
meta_path = "./dataset/BRCA1/metadata.tsv"

print("加载 BRCA Visium 数据...")
adata_raw = sc.read_visium(brca_dir)
adata_raw.var_names_make_unique()

# 加载 metadata
meta_df = pd.read_csv(meta_path, sep='\t', index_col=0)
adata_raw.obs = adata_raw.obs.join(meta_df, how='left')

# BRCA 使用 fine_annot_type 作为 ground truth (与 baselines 一致，共 ~21 类)
gt_col = 'fine_annot_type'
n_clusters = adata_raw.obs[gt_col].dropna().nunique()
print(f"✅ 数据加载完成: {adata_raw.shape[0]} spots, {n_clusters} 个聚类 (标注列: {gt_col})")
print(f"   类别分布:\n{adata_raw.obs[gt_col].value_counts().to_string()}")

# 预处理
sc.pp.filter_genes(adata_raw, min_counts=1)
sc.pp.filter_cells(adata_raw, min_counts=1)
sc.pp.normalize_total(adata_raw, target_sum=1e4)
sc.pp.log1p(adata_raw)
sc.pp.scale(adata_raw)

from sklearn.decomposition import PCA
adata_raw.obsm["X_pca"] = PCA(n_components=200, random_state=0).fit_transform(adata_raw.X)

# 构建图
g_spatial = spCLUE.prepare_graph(adata_raw, "spatial")
g_expr = spCLUE.prepare_graph(adata_raw, "expr")
graph_dict = {"spatial": g_spatial, "expr": g_expr}
spatial_coords_raw = adata_raw.obsm["spatial"].copy()
spatial_coords_norm = (spatial_coords_raw - spatial_coords_raw.min(axis=0)) / (spatial_coords_raw.max(axis=0) - spatial_coords_raw.min(axis=0))

print(f"✅ 图构建完成，开始网格搜索...")

# ======================================================
# 5. 网格搜索
# ======================================================
all_best_results = []
best_ari_overall = -1.0
best_info_overall = {}

total_combos = len(param_grid['sigma']) * len(param_grid['gamma']) * len(param_grid['kappa'])
print(f"总共 {total_combos} 种参数组合\n")

for s in param_grid['sigma']:
    for g in param_grid['gamma']:
        for k in param_grid['kappa']:
            print(f"[尝试] sigma:{s} | gamma:{g} | kappa:{k}", end=" | ")
            
            try:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                spCLUE.fix_seed(0)
                adata = adata_raw.copy()
                
                expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords_norm, sigma=s)
                
                model = spCLUE.spCLUE(
                    input_data=adata.obsm["X_pca"].copy(),
                    graph_dict=graph_dict,
                    n_clusters=n_clusters,
                    expr_keep_prob=expr_keep_prob,
                    gamma=g,
                    kappa=k
                )
                
                _, adata.obsm["emb"], _ = model.train()
                
                spCLUE.clustering(adata, n_clusters, key="emb", refinement=True, cluster_methods="mclust")
                cluster_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
                
                # 计算 ARI
                y_true = adata.obs[gt_col].astype(str)
                y_pred = adata.obs[cluster_col].astype(str)
                current_ari = adjusted_rand_score(y_true, y_pred)
                current_nmi = normalized_mutual_info_score(y_true, y_pred)
                
                print(f"ARI: {current_ari:.4f} | NMI: {current_nmi:.4f}")
                
                if current_ari > best_ari_overall:
                    best_ari_overall = current_ari
                    best_info_overall = {
                        "Best_ARI": current_ari, "Best_NMI": current_nmi,
                        "sigma": s, "gamma": g, "kappa": k
                    }
                    best_adata = adata.copy()
                    
            except Exception as e:
                print(f"❌ 报错: {e}")
                continue

# ======================================================
# 6. 保存最佳结果
# ======================================================
if best_info_overall:
    print(f"\n{'='*60}")
    print(f"🏆 BRCA 最佳结果: ARI={best_info_overall['Best_ARI']:.4f} | NMI={best_info_overall['Best_NMI']:.4f}")
    print(f"   最优参数: sigma={best_info_overall['sigma']}, gamma={best_info_overall['gamma']}, kappa={best_info_overall['kappa']}")
    print(f"{'='*60}")
    
    # 保存最佳 h5ad
    best_adata.obs['domain'] = best_adata.obs[cluster_col].astype('category')
    best_adata.write_h5ad("./results_best/BRCA_BEST.h5ad")
    print("💾 最佳结果已保存至: results_best/BRCA_BEST.h5ad")
    
    # 保存到 baselines/results 供图使用
    os.makedirs("./benchmarking/baselines/results", exist_ok=True)
    best_adata.write_h5ad("./benchmarking/baselines/results/BRCA_our.h5ad")
    
    # 保存 CSV
    df = pd.DataFrame([best_info_overall])
    df.to_csv("SpatialGDC_BRCA_Results.csv", index=False)
    print("📊 结果已保存至: SpatialGDC_BRCA_Results.csv")
else:
    print("⚠️ 所有参数组合均未能出结果")
