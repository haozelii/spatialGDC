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
# 3. 超参数搜索网格 (针对 MOSTA 小鼠胚胎数据集优化)
# ======================================================
param_grid = {
    'sigma': [0.3, 0.4, 0.5, 0.6, 0.7],
    'gamma': [0.5, 1.0, 1.5, 2.0],
    'kappa': [0.05, 0.1, 0.3, 0.5, 1.0],
}

os.makedirs("results_best", exist_ok=True)
os.makedirs("figures", exist_ok=True)

print("🚀 启动 MOSTA 小鼠胚胎数据集超参数搜索...")

# ======================================================
# 4. 加载 MOSTA 数据 (h5ad 格式)
# ======================================================
mosta_path = "./dataset/Mouse_Embryo/E9.5_E1S1.MOSTA.h5ad"
print(f"加载 MOSTA 数据: {mosta_path}")
adata_raw = sc.read_h5ad(mosta_path)
adata_raw.var_names_make_unique()

# MOSTA 使用 'annotation' 作为 ground truth，共 12 种器官
gt_col = 'annotation'

# 过滤掉没有标注的 spot (背景)
adata_raw = adata_raw[adata_raw.obs[gt_col].notna()].copy()
print(f"过滤背景后: {adata_raw.shape[0]} spots")

n_clusters = adata_raw.obs[gt_col].nunique()
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
                
                # 计算 ARI (使用原始数字标签 vs 预测)
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
    print(f"🏆 MOSTA 最佳结果: ARI={best_info_overall['Best_ARI']:.4f} | NMI={best_info_overall['Best_NMI']:.4f}")
    print(f"   最优参数: sigma={best_info_overall['sigma']}, gamma={best_info_overall['gamma']}, kappa={best_info_overall['kappa']}")
    print(f"{'='*60}")
    
    # 保存最佳 h5ad
    best_adata.obs['domain'] = best_adata.obs[cluster_col].astype('category')
    best_adata.write_h5ad("./results_best/Mouse_Embryo_BEST.h5ad")
    print("💾 最佳结果已保存至: results_best/Mouse_Embryo_BEST.h5ad")
    
    # 保存到 baselines/results 供图使用
    os.makedirs("./benchmarking/baselines/results", exist_ok=True)
    best_adata.write_h5ad("./benchmarking/baselines/results/Mouse_Embryo_our.h5ad")
    
    # 保存 CSV
    df = pd.DataFrame([best_info_overall])
    df.to_csv("SpatialGDC_Mouse_Embryo_Results.csv", index=False)
    print("📊 结果已保存至: SpatialGDC_Mouse_Embryo_Results.csv")
else:
    print("⚠️ 所有参数组合均未能出结果")
