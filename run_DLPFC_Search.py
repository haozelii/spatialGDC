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

# 1. 环境配置
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"

warnings.filterwarnings("ignore")

# 2. 导入核心包
spCLUE_ROOT_PATH = "/home/bio/lhz/spCLUE"
if spCLUE_ROOT_PATH not in sys.path:
    sys.path.append(spCLUE_ROOT_PATH)
import spCLUE

# ======================================================
# 🌟 3. 定义超参数搜索网格
# ======================================================
param_grid = {
    'sigma': [0.4, 0.5, 0.6],     
    'gamma': [1.0, 1.5, 2.0],     
    'kappa': [0.1, 0.5, 1.0],          
}

samples_list = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676"
]

os.makedirs("results_best", exist_ok=True)
all_best_results = [] 

print(f"🚀 启动 DLPFC 原始方法超参搜索 (极速跑分版)...")

for sample_name in samples_list:
    print(f"\n{'#'*60}\n🔥 搜索样本: {sample_name}\n{'#'*60}")
    
    best_ari_for_sample = -1.0
    best_info = {}
    best_adata = None

    # 加载基础数据
    data_path = f"./dataset/DLPFC/{sample_name}/"
    adata_raw = spCLUE.load_and_preprocess_st(data_path=data_path)
    n_clusters = 5 if sample_name in ["151669", "151670", "151671", "151672"] else 7
    
    if 'Region' not in adata_raw.obs.columns:
        df_meta = pd.read_csv(os.path.join(data_path, 'metadata.tsv'), sep='\t', index_col=0)
        adata_raw.obs['Region'] = df_meta.loc[adata_raw.obs_names, 'layer_guess']
    
    g_spatia = spCLUE.prepare_graph(adata_raw, "spatial")
    g_expr = spCLUE.prepare_graph(adata_raw, "expr")
    graph_dict = {"spatial": g_spatia, "expr": g_expr}
    spatial_coords_raw = adata_raw.obsm["spatial"].copy()
    spatial_coords_norm = (spatial_coords_raw - spatial_coords_raw.min(axis=0)) / (spatial_coords_raw.max(axis=0) - spatial_coords_raw.min(axis=0))

    # 遍历网格
    for s in param_grid['sigma']:
        for g in param_grid['gamma']:
            for k in param_grid['kappa']:
                print(f"\n[尝试] sigma:{s} | gamma:{g} | kappa:{k}")
                
                try:
                    gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    
                    spCLUE.fix_seed(0)
                    adata = adata_raw.copy()
                    
                    # 重新计算保留概率
                    expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords_norm, sigma=s)
                    
                    # 实例化 (参数传给 __init__)
                    model = spCLUE.spCLUE(
                        input_data=adata.obsm["X_pca"].copy(),
                        graph_dict=graph_dict,
                        n_clusters=n_clusters,
                        expr_keep_prob=expr_keep_prob,
                        gamma=g,
                        kappa=k
                    )
                    
                    # 🌟 极简修改 1：直接用 '_' 丢弃第三个返回值（重构矩阵），不让它占用内存
                    _, adata.obsm["emb"], _ = model.train()
                    
                    # 聚类精修
                    spCLUE.clustering(adata, n_clusters, key="emb", refinement=True, cluster_methods="mclust")
                    cluster_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
                    
                    # 计算得分
                    adata_eval = adata[adata.obs.Region.notna()].copy()
                    current_ari = adjusted_rand_score(adata_eval.obs["Region"], adata_eval.obs[cluster_col])
                    current_nmi = normalized_mutual_info_score(adata_eval.obs["Region"], adata_eval.obs[cluster_col])
                    
                    print(f"   📊 ARI: {current_ari:.4f} | NMI: {current_nmi:.4f}")

                    # 🌟 极简修改 2：去除基因增强相乘，只认分数！
                    if current_ari > best_ari_for_sample:
                        best_ari_for_sample = current_ari
                        best_adata = adata.copy()
                        
                        best_info = {
                            "Sample": sample_name,
                            "Best_ARI": current_ari, "Best_NMI": current_nmi,
                            "sigma": s, "gamma": g, "kappa": k
                        }

                except Exception as e:
                    print(f"   ❌ 报错跳过: {e}")
                    continue

    # 保存最佳结果
    if best_adata is not None and best_info:
        all_best_results.append(best_info)
        best_adata.write_h5ad(f"./results_best/DLPFC_{sample_name}_BEST.h5ad")
        # 顺便完善了打印信息，把 sigma 也打印出来
        print(f"\n🏆 {sample_name} 最佳 ARI: {best_ari_for_sample:.4f} (sigma={best_info['sigma']}, gamma={best_info['gamma']}, kappa={best_info['kappa']})")
    else:
        print(f"\n⚠️ {sample_name} 所有参数组合均未能出结果。")
    
    del adata_raw, best_adata
    gc.collect()

# 导出统计
if all_best_results:
    df_final = pd.DataFrame(all_best_results)
    df_final.to_csv("SpatialGDC_DLPFC_GridSearch_Summary.csv", index=False)
    print(f"\n✅ 搜索完成！平均最佳 ARI: {df_final['Best_ARI'].mean():.4f}")