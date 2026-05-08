import os
import sys
import gc
import warnings
import itertools
import torch
import pandas as pd
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# =========================================================================
# 0. 环境与路径配置 (直接从你的 Jupyter Cell 1 搬过来的，非常关键！)
# =========================================================================
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"
spCLUE_ROOT_PATH = "/home/bio/lhz/spatialGDC"
sys.path.append(spCLUE_ROOT_PATH)

import spCLUE
spCLUE.fix_seed(0)

# =========================================================================
# 1. 定义实验基本配置
# =========================================================================
dlpfc_samples = [
    "151671", "151672", "151673"
]
# [
#     "151507", "151508", "151509", "151510", 
#     "151669", "151670", "151671", "151672", 
#     "151673", "151674", "151675", "151676"
# ]
search_space = {
    'alpha': [0.05, 0.1, 0.2],      # 簇级对比权重 DCDLoss
    'beta': [0.005, 0.01, 0.02],    # 实例级平滑权重 RNGPALoss
    'warmup': [50, 100]             # 预热轮数
}

keys, values = zip(*search_space.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
final_results = []

print(f"🚀 开始批量搜索，共 {len(dlpfc_samples)} 个切片，每个切片 {len(param_combinations)} 组参数...")

# =========================================================================
# 2. 外层循环：遍历 12 个切片
# =========================================================================
for sample_name in dlpfc_samples:
    print(f"\n{'='*60}")
    print(f"正在处理切片: {sample_name}")
    print(f"{'='*60}")
    
    input_dir = f"../dataset/DLPFC/{sample_name}/"
    adata = spCLUE.preprocess_data(input_dir=input_dir)
    
    # 过滤掉没有手动注释(Region)的细胞
    adata = adata[adata.obs.Region.notna()].copy()
    n_clusters = 5 if sample_name in ["151669", "151670", "151671", "151672"] else 7
    
    # PCA 与图构建
    adata.obsm["X_pca"] = PCA(n_components=200, random_state=0).fit_transform(adata.X)
    g_spatial = spCLUE.prepare_graph(adata, "spatial")
    g_expr = spCLUE.prepare_graph(adata, "expr")
    graph_dict = {"spatial": g_spatial, "expr": g_expr}
    
    best_ari = 0
    best_nmi = 0
    best_params_for_slice = None
    
    for i, params in enumerate(param_combinations):
        print(f"  [{sample_name} | 实验 {i+1}/{len(param_combinations)}] 参数: {params}")
        
        test_model = spCLUE.spCLUE(adata.obsm["X_pca"], graph_dict, n_clusters)
        
        _, adata.obsm["spCLUE"] = test_model.train(
            warmup_epochs=params['warmup'], 
            alpha_dcd=params['alpha'], 
            beta_rngpa=params['beta']
        )
        
        spCLUE.clustering(
            adata,
            n_clusters,
            key="spCLUE",
            refinement=True,
            cluster_methods="mclust"
        )
        
        ari = adjusted_rand_score(adata.obs["Region"], adata.obs["mclust_refined"])
        nmi = normalized_mutual_info_score(adata.obs["Region"], adata.obs["mclust_refined"])
        
        print(f"  >>> 结果: ARI = {ari:.4f}, NMI = {nmi:.4f}")
        
        if ari > best_ari:
            best_ari = ari
            best_nmi = nmi
            best_params_for_slice = params
            
        # 清理显存防爆炸
        del test_model
        torch.cuda.empty_cache()
        gc.collect()
        
    print(f"\n✅ [切片 {sample_name} 完成] 最高 ARI: {best_ari:.4f}, 最高 NMI: {best_nmi:.4f}")
    
    final_results.append({
        'Sample': sample_name,
        'Best_ARI': round(best_ari, 4),
        'Best_NMI': round(best_nmi, 4),
        'Alpha': best_params_for_slice['alpha'],
        'Beta': best_params_for_slice['beta'],
        'Warmup': best_params_for_slice['warmup']
    })

# =========================================================================
# 3. 汇总与保存结果
# =========================================================================
df_results = pd.DataFrame(final_results)
mean_ari = df_results['Best_ARI'].mean()
mean_nmi = df_results['Best_NMI'].mean()

print("\n" + "="*50)
print(f"🎉 全部 12 个切片跑完！")
print(f"📊 平均 ARI: {mean_ari:.4f}")
print(f"📊 平均 NMI: {mean_nmi:.4f}")
print("="*50)

csv_filename = "DLPFC_12_slices_grid_search_results_去掉自环纯净图.csv"
df_results.to_csv(csv_filename, index=False)
print(f"\n📁 结果已保存至当前目录下的: {csv_filename}")