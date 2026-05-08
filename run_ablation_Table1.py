import os
import sys
import gc
import warnings
import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import torch
from sklearn.metrics import adjusted_rand_score
import argparse

spCLUE_ROOT_PATH = "/home/bio/lhz/spatialGDC"
if spCLUE_ROOT_PATH not in sys.path:
    sys.path.append(spCLUE_ROOT_PATH)
import spCLUE

warnings.filterwarnings("ignore")

# ==========================================
# 1. 实验核心配置
# ==========================================
DLPFC_DIR = "/home/bio/lhz/spatialGDC/dataset/DLPFC/"
BRCA_DIR = "/home/bio/lhz/spatialGDC/dataset/BRCA1/"

dlpfc_samples = ["151507", "151508", "151509", "151510", "151669", "151670", "151671", "151672", "151673", "151674", "151675", "151676"]
brca_samples = ["V1_Human_Breast_Cancer_Block_A_Section_1"]

# 定义对标论文的 4 个消融变体
ablation_modes = [
    "spCLUE",              # 满血版
    "w/o Dual-graph",      # 去掉双图融合 (用单图)
    "w/o Instance-CL",     # 去掉实例级对比
    "w/o Spatial-guide"    # 去掉空间引导丢边
]

# 存储结果的字典
results_dlpfc_raw = {mode: [] for mode in ablation_modes}
results_brca_raw = {mode: [] for mode in ablation_modes}

os.makedirs("results/Ablation", exist_ok=True)

# ==========================================
# 2. 核心执行逻辑函数
# ==========================================
def run_ablation_on_sample(dataset_type, sample_name, base_dir):
    print(f"\n{'='*60}\n🔬 正在处理 {dataset_type} 切片: {sample_name}\n{'='*60}")
    
    # --- A. 数据加载与对齐 ---
    data_path = os.path.join(base_dir, sample_name)
    adata = spCLUE.load_and_preprocess_st(data_path=data_path)
    
    if dataset_type == 'DLPFC':
        df_meta = pd.read_csv(os.path.join(data_path, 'metadata.tsv'), sep='\t', index_col=0)
        adata.obs['Region'] = df_meta.loc[adata.obs_names, 'layer_guess']
        n_clusters = 5 if sample_name in ["151669", "151670", "151671", "151672"] else 7
        
    elif dataset_type == 'BRCA':
        meta_path = os.path.join(base_dir, "metadata.tsv")
        meta_df = pd.read_csv(meta_path, sep='\t', index_col=0)
        adata.obs.index = adata.obs.index.astype(str).str.replace('-1', '', regex=False).str.strip()
        meta_df.index = meta_df.index.astype(str).str.replace('-1', '', regex=False).str.strip()
        
        annot_col = None
        for col in meta_df.columns:
            if 18 <= meta_df[col].nunique() <= 25: annot_col = col; break
        if annot_col is None:
            max_unique = 0
            for col in meta_df.columns:
                n_unique = meta_df[col].nunique()
                if 5 < n_unique <= 30 and n_unique > max_unique:
                    max_unique = n_unique
                    annot_col = col
                    
        adata.obs = adata.obs.merge(meta_df[[annot_col]], left_index=True, right_index=True, how='left')
        adata.obs['Region'] = adata.obs[annot_col]
        n_clusters = adata.obs['Region'].nunique()

    adata = adata[adata.obs['Region'].notna()].copy() 
    if sp.issparse(adata.X): adata.X = adata.X.toarray()
    
    # --- B. 构建基础图 ---
    g_spatia = spCLUE.prepare_graph(adata, "spatial")
    g_expr = spCLUE.prepare_graph(adata, "expr")
    
    spatial_coords = adata.obsm["spatial"].copy()
    spatial_coords = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))
    base_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords, sigma=0.5)
    
    res_dict = {}

    # --- C. 遍历运行 4 个变体 ---
    for mode in ablation_modes:
        print(f"  🧪 运行变体: 【{mode}】")
        spCLUE.fix_seed(0) 
        
        # 默认满血参数
        current_keep_prob = base_keep_prob
        use_instance_cl = True 
        graph_dict = {"spatial": g_spatia, "expr": g_expr}
        
        # 变体配置
        if mode == "w/o Dual-graph":
            graph_dict = {"spatial": g_spatia, "expr": g_spatia} # 强制使用双空间图退化为单图
            current_keep_prob = None # 单图不需要表达图保留概率
        elif mode == "w/o Instance-CL":
            use_instance_cl = False  
        elif mode == "w/o Spatial-guide":
            current_keep_prob = None # 退化为全局均匀随机丢边

        # 训练
        spCLUE_model = spCLUE.spCLUE(
            input_data=adata.obsm["X_pca"], 
            graph_dict=graph_dict, 
            n_clusters=n_clusters,
            expr_keep_prob=current_keep_prob,
            use_instance_cl=use_instance_cl
        )
        _, adata.obsm["spCLUE_emb"], _ = spCLUE_model.train()
        
        # 聚类精修
        spCLUE.clustering(adata, n_clusters, key="spCLUE_emb", refinement=True, cluster_methods="mclust")
        cluster_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
        
        ari = adjusted_rand_score(adata.obs["Region"], adata.obs[cluster_col])
        print(f"    🏆 ARI = {ari:.4f}")
        res_dict[mode] = ari
        
        del spCLUE_model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    return res_dict

# ==========================================
# 3. 开启全量化屠榜
# ==========================================
print(f"🚀 开始进行跨数据集消融实验...")

# 跑 DLPFC
for sample in dlpfc_samples:
    res = run_ablation_on_sample('DLPFC', sample, DLPFC_DIR)
    for mode in ablation_modes:
        results_dlpfc_raw[mode].append(res[mode])

# 跑 BRCA
for sample in brca_samples:
    res = run_ablation_on_sample('BRCA', sample, BRCA_DIR)
    for mode in ablation_modes:
        results_brca_raw[mode].append(res[mode])

# ==========================================
# 4. 生成完美对标论文的 Table 1
# ==========================================
final_table = []
for mode in ablation_modes:
    mean_dlpfc = np.mean(results_dlpfc_raw[mode])
    mean_brca = np.mean(results_brca_raw[mode])
    final_table.append({
        'Method': mode,
        'ARI(DLPFC)': f"{mean_dlpfc:.2f}",  # 保留两位小数，和论文一模一样
        'ARI(BC)': f"{mean_brca:.2f}"
    })

df_table1 = pd.DataFrame(final_table)

print("\n" + "=" * 50)
print("Table 1: Performance comparison of spCLUE and its ablated variants")
print("=" * 50)
print(df_table1.to_string(index=False))
print("=" * 50)

df_table1.to_csv("results/Ablation/Table1_Ablation_spCLUE.csv", index=False)
print("✅ 表格已保存至: results/Ablation/Table1_Ablation_spCLUE.csv")