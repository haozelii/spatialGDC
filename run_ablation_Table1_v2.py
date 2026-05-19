"""
消融实验 v2 — 优化版
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
改进点:
1. 使用主实验的 per-sample 最优参数（grid search 结果），不做二次调参
2. 3 个 seed 取均值，减少方差
3. DLPFC 12 切片 + BRCA 1 切片
4. 4 个变体: spatialGDC / w/o Dual-graph / w/o Instance-CL / w/o Spatial-guide
5. 满血版参数跟主实验完全一致，消融变体只改对应组件
"""

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
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"

spCLUE_ROOT_PATH = "/home/bio/lhz/spatialGDC"
if spCLUE_ROOT_PATH not in sys.path:
    sys.path.append(spCLUE_ROOT_PATH)
import spCLUE

warnings.filterwarnings("ignore")

# ==========================================
# 1. 配置
# ==========================================
DLPFC_DIR = "/home/bio/lhz/spatialGDC/dataset/DLPFC/"
BRCA_DIR = "/home/bio/lhz/spatialGDC/dataset/BRCA1/"

dlpfc_samples = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]
brca_samples = ["V1_Human_Breast_Cancer_Block_A_Section_1"]

SEEDS = [0, 1, 2]

ablation_modes = [
    "spatialGDC",           # 满血版
    "w/o Dual-graph",       # 双图退化为单图 (expr→spatial)
    "w/o Instance-CL",      # 去掉实例级对比损失
    "w/o Spatial-guide",    # 去掉空间引导丢边
]

# DLPFC 每个切片的最优参数 (来自 Grid Search)
dlpfc_best_params = {
    "151507": {"sigma": 0.5, "gamma": 1.0, "kappa": 0.1},
    "151508": {"sigma": 0.5, "gamma": 1.0, "kappa": 0.1},
    "151509": {"sigma": 0.5, "gamma": 1.0, "kappa": 0.1},
    "151510": {"sigma": 0.6, "gamma": 1.5, "kappa": 1.0},
    "151669": {"sigma": 0.5, "gamma": 1.5, "kappa": 1.0},
    "151670": {"sigma": 0.6, "gamma": 1.5, "kappa": 1.0},
    "151671": {"sigma": 0.5, "gamma": 1.0, "kappa": 0.1},
    "151672": {"sigma": 0.5, "gamma": 1.0, "kappa": 0.1},
    "151673": {"sigma": 0.5, "gamma": 1.0, "kappa": 0.1},
    "151674": {"sigma": 0.4, "gamma": 1.5, "kappa": 0.1},
    "151675": {"sigma": 0.4, "gamma": 1.0, "kappa": 1.0},
    "151676": {"sigma": 0.4, "gamma": 2.0, "kappa": 0.1},
}

# BRCA 最优参数
brca_best_params = {"sigma": 0.7, "gamma": 2.0, "kappa": 0.1}

os.makedirs("results/Ablation", exist_ok=True)

# ==========================================
# 2. 核心函数
# ==========================================
def run_ablation_on_sample(dataset_type, sample_name, base_dir, best_params):
    """对单个样本跑所有消融变体×多seed"""
    print(f"\n{'='*60}\nProcessing {dataset_type}: {sample_name}\n{'='*60}")

    # --- A. 数据加载 ---
    if dataset_type == 'DLPFC':
        data_path = os.path.join(base_dir, sample_name)
        adata = spCLUE.load_and_preprocess_st(data_path=data_path)
        df_meta = pd.read_csv(os.path.join(data_path, 'metadata.tsv'), sep='\t', index_col=0)
        adata.obs['Region'] = df_meta.loc[adata.obs_names, 'layer_guess']
        n_clusters = 5 if sample_name in ["151669", "151670", "151671", "151672"] else 7

    elif dataset_type == 'BRCA':
        brca_dir_specific = os.path.join(base_dir, sample_name)
        adata = sc.read_visium(brca_dir_specific)
        adata.var_names_make_unique()

        meta_path = os.path.join(base_dir, "metadata.tsv")
        meta_df = pd.read_csv(meta_path, sep='\t', index_col=0)
        adata.obs.index = adata.obs.index.astype(str).str.replace('-1', '', regex=False).str.strip()
        meta_df.index = meta_df.index.astype(str).str.replace('-1', '', regex=False).str.strip()
        adata.obs = adata.obs.merge(meta_df, left_index=True, right_index=True, how='left')
        adata.obs['Region'] = adata.obs['fine_annot_type']

        # 预处理 (与 run_BRCA_Enhanced.py 一致)
        sc.pp.filter_genes(adata, min_counts=1)
        sc.pp.filter_cells(adata, min_counts=1)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata)

        from sklearn.decomposition import PCA
        pca_model = PCA(n_components=200, random_state=0)
        adata.obsm["X_pca"] = pca_model.fit_transform(adata.X.toarray() if sp.issparse(adata.X) else adata.X)

        n_clusters = adata.obs['Region'].dropna().nunique()

    adata = adata[adata.obs['Region'].notna()].copy()
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()

    # --- B. 构建基础图 ---
    g_spatial = spCLUE.prepare_graph(adata, "spatial")
    g_expr = spCLUE.prepare_graph(adata, "expr")
    spatial_coords = adata.obsm["spatial"].copy()
    spatial_coords_norm = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))

    sigma = best_params["sigma"]
    gamma = best_params["gamma"]
    kappa = best_params["kappa"]

    # 用最优参数计算保留概率
    base_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords_norm, sigma=sigma)

    # --- C. 遍历变体 × Seeds ---
    # results[mode] = [ari_seed0, ari_seed1, ari_seed2, ...]
    results = {mode: [] for mode in ablation_modes}

    for seed in SEEDS:
        print(f"\n--- Seed {seed} ---")

        for mode in ablation_modes:
            spCLUE.fix_seed(seed)

            # 默认满血参数
            current_keep_prob = base_keep_prob
            use_instance_cl = True
            graph_dict = {"spatial": g_spatial, "expr": g_expr}
            current_gamma = gamma
            current_kappa = kappa

            # 变体配置
            if mode == "w/o Dual-graph":
                # 双图退化为单图：expr 分支也走 spatial 图
                graph_dict = {"spatial": g_spatial, "expr": g_spatial}
                current_keep_prob = None  # 单图不需要保留概率
            elif mode == "w/o Instance-CL":
                use_instance_cl = False
            elif mode == "w/o Spatial-guide":
                current_keep_prob = None  # 退化为均匀随机丢边

            # 训练
            try:
                model = spCLUE.spCLUE(
                    input_data=adata.obsm["X_pca"].copy(),
                    graph_dict=graph_dict,
                    n_clusters=n_clusters,
                    expr_keep_prob=current_keep_prob,
                    use_instance_cl=use_instance_cl,
                    gamma=current_gamma,
                    kappa=current_kappa,
                )
                _, adata.obsm["spCLUE_emb"], _ = model.train()

                # 聚类精修
                spCLUE.clustering(adata, n_clusters, key="spCLUE_emb", refinement=True, cluster_methods="mclust")
                cluster_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'

                ari = adjusted_rand_score(adata.obs["Region"], adata.obs[cluster_col])
                print(f"  [{mode}] seed={seed} ARI={ari:.4f}")
                results[mode].append(ari)

                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"  [{mode}] seed={seed} FAILED: {e}")
                results[mode].append(np.nan)

    return results


# ==========================================
# 3. 执行
# ==========================================
print(f"Starting ablation experiments (seeds={SEEDS})...")

# 汇总: final[mode][dataset] = list of per-sample mean ARIs
final_results = {mode: {"DLPFC": [], "BRCA": []} for mode in ablation_modes}
# 详细: per-sample per-seed
detail_results = []

# 跑 DLPFC
for sample in tqdm(dlpfc_samples, desc="DLPFC"):
    res = run_ablation_on_sample('DLPFC', sample, DLPFC_DIR, dlpfc_best_params[sample])
    for mode in ablation_modes:
        mean_ari = np.nanmean(res[mode])
        final_results[mode]["DLPFC"].append(mean_ari)
        for s_idx, ari_val in enumerate(res[mode]):
            detail_results.append({
                "Dataset": "DLPFC", "Sample": sample, "Mode": mode,
                "Seed": SEEDS[s_idx], "ARI": ari_val,
            })

# 跑 BRCA
for sample in tqdm(brca_samples, desc="BRCA"):
    res = run_ablation_on_sample('BRCA', sample, BRCA_DIR, brca_best_params)
    for mode in ablation_modes:
        mean_ari = np.nanmean(res[mode])
        final_results[mode]["BRCA"].append(mean_ari)
        for s_idx, ari_val in enumerate(res[mode]):
            detail_results.append({
                "Dataset": "BRCA", "Sample": sample, "Mode": mode,
                "Seed": SEEDS[s_idx], "ARI": ari_val,
            })

# ==========================================
# 4. 生成 Table 1
# ==========================================
table_rows = []
for mode in ablation_modes:
    dlpfc_mean = np.mean(final_results[mode]["DLPFC"])
    brca_mean = np.mean(final_results[mode]["BRCA"])
    table_rows.append({
        "Method": mode,
        "ARI(DLPFC)": f"{dlpfc_mean:.2f}",
        "ARI(BC)": f"{brca_mean:.2f}",
    })

df_table1 = pd.DataFrame(table_rows)

print("\n" + "=" * 60)
print("Table 1: Ablation Study (mean over 3 seeds)")
print("=" * 60)
print(df_table1.to_string(index=False))
print("=" * 60)

# 保存
df_table1.to_csv("results/Ablation/Table1_Ablation_spCLUE.csv", index=False)

# 详细结果
df_detail = pd.DataFrame(detail_results)
df_detail.to_csv("results/Ablation/Table1_Ablation_detail.csv", index=False)

# 也输出每个 sample 的 per-mode mean (3-seed average)
print("\nPer-sample detail (3-seed mean):")
pivot = df_detail.groupby(["Dataset", "Sample", "Mode"])["ARI"].mean().reset_index()
pivot_table = pivot.pivot(index=["Dataset", "Sample"], columns="Mode", values="ARI")
print(pivot_table.to_string(float_format="%.4f"))

print("\nDone! Results saved to results/Ablation/")
