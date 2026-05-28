"""
消融实验 v3 — 最终优化版
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
核心优化:
1. seed=0 只跑一次, 与主实验 grid search 一致
2. 数据加载前先 fix_seed(0), 保证 PCA 等随机操作与主实验一致
3. 使用主实验的 per-sample 最优参数
4. 为防止图构建等操作的随机性, 每个变体都从同一份 adata copy 出发
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

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"

PROJECT_ROOT = "/home/bio/lhz/spatialGDC"
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
import spatialgdc

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
# 2. 核心函数 — 一次加载 + 多变体共享
# ==========================================
def load_and_prepare_dlpfc(sample_name):
    """加载DLPFC数据，返回共享的 adata + 图 + keep_prob"""
    spatialgdc.fix_seed(0)  # 数据加载前 seed，保证 PCA 一致性
    data_path = os.path.join(DLPFC_DIR, sample_name)
    adata = spatialgdc.load_and_preprocess_st(data_path=data_path)
    df_meta = pd.read_csv(os.path.join(data_path, 'metadata.tsv'), sep='\t', index_col=0)
    adata.obs['Region'] = df_meta.loc[adata.obs_names, 'layer_guess']
    adata = adata[adata.obs['Region'].notna()].copy()
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()
    return adata


def load_and_prepare_brca(sample_name):
    """加载BRCA数据"""
    spatialgdc.fix_seed(0)
    brca_dir_specific = os.path.join(BRCA_DIR, sample_name)
    adata = sc.read_visium(brca_dir_specific)
    adata.var_names_make_unique()

    meta_path = os.path.join(BRCA_DIR, "metadata.tsv")
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
    return adata


def run_ablation_on_sample(dataset_type, sample_name, best_params):
    """对单个样本跑所有消融变体（seed=0 only, 与主实验一致）"""
    print(f"\n{'='*60}\nProcessing {dataset_type}: {sample_name}\n{'='*60}")

    # 加载数据 (内部已 fix_seed(0))
    if dataset_type == 'DLPFC':
        adata_base = load_and_prepare_dlpfc(sample_name)
        n_clusters = 5 if sample_name in ["151669", "151670", "151671", "151672"] else 7
    else:
        adata_base = load_and_prepare_brca(sample_name)
        n_clusters = adata_base.obs['Region'].dropna().nunique()

    # 构建图 (只构建一次)
    g_spatial = spatialgdc.prepare_graph(adata_base, "spatial")
    g_expr = spatialgdc.prepare_graph(adata_base, "expr")
    spatial_coords = adata_base.obsm["spatial"].copy()
    spatial_coords_norm = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))

    sigma = best_params["sigma"]
    gamma = best_params["gamma"]
    kappa = best_params["kappa"]
    base_keep_prob = spatialgdc.compute_spatial_keep_prob(g_expr, spatial_coords_norm, sigma=sigma)

    results = {}

    for mode in ablation_modes:
        # 每个变体重置 seed，保证训练过程一致
        spatialgdc.fix_seed(0)

        # 从同一份 base adata 拷贝
        adata = adata_base.copy()

        # 默认满血参数
        current_keep_prob = base_keep_prob
        use_instance_cl = True
        graph_dict = {"spatial": g_spatial, "expr": g_expr}
        current_gamma = gamma
        current_kappa = kappa

        # 变体配置
        if mode == "w/o Dual-graph":
            graph_dict = {"spatial": g_spatial, "expr": g_spatial}
            current_keep_prob = None
        elif mode == "w/o Instance-CL":
            use_instance_cl = False
        elif mode == "w/o Spatial-guide":
            current_keep_prob = None

        # 训练
        try:
            model = spatialgdc.SpatialGDC(
                input_data=adata.obsm["X_pca"],
                graph_dict=graph_dict,
                n_clusters=n_clusters,
                expr_keep_prob=current_keep_prob,
                use_instance_cl=use_instance_cl,
                gamma=current_gamma,
                kappa=current_kappa,
            )
            _, adata.obsm["spatialgdc_emb"], _ = model.train()

            # 聚类精修
            spatialgdc.clustering(adata, n_clusters, key="spatialgdc_emb", refinement=True, cluster_methods="mclust")
            cluster_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'

            ari = adjusted_rand_score(adata.obs["Region"], adata.obs[cluster_col])
            print(f"  [{mode}] ARI={ari:.4f}")
            results[mode] = ari

            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"  [{mode}] FAILED: {e}")
            results[mode] = np.nan

    del adata_base
    gc.collect()
    return results


# ==========================================
# 3. 执行
# ==========================================
print("Starting ablation experiments (seed=0, same as main experiment)...")

detail_rows = []

# 跑 DLPFC
for sample in dlpfc_samples:
    res = run_ablation_on_sample('DLPFC', sample, dlpfc_best_params[sample])
    for mode in ablation_modes:
        detail_rows.append({
            "Dataset": "DLPFC", "Sample": sample, "Mode": mode,
            "ARI": res.get(mode, np.nan),
        })

# 跑 BRCA
for sample in brca_samples:
    res = run_ablation_on_sample('BRCA', sample, brca_best_params)
    for mode in ablation_modes:
        detail_rows.append({
            "Dataset": "BRCA", "Sample": sample, "Mode": mode,
            "ARI": res.get(mode, np.nan),
        })

# ==========================================
# 4. 生成 Table 1
# ==========================================
df_detail = pd.DataFrame(detail_rows)
df_detail.to_csv("results/Ablation/Table1_Ablation_detail.csv", index=False)

# 计算 per-dataset 均值
table_rows = []
for mode in ablation_modes:
    dlpfc_mean = df_detail[(df_detail.Dataset=='DLPFC') & (df_detail.Mode==mode)]['ARI'].mean()
    brca_mean = df_detail[(df_detail.Dataset=='BRCA') & (df_detail.Mode==mode)]['ARI'].mean()
    table_rows.append({
        "Method": mode,
        "ARI(DLPFC)": f"{dlpfc_mean:.2f}",
        "ARI(BC)": f"{brca_mean:.2f}",
    })

df_table1 = pd.DataFrame(table_rows)

print("\n" + "=" * 60)
print("Table 1: Ablation Study (seed=0, per-sample最佳参数)")
print("=" * 60)
print(df_table1.to_string(index=False))
print("=" * 60)

df_table1.to_csv("results/Ablation/Table1_Ablation_spatialgdc.csv", index=False)

# 详细 per-sample 表
print("\nPer-sample detail:")
pivot = df_detail.pivot(index=["Dataset","Sample"], columns="Mode", values="ARI")
print(pivot.to_string(float_format="%.4f"))

# 对比 grid search 的 spatialGDC 结果
gs_df = pd.read_csv("SpatialGDC_DLPFC_GridSearch_Summary.csv")
gs_mean = gs_df['Best_ARI'].mean()
print(f"\nGrid search mean ARI (spatialGDC, seed=0): {gs_mean:.4f}")
ablation_full = df_detail[(df_detail.Dataset=='DLPFC') & (df_detail.Mode=='spatialGDC')]['ARI'].mean()
print(f"Ablation mean ARI (spatialGDC, seed=0): {ablation_full:.4f}")
if abs(gs_mean - ablation_full) < 0.01:
    print("✅ Results match grid search!")
else:
    print(f"⚠️  Difference: {gs_mean - ablation_full:.4f}")

print("\nDone!")
