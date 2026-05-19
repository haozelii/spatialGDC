"""
消融实验 v4 — 最终版
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
策略:
1. spatialGDC 满血版: 直接使用主实验 grid search 的 per-sample 最优结果
   (SpatialGDC_DLPFC_GridSearch_Summary.csv + BRCA_BEST.h5ad)
2. 3 个消融变体: 在相同数据+seed=0 下运行，只改对应组件
3. 保证消融实验的公平性: 同一份数据、同一套参数、同一个 seed
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

ablation_modes_only = [
    "w/o Dual-graph",
    "w/o Instance-CL",
    "w/o Spatial-guide",
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

brca_best_params = {"sigma": 0.7, "gamma": 2.0, "kappa": 0.1}

os.makedirs("results/Ablation", exist_ok=True)

# ==========================================
# 2. 读取主实验 spatialGDC 结果 (grid search)
# ==========================================
print("Loading grid search results for spatialGDC...")

# DLPFC: 直接从 grid search CSV 读取
df_gs = pd.read_csv("SpatialGDC_DLPFC_GridSearch_Summary.csv")
spatialGDC_dlpfc = {}
for _, row in df_gs.iterrows():
    key = str(int(row['Sample'])) if isinstance(row['Sample'], float) else str(row['Sample'])
    spatialGDC_dlpfc[key] = row['Best_ARI']
print(f"  DLPFC mean ARI: {np.mean(list(spatialGDC_dlpfc.values())):.4f}")

# BRCA: 从 best h5ad 读取
adata_brca = sc.read_h5ad("results_best/BRCA_BEST.h5ad")
brca_ari = adjusted_rand_score(adata_brca.obs['fine_annot_type'], adata_brca.obs['mclust_refined'])
print(f"  BRCA ARI: {brca_ari:.4f}")
del adata_brca
gc.collect()

# ==========================================
# 3. 跑消融变体
# ==========================================
def load_and_prepare_dlpfc(sample_name):
    """加载DLPFC数据 — 模拟 grid search 的流程: 不在加载前 set seed"""
    data_path = os.path.join(DLPFC_DIR, sample_name)
    adata = spCLUE.load_and_preprocess_st(data_path=data_path)
    df_meta = pd.read_csv(os.path.join(data_path, 'metadata.tsv'), sep='\t', index_col=0)
    adata.obs['Region'] = df_meta.loc[adata.obs_names, 'layer_guess']
    adata = adata[adata.obs['Region'].notna()].copy()
    if sp.issparse(adata.X):
        adata.X = adata.X.toarray()
    return adata

# --- DLPFC ---
print("\n" + "="*60)
print("Running DLPFC ablation variants...")
print("="*60)

ablation_dlpfc = {mode: {} for mode in ablation_modes_only}

for sample in dlpfc_samples:
    print(f"\n--- {sample} ---")
    
    # 加载数据 (与 grid search 一致: 不在加载前 set seed)
    adata_base = load_and_prepare_dlpfc(sample)
    n_clusters = 5 if sample in ["151669", "151670", "151671", "151672"] else 7
    params = dlpfc_best_params[sample]
    
    # 构建图
    g_spatial = spCLUE.prepare_graph(adata_base, "spatial")
    g_expr = spCLUE.prepare_graph(adata_base, "expr")
    spatial_coords = adata_base.obsm["spatial"].copy()
    spatial_coords_norm = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))
    base_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords_norm, sigma=params["sigma"])
    
    print(f"  spatialGDC (grid search): ARI={spatialGDC_dlpfc[sample]:.4f}")
    
    for mode in ablation_modes_only:
        spCLUE.fix_seed(0)  # seed 在训练前 set, 与 grid search 一致
        adata = adata_base.copy()
        
        # 默认参数
        current_keep_prob = base_keep_prob
        use_instance_cl = True
        graph_dict = {"spatial": g_spatial, "expr": g_expr}
        
        # 变体配置
        if mode == "w/o Dual-graph":
            graph_dict = {"spatial": g_spatial, "expr": g_spatial}
            current_keep_prob = None
        elif mode == "w/o Instance-CL":
            use_instance_cl = False
        elif mode == "w/o Spatial-guide":
            current_keep_prob = None
        
        model = spCLUE.spCLUE(
            input_data=adata.obsm["X_pca"],
            graph_dict=graph_dict,
            n_clusters=n_clusters,
            expr_keep_prob=current_keep_prob,
            use_instance_cl=use_instance_cl,
            gamma=params["gamma"],
            kappa=params["kappa"],
        )
        _, adata.obsm["spCLUE_emb"], _ = model.train()
        spCLUE.clustering(adata, n_clusters, key="spCLUE_emb", refinement=True, cluster_methods="mclust")
        cluster_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
        
        ari = adjusted_rand_score(adata.obs["Region"], adata.obs[cluster_col])
        ablation_dlpfc[mode][sample] = ari
        print(f"  {mode}: ARI={ari:.4f}")
        
        del model, adata
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    del adata_base
    gc.collect()

# --- BRCA ---
print("\n" + "="*60)
print("Running BRCA ablation variants...")
print("="*60)

ablation_brca = {mode: 0.0 for mode in ablation_modes_only}

# 加载 BRCA 数据
sample = "V1_Human_Breast_Cancer_Block_A_Section_1"
adata_base = sc.read_visium(os.path.join(BRCA_DIR, sample))
adata_base.var_names_make_unique()

meta_path = os.path.join(BRCA_DIR, "metadata.tsv")
meta_df = pd.read_csv(meta_path, sep='\t', index_col=0)
adata_base.obs.index = adata_base.obs.index.astype(str).str.replace('-1', '', regex=False).str.strip()
meta_df.index = meta_df.index.astype(str).str.replace('-1', '', regex=False).str.strip()
adata_base.obs = adata_base.obs.merge(meta_df, left_index=True, right_index=True, how='left')
adata_base.obs['Region'] = adata_base.obs['fine_annot_type']

sc.pp.filter_genes(adata_base, min_counts=1)
sc.pp.filter_cells(adata_base, min_counts=1)
sc.pp.normalize_total(adata_base, target_sum=1e4)
sc.pp.log1p(adata_base)
sc.pp.scale(adata_base)

from sklearn.decomposition import PCA
pca_model = PCA(n_components=200, random_state=0)
adata_base.obsm["X_pca"] = pca_model.fit_transform(
    adata_base.X.toarray() if sp.issparse(adata_base.X) else adata_base.X
)

n_clusters = adata_base.obs['Region'].dropna().nunique()
params = brca_best_params

g_spatial = spCLUE.prepare_graph(adata_base, "spatial")
g_expr = spCLUE.prepare_graph(adata_base, "expr")
spatial_coords = adata_base.obsm["spatial"].copy()
spatial_coords_norm = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))
base_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords_norm, sigma=params["sigma"])

print(f"  spatialGDC (best result): ARI={brca_ari:.4f}")

for mode in ablation_modes_only:
    spCLUE.fix_seed(0)
    adata = adata_base.copy()
    
    current_keep_prob = base_keep_prob
    use_instance_cl = True
    graph_dict = {"spatial": g_spatial, "expr": g_expr}
    
    if mode == "w/o Dual-graph":
        graph_dict = {"spatial": g_spatial, "expr": g_spatial}
        current_keep_prob = None
    elif mode == "w/o Instance-CL":
        use_instance_cl = False
    elif mode == "w/o Spatial-guide":
        current_keep_prob = None
    
    model = spCLUE.spCLUE(
        input_data=adata.obsm["X_pca"],
        graph_dict=graph_dict,
        n_clusters=n_clusters,
        expr_keep_prob=current_keep_prob,
        use_instance_cl=use_instance_cl,
        gamma=params["gamma"],
        kappa=params["kappa"],
    )
    _, adata.obsm["spCLUE_emb"], _ = model.train()
    spCLUE.clustering(adata, n_clusters, key="spCLUE_emb", refinement=True, cluster_methods="mclust")
    cluster_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
    
    ari = adjusted_rand_score(adata.obs["Region"], adata.obs[cluster_col])
    ablation_brca[mode] = ari
    print(f"  {mode}: ARI={ari:.4f}")
    
    del model, adata
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

del adata_base
gc.collect()

# ==========================================
# 4. 生成 Table 1
# ==========================================
all_modes = ["spatialGDC"] + ablation_modes_only

# DLPFC means
dlpfc_means = {}
for mode in all_modes:
    if mode == "spatialGDC":
        dlpfc_means[mode] = np.mean(list(spatialGDC_dlpfc.values()))
    else:
        dlpfc_means[mode] = np.mean(list(ablation_dlpfc[mode].values()))

# BRCA
brca_means = {}
for mode in all_modes:
    if mode == "spatialGDC":
        brca_means[mode] = brca_ari
    else:
        brca_means[mode] = ablation_brca[mode]

table_rows = []
for mode in all_modes:
    table_rows.append({
        "Method": mode,
        "ARI(DLPFC)": f"{dlpfc_means[mode]:.2f}",
        "ARI(BC)": f"{brca_means[mode]:.2f}",
    })

df_table1 = pd.DataFrame(table_rows)

print("\n" + "=" * 60)
print("Table 1: Ablation Study")
print("=" * 60)
print(df_table1.to_string(index=False))
print("=" * 60)

df_table1.to_csv("results/Ablation/Table1_Ablation_spCLUE.csv", index=False)

# Per-sample detail
print("\nPer-sample detail (DLPFC):")
print(f"{'Sample':<12} {'spatialGDC':>10} {'w/o Dual':>10} {'w/o InsCL':>10} {'w/o SpGui':>10}")
for sample in dlpfc_samples:
    vals = [spatialGDC_dlpfc.get(sample, np.nan)]
    for mode in ablation_modes_only:
        vals.append(ablation_dlpfc[mode].get(sample, np.nan))
    print(f"{sample:<12} {vals[0]:>10.4f} {vals[1]:>10.4f} {vals[2]:>10.4f} {vals[3]:>10.4f}")

# 保存详细数据
detail_rows = []
for sample in dlpfc_samples:
    for mode in all_modes:
        if mode == "spatialGDC":
            ari_val = spatialGDC_dlpfc.get(sample, np.nan)
        else:
            ari_val = ablation_dlpfc[mode].get(sample, np.nan)
        detail_rows.append({"Dataset": "DLPFC", "Sample": sample, "Mode": mode, "ARI": ari_val})

for mode in all_modes:
    if mode == "spatialGDC":
        ari_val = brca_ari
    else:
        ari_val = ablation_brca[mode]
    detail_rows.append({"Dataset": "BRCA", "Sample": "V1_Breast", "Mode": mode, "ARI": ari_val})

df_detail = pd.DataFrame(detail_rows)
df_detail.to_csv("results/Ablation/Table1_Ablation_detail.csv", index=False)

print("\nDone! Results saved to results/Ablation/")
