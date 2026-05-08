import os
import sys
import warnings
import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp
# 🌟 新增：导入 NMI 计算模块
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from tqdm import tqdm
import gc      # 引入垃圾回收模块
import torch   # 引入 torch 以便调用 cuda 清理缓存
sc.settings.figdir = './figures/'

# 1. 环境锁定与确定性设置
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["R_HOME"] = "/home/bio/miniconda3/envs/spCLUE/lib/R"

warnings.filterwarnings("ignore")

# 2. 导入 底层模型核心包
spCLUE_ROOT_PATH = "/home/bio/lhz/spCLUE"
if spCLUE_ROOT_PATH not in sys.path:
    sys.path.append(spCLUE_ROOT_PATH)
import spCLUE

# 3. 配置参数与样本列表
samples_list = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676"
]

results = [] # 用于存储 ARI 和 NMI 结果

# 创建保存图片和 h5ad 数据的文件夹
os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True) # 🌟 新增结果保存目录

print(f"🚀 启动 SpatialGDC 全量测试 12 个切片，共计 {len(samples_list)} 个样本...")

# 4. 循环测试流程
for sample_name in samples_list:
    print(f"\n{'='*40}")
    print(f"正在处理样本: {sample_name}")
    print(f"{'='*40}")
    
    try:
        # 固定每个样本的随机种子，保证可复现性
        spCLUE.fix_seed(0)
        
        # ==========================================
        # 🌟 数据加载与预处理
        # ==========================================
        data_path = f"./dataset/DLPFC/{sample_name}/"
        adata = spCLUE.load_and_preprocess_st(data_path=data_path)
        
        # 自动判定聚类数 (151669-151672 是 5 层，其他是 7 层)
        n_clusters = 5 if sample_name in ["151669", "151670", "151671", "151672"] else 7
        print(f"检测到 Spot 数量: {adata.shape[0]}, 设置聚类数: {n_clusters}")

        # 🌟 核心修复：确保提前准备好 PCA 的载荷矩阵 (PCs)，用于后续基因重构
        if 'X_pca' not in adata.obsm.keys() or 'PCs' not in adata.varm.keys():
            print("  >>> 正在进行基础特征预处理并提取 200 维 PCA...")
            if 'log1p' not in adata.uns_keys():
                try:
                    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=3000)
                    sc.pp.normalize_total(adata, target_sum=1e4)
                    sc.pp.log1p(adata)
                except Exception:
                    pass 
            sc.tl.pca(adata, svd_solver='arpack', n_comps=200)
        
        # 提取主成分权重矩阵，留作逆变换
        pc_matrix = adata.varm['PCs']

        # ==========================================
        # 🌟 构建双图与空间先验概率
        # ==========================================
        g_spatia = spCLUE.prepare_graph(adata, "spatial")
        g_expr = spCLUE.prepare_graph(adata, "expr")
        graph_dict = {"spatial": g_spatia, "expr": g_expr}

        spatial_coords = adata.obsm["spatial"].copy()
        # 坐标 Min-Max 归一化！确保 sigma 参数对所有切片公平生效
        spatial_coords = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))
        
        expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords, sigma=0.5)

        # ==========================================
        # 🌟 模型训练 (接收底层触发的 Best 结果)
        # ==========================================
        spCLUE_model = spCLUE.spCLUE(
            input_data=adata.obsm["X_pca"], 
            graph_dict=graph_dict, 
            n_clusters=n_clusters,
            expr_keep_prob=expr_keep_prob  
        )
        
        # 🌟 修改：同时接收最佳 emb 和最佳 x_rec_pca
        _, adata.obsm["SpatialGDC_emb"], best_x_rec_pca = spCLUE_model.train()

        # ==========================================
        # 🌟 基因增强逆变换 (完美复用预处理时的 pc_matrix)
        # ==========================================
        if torch.is_tensor(best_x_rec_pca):
            best_x_rec_pca = best_x_rec_pca.detach().cpu().numpy()
            
        x_rec_gene = np.dot(best_x_rec_pca, pc_matrix.T)
        adata.layers["SpatialGDC_enhanced"] = x_rec_gene
        print(f"  ✨ 基因表达特征增强与去噪完成。")

        # ==========================================
        # 🌟 聚类精修 (mclust)
        # ==========================================
        try:
            pred = spCLUE.clustering(
                adata,
                n_clusters,
                key="SpatialGDC_emb", # 🌟 修改
                refinement=True,
                cluster_methods="mclust",
            )
        except Exception as e:
            print(f"  ⚠️ 空间精修遇到异常 ({e})，正在退回非 refinement 模式...")
            spCLUE.clustering(adata, n_clusters, key="SpatialGDC_emb", refinement=False, cluster_methods="mclust")

        cluster_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
        adata.obs['domain'] = adata.obs[cluster_col].astype('category')

        # ==========================================
        # 🌟 计算评测指标 (ARI & NMI)
        # ==========================================
        adata_eval = adata[adata.obs.Region.notna()].copy()
        ARI = adjusted_rand_score(adata_eval.obs["Region"], adata_eval.obs["domain"])
        NMI = normalized_mutual_info_score(adata_eval.obs["Region"], adata_eval.obs["domain"]) # 🌟 新增
        
        print(f"🏆 样本 {sample_name} 测试完成! ARI = {ARI:.4f} | NMI = {NMI:.4f}")

        # 记录结果
        results.append({"Sample": sample_name, "ARI": ARI, "NMI": NMI, "Spots": adata.shape[0], "Clusters": n_clusters})

        # ==========================================
        # 🌟 保存可视化结果与 h5ad 模型数据
        # ==========================================
        adata_eval.obs["SpatialGDC_Result"] = adata_eval.obs["domain"]
        sc.pl.spatial(
            adata_eval, 
            color=["Region", "SpatialGDC_Result"], 
            title=["Manual", f"{sample_name} (ARI={round(ARI, 3)})"],
            show=False,
            save=f"_{sample_name}_SpatialGDC_result.png"
        )
        
        # 🌟 新增：保存包含最佳 emb 和增强基因表达的 h5ad
        save_path = f"./results/DLPFC_{sample_name}_SpatialGDC.h5ad"
        adata.write_h5ad(save_path)
        print(f"💾 最佳特征及聚类结果已安全保存至: {save_path}")

    except Exception as e:
        print(f"❌ 样本 {sample_name} 出错: {e}")
        import traceback
        traceback.print_exc()
        results.append({"Sample": sample_name, "ARI": "Error", "NMI": "Error", "Spots": "N/A", "Clusters": "N/A"})
        
    finally:
        # ⚡ 核心修复模块：彻底清空当前切片占用的计算图和显存 ⚡
        if 'spCLUE_model' in locals(): del spCLUE_model
        if 'adata' in locals(): del adata
        if 'adata_eval' in locals(): del adata_eval
        if 'g_spatia' in locals(): del g_spatia
        if 'g_expr' in locals(): del g_expr
        if 'expr_keep_prob' in locals(): del expr_keep_prob
        if 'spatial_coords' in locals(): del spatial_coords
            
        # 强制 Python 垃圾回收
        gc.collect()
        
        # 强行清空 GPU 缓存的显存碎片
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"🧹 样本 {sample_name} 的显存清理完毕。")

# 5. 保存结果到 CSV
df_results = pd.DataFrame(results)

# 计算有效运行的平均值并打印
valid_results = df_results[df_results['ARI'] != 'Error']
if not valid_results.empty:
    avg_ari = valid_results['ARI'].mean()
    avg_nmi = valid_results['NMI'].mean()
    print(f"\n📊 12 个切片平均表现: Mean ARI = {avg_ari:.4f} | Mean NMI = {avg_nmi:.4f}")

csv_filename = "SpatialGDC_DLPFC_Metrics_Results.csv"
df_results.to_csv(csv_filename, index=False)

print(f"\n{'='*50}")
print(f"🎉 所有测试已完成！")
print(f"📊 汇总结果已保存至: {csv_filename}")
print(df_results)
print(f"{'='*50}")