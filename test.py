import os
import sys
import warnings
import pandas as pd
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score
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

# 2. 导入 spCLUE 
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

results = [] # 用于存储 ARI 结果

# 创建保存图片的文件夹
if not os.path.exists("figures"):
    os.makedirs("figures")

print(f"🚀 开始全量测试 12 个切片，共计 {len(samples_list)} 个样本...")

# 4. 循环测试流程
for sample_name in samples_list:
    print(f"\n{'='*30}")
    print(f"正在处理样本: {sample_name}")
    print(f"{'='*30}")
    
    try:
        # 固定每个样本的随机种子，保证可复现性
        spCLUE.fix_seed(0)
        
        # 加载与预处理
        data_path = f"./dataset/DLPFC/{sample_name}/"
        adata = spCLUE.load_and_preprocess_st(data_path=data_path)
        
        # 自动判定聚类数 (151669-151672 是 5 层，其他是 7 层)
        n_clusters = 5 if sample_name in ["151669", "151670", "151671", "151672"] else 7
        print(f"检测到 Spot 数量: {adata.shape[0]}, 设置聚类数: {n_clusters}")

        # 构建空间图与表达图
        g_spatia = spCLUE.prepare_graph(adata, "spatial")
        g_expr = spCLUE.prepare_graph(adata, "expr")
        graph_dict = {"spatial": g_spatia, "expr": g_expr}

        # 🌟 新增：计算基于空间先验的特征图保留概率
        spatial_coords = adata.obsm["spatial"].copy()
        
        # 🌟 核心修复：坐标 Min-Max 归一化！
        # 将 X 和 Y 坐标分别严格缩放到 0-1 之间，确保 sigma 参数对所有切片公平生效
        spatial_coords = (spatial_coords - spatial_coords.min(axis=0)) / (spatial_coords.max(axis=0) - spatial_coords.min(axis=0))
        
        # 计算保留概率 (此时 sigma=0.5 将在一个统一的 0-1 坐标系下工作)
        expr_keep_prob = spCLUE.compute_spatial_keep_prob(g_expr, spatial_coords, sigma=0.5)

        # 实例化并训练模型，传入我们算好的 expr_keep_prob
        spCLUE_model = spCLUE.spCLUE(
            input_data=adata.obsm["X_pca"], 
            graph_dict=graph_dict, 
            n_clusters=n_clusters,
            expr_keep_prob=expr_keep_prob  # 👈 传入概率
        )
        _, adata.obsm["spCLUE"] = spCLUE_model.train()

        # mclust 聚类精修
        pred = spCLUE.clustering(
            adata,
            n_clusters,
            key="spCLUE",
            refinement=True,
            cluster_methods="mclust",
        )

        # 过滤掉无标签点并计算 ARI
        adata_eval = adata[adata.obs.Region.notna()].copy()
        ARI = adjusted_rand_score(adata_eval.obs["Region"], adata_eval.obs["mclust_refined"])
        
        print(f"✅ 样本 {sample_name} 测试完成! ARI = {ARI:.4f}")

        # 记录结果
        results.append({"Sample": sample_name, "ARI": ARI, "Spots": adata.shape[0], "Clusters": n_clusters})

        # 保存可视化结果
        adata_eval.obs["spCLUE_Result"] = adata_eval.obs["mclust_refined"]
        sc.pl.spatial(
            adata_eval, 
            color=["Region", "spCLUE_Result"], 
            title=["Manual", f"{sample_name} (ARI={round(ARI, 3)})"],
            show=False,
            save=f"_{sample_name}_intersection_result.png" # 稍微修复了一下保存路径，放到 figures 文件夹下
        )

    except Exception as e:
        print(f"❌ 样本 {sample_name} 出错: {e}")
        results.append({"Sample": sample_name, "ARI": "Error", "Spots": "N/A", "Clusters": "N/A"})
        
    finally:
        # ⚡ 核心修复模块：彻底清空当前切片占用的计算图和显存 ⚡
        if 'spCLUE_model' in locals():
            del spCLUE_model
        if 'adata' in locals():
            del adata
        if 'adata_eval' in locals():
            del adata_eval
        if 'g_spatia' in locals():
            del g_spatia
        if 'g_expr' in locals():
            del g_expr
        if 'expr_keep_prob' in locals(): 
            del expr_keep_prob
        if 'spatial_coords' in locals(): # 顺手把提取的坐标也清掉
            del spatial_coords
            
        # 强制 Python 垃圾回收
        gc.collect()
        
        # 强行清空 GPU 缓存的显存碎片
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"🧹 样本 {sample_name} 的显存清理完毕。")

# 5. 保存结果到 CSV
df_results = pd.DataFrame(results)
csv_filename = "spCLUE_Intersection_ARI_Results.csv"
df_results.to_csv(csv_filename, index=False)

print(f"\n{'='*50}")
print(f"🎉 所有测试已完成！")
print(f"📊 汇总结果已保存至: {csv_filename}")
print(df_results)
print(f"{'='*50}")