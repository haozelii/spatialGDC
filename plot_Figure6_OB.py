import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

print("🚀 开始绘制 Figure 6: 小鼠嗅球全套可视化验证 (终极完整版)...")

# ==========================================
# 1. 配置文件路径与数据加载
# ==========================================
res_file = "./results/Mouse_OB_our.h5ad" 
if not os.path.exists(res_file):
    print(f"❌ 找不到结果文件 {res_file}，请确保 spCLUE 模型已运行完毕！")
    exit()

adata = sc.read_h5ad(res_file)

# 获取聚类标签
pred_col = 'mclust_refined' if 'mclust_refined' in adata.obs.columns else 'mclust'
if 'domain' not in adata.obs.columns:
    adata.obs['domain'] = adata.obs[pred_col]
adata.obs['domain'] = adata.obs['domain'].astype(str)

# ==========================================
# 🌟 A：计算 SC 分数 (Silhouette Coefficient)
# ==========================================
print("🧮 正在计算 SC (轮廓系数) 分数...")
if 'spCLUE' in adata.obsm.keys():
    sc_score = silhouette_score(adata.obsm['spCLUE'], adata.obs['domain'])
    print(f"\n🏆 模型当前潜入空间的 SC 分数 (Silhouette Coefficient): {sc_score:.4f}\n")
else:
    print("⚠️ 找不到 'spCLUE' 降维特征，跳过 SC 分数计算。")

# ==========================================
# 🌟 B：映射生物学标签 (解剖学注释)
# ==========================================
print("🏷️ 正在将数字簇映射为小鼠嗅球解剖学标签...")
# ⚠️ 注意：这是一个默认的映射字典。
# 当你画出图后，你需要对照 Panel C/D 的基因表达结果，将真实的数字与标签对应起来并修改这里！
cluster2name = {
    # Evidence-based mapping from 8 marker genes + spatial position
    # Gabra1->GL, Slc6a11->EPL, Mbp->MCL, Atp2b4->GCL, Pcp4->ONL, Nrgn->IPL, Cck->EPLi, Kctd12->GL/EPL
    '0': 'ONL',  '6': 'ONL',       # Pcp4=2.10(ONL), peripheral
    '10': 'GL',  '4': 'GL', '5': 'GL',  # Gabra1=1.12(GL), Kctd12=0.77(GL/EPL)
    '3': 'EPL',  '8': 'EPL', '9': 'EPL', '7': 'EPL',  # Cck, Slc6a11 (EPL)
    '2': 'MCL',                    # Mbp=0.93(MCL)
    '1': 'GCL',                    # Nrgn=0.67, Atp2b4=0.23, innermost core
}

# 替换数字为文本标签
adata.obs['domain'] = adata.obs['domain'].map(lambda x: cluster2name.get(x, f"Cluster_{x}"))

# 规范化分类顺序（从外层到内层），确保图例排版学术且美观
category_order = ['ONL', 'GL', 'EPL', 'MCL', 'GCL', 'RMS']
existing_cats = [cat for cat in category_order if cat in adata.obs['domain'].unique()]
# 补充未匹配的数字簇
for cat in np.unique(adata.obs['domain']):
    if cat not in existing_cats:
        existing_cats.append(cat)
        
adata.obs['domain'] = pd.Categorical(adata.obs['domain'], categories=existing_cats, ordered=True)
print(f"✅ 标签替换完成！当前包含的解剖结构: {existing_cats}")

# ==========================================
# 🌟 C：核心方向修正 (交换空间坐标轴)
# ==========================================
print("🔄 正在交换空间坐标轴以匹配参考图的垂直对齐方向...")
spatial = adata.obsm['spatial'].copy()
# 将原 x 变为新 y，原 y 变为新 x
adata.obsm['spatial'] = spatial[:, [1, 0]] 

# ==========================================
# 2. 数据预处理
# ==========================================
marker_genes = ["Gabra1", "Slc6a11", "Mbp", "Atp2b4", "Pcp4", "Nrgn", "Cck", "Kctd12"]

if 'log1p' not in adata.uns_keys():
    print(">>> 正在对基因表达量进行 Log 标准化...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

valid_genes = [g for g in marker_genes if g in adata.var_names]
os.makedirs("figures", exist_ok=True)

# ==========================================
# 3. Panel B: 纯 Matplotlib 绘制空间域
# ==========================================
print("🎨 正在绘制 Panel B: spCLUE 空间域...")
fig_b, ax_b = plt.subplots(figsize=(8, 8), facecolor='white')

domains = adata.obs['domain'].cat.categories
n_domains = len(domains)
cmap_colors = plt.get_cmap('tab20')(np.linspace(0, 1, n_domains))

x = adata.obsm['spatial'][:, 0]
y = adata.obsm['spatial'][:, 1]
labels = adata.obs['domain']

for i, cat in enumerate(domains):
    idx = (labels == cat)
    ax_b.scatter(
        x[idx], y[idx], 
        color=cmap_colors[i], 
        label=cat,
        s=12, marker="o", edgecolors='none', antialiased=False
    )

ax_b.set_title("spCLUE Domains (Mouse OB)", fontsize=28, fontweight='bold', pad=20)
ax_b.set_aspect('equal')
ax_b.set_xticks([]); ax_b.set_yticks([]) # 去除坐标轴刻度

for spine in ax_b.spines.values():
    spine.set_linewidth(2)

lgnd = ax_b.legend(loc='center left', bbox_to_anchor=(1, 0.5), scatterpoints=1, fontsize=14, frameon=False, title="Anatomy Layers")
for handle in lgnd.legendHandles:
    handle.set_sizes([120]) 

fig_b.savefig("figures/Fig6_PanelB_Domains.png", dpi=300, bbox_inches='tight')

# ==========================================
# 4. Panel C: 标志基因空间表达热图
# ==========================================
print("🎨 正在绘制 Panel C: 8联排基因空间热图...")
fig_c, axes_c = plt.subplots(1, len(valid_genes), figsize=(4 * len(valid_genes), 4), facecolor='white')
if len(valid_genes) == 1: axes_c = [axes_c] 

for i, gene in enumerate(valid_genes):
    ax = axes_c[i]
    exp = adata[:, gene].X.toarray().flatten() if hasattr(adata[:, gene].X, 'toarray') else adata[:, gene].X.flatten()
    order = np.argsort(exp) # 核心：将高表达细胞置于顶层
    
    scatter = ax.scatter(
        x[order], y[order], 
        c=exp[order], cmap="Reds", s=10, marker='o', edgecolors='none'
    )
    
    ax.set_title(gene, fontsize=22, fontstyle='italic', fontweight='bold', pad=10)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

plt.subplots_adjust(wspace=0.1)
fig_c.savefig("figures/Fig6_PanelC_Markers.png", dpi=300, bbox_inches='tight')

# ==========================================
# 5. Panel D: 域-基因表达小提琴图
# ==========================================
print("🎨 正在绘制 Panel D: 基因表达特异性小提琴图...")

# 极简调用，防止 Matplotlib 与 Seaborn 参数冲突
sc.pl.stacked_violin(
    adata, 
    var_names=valid_genes, 
    groupby='domain', 
    title="Marker Gene Expression across Anatomical Layers",
    standard_scale='var', 
    show=False
)
plt.savefig("figures/Fig6_PanelD_Violin.png", dpi=300, bbox_inches='tight')

print("\n🎉 Figure 6 终极完整版全套绘图完成！SC分数、解剖学标签、垂直方向均已就绪！")