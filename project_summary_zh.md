# SpatialGDC — 高保真科研上下文（v3·术语锚定版）

---

## 1. Project Overview

SpatialGDC 是一个面向空间转录组学（ST）数据空间域识别任务的 Informed Machine Learning 框架。输入为 10x Visium / Stereo-seq / Slide-seqV2 平台的空间基因表达矩阵与二维物理坐标，输出为每个捕获位点（spot）的空间域归属标签。当前目标是将 SpatialGDC 投稿至生物信息学或计算生物学顶刊。方法论路线：在空间图和表达图两个互补模态上，通过共享权重 GCN 编码器构建非对称对比视图，以空间先验引导的动态图拓扑优化纯化特征图的信息流，并辅以同质近邻保留的软对比优化维护组织连续性，实现对复杂组织架构的高精度解析。

---

## 2. Problem Definition

### 痛点一：零膨胀驱动的远端伪关联与空间不敏感的拓扑污染

空间转录组数据存在极度的稀疏性——受限于测序通量和技术性 dropout 效应，大量基因在多数 spot 上表现为零值。两个物理空间上相距极远、毫无生物学关联的 spot，仅因共享大量"技术性零值"，在 Pearson 相关空间中即表现出虚假的高相似度。传统 GNN 框架缺乏对物理空间距离的硬约束，盲目地在这些远端伪边上进行全局消息传递，导致：(i) 远端噪声信号污染局部微环境；(ii) 引发特征过平滑（over-smoothing）；(iii) 最终模糊组织边界，尤其在层状结构紧密叠加的脑区切片中表现尤为突出。

### 痛点二：同质近邻的语义冲突

标准 InfoNCE 对比损失将所有非锚点样本均视为等权负样本进行强制排斥（hard repulsion）。在空间结构化的组织中，物理相邻且转录组相似的 spot 极可能属于同一功能域（如在皮层第 4 层中相邻的两个神经元）。强制排斥这些"同质近邻"严重违背组织的局部连续性先验——模型被训练去撕裂本就应当聚合的同类细胞表征，导致特征空间碎裂（feature space fragmentation），从而抹杀同类细胞间至关重要的生物学共性，在边界处产生碎片化聚类。

---

## 3. Core Innovation

**一句话总结：** SpatialGDC 通过非对称对比视图构建引入物理空间锚点进行图拓扑纯化，并通过同质近邻保留的软对比优化，在防止特征坍缩的同时维护解剖学边界的连续性。

### 创新一：非对称对比视图构建（Asymmetric Contrastive View Construction）

传统对比学习采用对称加噪策略——对两个视图施加相同类型的随机扰动（如均匀 DropEdge、特征 Dropout）。SpatialGDC 打破此范式：对空间图施加轻量均匀 DropEdge（V3），对表达图则施加基于空间先验引导的动态图拓扑优化（SP-DropEdge，V4）。后者的核心哲学是：在 ST 数据中，物理距离是唯一可用于区分"真实生物学关联"和"技术噪声伪关联"的绝对锚点。SP-DropEdge 使用高斯衰减函数将物理距离编码为每条边的差异化保留概率（P_ij = 0.5 + 0.5·exp(-d²/2σ²)），使空间近端的边以高概率保留、空间远端的边以低概率通过。这是一种**去噪纯化（denoising purification）**操作，而非传统意义上的数据增强（加噪破坏）。它在信息流层面——而非表征层面——将空间先验注入表达图的消息传递过程，使得 GCN 的消息传递优先沿解剖学合理的路径进行。

### 创新二：同质近邻保留的软对比优化（Homology-preserving Soft Contrastive Optimization）

标准 InfoNCE 的硬排斥假设所有负样本等权——这在空间转录组中导致了"空间相邻的同类细胞被强制推开"的语义冲突。SpatialGDC 识别一个特殊的 spot 对集合：**同质近邻**——在空间图 A_s 和表达图 A_f 中同时为邻居的 spot 对。这些 spot 对在解剖学上极可能具有功能相似性，其被强制排斥是生物学错误的。SpatialGDC 采用"求同存异"的软惩罚机制：在同质近邻对的 InfoNCE 分母贡献中减去 η 的软惩罚项（η=2.0），使其排斥权重降低约 7.4 倍而**不被彻底消除**。这种温和的轻推（soft nudge）既防止了过强排斥导致的组织撕裂，又保留了弱梯度信号以防止特征空间坍缩为无意义的单点。

---

## 4. Mechanism Chain

```
零膨胀效应（Zero-inflation）
  → 远端伪边污染表达相关图
  → SP-DropEdge 以物理距离为先验进行非对称图拓扑纯化
  → 优先剪除远端伪边、保留近端生物学边
  → 信息流被限制在解剖学合理范围内
  → 缓解过平滑（over-smoothing），保护组织边界

空间相邻的同质近邻对
  → 在空间图和表达图中同时被识别（双图邻域交集）
  → 标准 InfoNCE 会对其进行强制排斥 → 组织连续性被撕裂
  → 软惩罚机制对其施加"求同存异"的轻推
  → 排斥力减弱 7.4 倍但未被消除
  → 保持解剖连续性 + 防止特征空间坍缩
```

---

## 5. Key Modules

### 5.1 双图构建模块（preprocess.py: prepare_graph）
- **功能：** 从空间坐标和 PCA 降维表达矩阵分别构建空间图 A_s 和表达图 A_f
- **解决的问题：** 为 GNN 编码器提供两种互补模态的图拓扑
- **输入：** AnnData 对象（含 spatial 坐标和 X_pca）
- **输出：** 对称归一化的稀疏邻接矩阵（coo_matrix）
- **关系：** 是所有下游编码和剪枝操作的基础

### 5.2 空间先验引导的动态图拓扑优化（network.py: get_dropped_adj + preprocess.py: compute_spatial_keep_prob）
- **功能：** 以物理距离为锚点计算表达图中每条边的差异化保留概率，执行非均匀 Bernoulli 边丢弃
- **解决的问题：** 远端伪关联对局部消息传递的污染
- **输入：** 表达图 A_f + 空间坐标 + 高斯带宽 σ
- **输出：** 剪枝后的表达图 Ã_f（每轮前向传播随机生成）
- **关系：** 仅作用于 V4（表达增强视图），是实现非对称视图构建的核心

### 5.3 共享权重 GCN 编码器（network.py: encoder）
- **功能：** 以共享参数处理四个视图，提取低维表征
- **解决的问题：** 强制不同图拓扑下的表征落入同一潜在空间
- **输入：** PCA 特征 X' (200维) + 四个邻接矩阵
- **输出：** L2 归一化的嵌入 Z_1, Z_2, Z_3, Z_4 ∈ R^(N×24)
- **关系：** 孪生设计（siamese），参数减半，潜在空间一致

### 5.4 两级融合模块（network.py: _fuse_pair + forward）
- **功能：** Level 1 融合同拓扑内原始/增强视图 → Z_spa, Z_expr；Level 2 跨拓扑融合 → Z_fuse
- **解决的问题：** 区分增强扰动方差和模态互补信息
- **输入：** 四个 L2 归一化嵌入
- **输出：** Z_fuse ∈ R^(N×24)
- **关系：** 为对比学习提供干净的跨拓扑对；为重建提供融合特征

### 5.5 同质近邻保留的软对比损失（loss.py: IntersectionContrastiveLoss）
- **功能：** 在实例级 InfoNCE 中识别同质近邻对并施加软惩罚
- **解决的问题：** 标准 InfoNCE 强制排斥同质近邻导致语义冲突
- **输入：** h_spa, h_expr (L2 归一化投影特征) + 原始邻接矩阵 A_s, A_f
- **输出：** 标量损失值
- **关系：** 与 ClusterLoss 和 MSELoss 联合优化

### 5.6 聚类投影与推理模块（network.py: projectClsHead + getCluster）
- **功能：** 用 Softmax 产生软聚类分配用于簇级对比学习；推理时输出硬标签
- **解决的问题：** 簇级结构对齐 + 最终空间域识别
- **输入：** Z_spa, Z_expr（训练）/ Z_fuse（推理）
- **输出：** 软分配概率分布（训练）/ 硬聚类标签（推理，经由 mclust）
- **关系：** 训练时提供簇级对比信号，推理时产出最终域标签

---

## 6. Loss Function Analysis

### 6.1 标准 InfoNCE 在 ST 数据上的缺陷

标准 InfoNCE 定义为 L = -log[exp(sim(x,x⁺)/τ) / Σ_j exp(sim(x,xⱼ)/τ)]，其核心假设是：所有非正样本均为等权负样本，应被同等力度排斥。在自然图像中，该假设成立——不同图像之间通常无相关性。但在 ST 数据中，空间相邻且表达相似的 spot 天然属于同一功能域。强制排斥它们：(i) 违背生物先验；(ii) 在聚类边界处产生碎片化（ARI 下降）；(iii) 抹杀同类 spot 的生物学共性。

### 6.2 软惩罚机制的数学依据

设同质近邻集合 Ω_H = {(i,j) | A_s(i,j)>0 ∧ A_f(i,j)>0}。对标准 InfoNCE 的相似度矩阵 S_ij 进行修正：

S_ij = (h_i·h_j)/τ  →  S'_ij = S_ij - η·I[(i,j)∈Ω_H]

其中 η=2.0，I[·] 为指示函数。修正后，同质近邻对在分母中的贡献减少 exp(-η)=exp(-2)≈0.135 倍（约 7.4 倍减弱），但未被完全消除。该操作等价于在对比目标的"均匀性（uniformity）"约束上施加了一个空间结构化的先验松弛——对解剖学上合理的负样本降低排斥力度，对真正的跨域负样本保持强排斥。

### 6.3 "求同存异"的生物学意义

"求同"——同质近邻对不完全排斥，保留其在表征空间中的适度聚集，维护组织内同类细胞的生物学共性；"存异"——不完全取消排斥，保留弱梯度信号以正则化表征空间，防止所有同质细胞坍缩为单一无意义点。这种温和的轻推机制在组织边界处尤为重要——它允许边界两侧的细胞在保持各自域特性的同时，不因过度排斥而产生虚假的"裂缝"。

---

## 7. Experiment Summary

### DLPFC 人脑前额叶皮层（12 切片，10x Visium）
| 指标 | 数值 | 说明 |
|------|------|------|
| 平均 ARI (网格搜索) | 0.582 | 12 切片最优参数下的均值 |
| 最佳切片 151671 | ARI 0.812 | 层状结构清晰分离 |
| 最差切片 151670 | ARI 0.475 | 解剖层压缩严重 |
| vs 第二名 Spatial-MGCN | +0.01 | 微弱优势，稳定性更优 |
| vs STAGATE（单图） | +0.05 | 显著优于单图方法 |

**规律：** 在层状结构清晰的切片上优势最明显；在解剖层压缩严重的切片上所有方法均困难，但 SpatialGDC 下降幅度最小（跨切片方差最低）。

### BRCA 人乳腺癌（21 类亚型，10x Visium）
| 指标 | 数值 | 说明 |
|------|------|------|
| 最佳 ARI | 0.663 | sigma=0.6, gamma=5.0, kappa=0.05, beta=2.0 |
| 最佳 NMI | 0.695 | |
| vs stGRL | +0.115 | 显著领先 |
| vs Spatial-MGCN | +0.023 | |

**规律：** 在高度异质的肿瘤微环境中，SP-DropEdge 的空间先验注入使其在聚类纯度（ARI）上领先，但在互信息（NMI）上 CCST 更高——表明空间先验更有利于精确的边界划分而非全局信息保留。

### MOSTA 小鼠胚胎（12 器官，Stereo-seq）
| 指标 | 数值 | 说明 |
|------|------|------|
| 最佳 ARI | 0.406 | |
| vs stGRL | +0.078 (+23.8%) | 两级融合在高分辨率数据上增益最大 |

**规律：** 在离散器官（心脏、肝脏）上表现优异，在渐变梯度器官（神经管）上难度较大。

### MOB 小鼠嗅球（Slide-seqV2）
| 指标 | 数值 | 说明 |
|------|------|------|
| 最佳 SC | 0.183 | 子采样 n=5000 |
| vs SEDR | -0.07 | SC 较低，但视觉层恢复准确 |

**规律：** SC 受子采样影响大；视觉检查确认 SpatialGDC 成功还原了嗅球的同心层状结构。

---

## 8. Ablation Insights

| 消融操作 | DLPFC ΔARI | BRCA ΔARI | 机制解释 |
|----------|-----------|-----------|---------|
| 移除 SP-DropEdge（回退为均匀 DropEdge） | **-0.06** | **-0.06** | 失去空间锚点后，远端伪边与近端生物学边被等概率丢弃/保留 → 噪声信号侵入局部邻域 → 组织边界模糊 → 跨数据集一致的大幅退化 |
| 移除软惩罚（回退为标准 InfoNCE） | -0.02 | 0.00 | DLPFC 层状结构中同质近邻对丰富 → 强制排斥撕裂连续性。BRCA 21 类细粒度异质肿瘤中同质近邻对极稀疏 → 软惩罚无作用。该数据集依赖行为本身是有意义的发现 |

**核心结论：** SP-DropEdge 是跨数据集最关键的创新；同质近邻保留的空间邻近容忍在规则组织（层状）中有效，在高度异质组织（肿瘤）中因同质近邻对稀少而自然失效——非 bug，而是特征。

---

## 9. Comparison With Existing Methods

| 维度 | STAGATE / SpaGCN 等单图方法 | Spatial-MGCN / stGRL 等双图方法 | SpatialGDC |
|------|---------------------------|-------------------------------|------------|
| 图构建 | 仅空间图 | 空间图 + 表达图 | 空间图 + 表达图 + **非对称视图构建** |
| 边去噪 | 无 / 均匀 DropEdge | 无 / 均匀 DropEdge | **空间先验引导的动态图拓扑优化**（非均匀，距离感知） |
| 对比学习 | 无 / 自编码器 | 标准 InfoNCE（硬排斥） | **同质近邻保留的软对比优化**（软容忍） |
| 融合策略 | 无 | 扁平融合 | **两级融合**（先同拓扑去扰动，再跨拓扑互补） |
| 核心缺陷 | 仅靠空间邻近 → 过平滑 | 远端伪边污染 + 同质近邻强制排斥 | — |

**SpatialGDC 的独家优势：**
1. 非对称去噪：以物理距离为绝对锚点进行图拓扑纯化，信息流沿解剖学合理路径流动
2. 同质近邻保留：以软容忍替代强制排斥，维护组织连续性
3. 两级融合：区分增强扰动与模态互补两种定性不同的变异源

---

## 10. Important Experimental Evidence

| 图表 | 证明内容 |
|------|---------|
| Fig 2B 箱线图 | DLPFC 12 切片上 SpatialGDC 均值最高且方差最低 → 非对称去噪提供了跨切片的鲁棒性 |
| Fig 3 综合图 (151672) | 8 方法空间图+UMAP+PAGA → SpatialGDC 还原了最清晰的皮层分层，边界整洁 |
| Fig BRCA 空间域 (5×2 组织背景) | SpatialGDC 在肿瘤核心/健康组织/边界区域均与 Ground Truth 高度吻合 |
| 消融表 (双数据集) | SP-DropEdge 移除 → 双数据集一致 -0.06 → 空间先验注入的跨平台普适性 |
| MOSTA ARI 0.406 vs stGRL 0.328 | 两级融合在 Stereo-seq 纳米珠分辨率下增益 +23.8% |
| 标记基因空间分布 (Sox2, Myh7, Afp, Shh) | SpatialGDC 识别域与经典发育标记基因共定位 → 域识别的生物学有效性 |

---

## 11. Method Draft For Paper

SpatialGDC 接受一个包含 N 个捕获位点的 ST 数据集，输入为 PCA 降维至 200 维的基因表达矩阵 X 和二维空间坐标 S。预处理阶段构建两个邻接图：空间图 A_s（欧式距离倒数 KNN, K=12，无阈值过滤）和表达图 A_f（Pearson 相关 KNN, K=12，阈值 0.1）。两图均进行对称归一化处理。

**非对称对比视图构建：** 生成四个视图。V1 和 V2 分别使用原始 A_s 和 A_f，无边丢弃。V3 对 A_s 施加均匀随机 DropEdge（p=0.4）。V4 对 A_f 施加空间先验引导的动态图拓扑优化：每条边 (i,j) 的保留概率 P_ij = 0.5 + 0.5·exp(-d_ij²/2σ²)，其中 d_ij 为归一化欧式距离，σ=0.5。在每轮前向传播中，独立 Bernoulli 采样决定边丢弃，保留边经概率倒数缩放以保持期望。所有视图的输入特征统一施加高斯噪声（α=0.01）和 Dropout（p=0.5）。

**共享权重编码与两级融合：** 一个两层共享权重 GCN（200→64→24, ELU 激活）处理全部四个视图。输出经 L2 归一化后进入两级融合。Level 1（同拓扑融合）：Z_spa = Attention(Z_1, Z_3)，Z_expr = Attention(Z_2, Z_4)，消除各拓扑内的扰动方差。Level 2（跨拓扑融合）：Z_fuse = Attention(Z_spa, Z_expr)，捕获互补信息。注意力机制由两层感知器（含 tanh）和 Softmax 归一化组成。

**同质近邻保留的软对比优化：** 通过投影 MLP 从 Z_spa 和 Z_expr 产生 L2 归一化特征 h_spa 和 h_expr。识别同质近邻集 Ω_H = {(i,j) | A_s(i,j)>0 ∧ A_f(i,j)>0}。修改 InfoNCE 相似度：对 (i,j)∈Ω_H，从相似度中减去 η=2.0，使排斥力降低 ~7.4 倍。簇级对比损失作用于通过 Softmax 产生的软聚类分配上，辅以负熵正则化防坍缩。重建损失通过转置 GCN 权重将 Z_fuse 解码回 200 维 PCA 空间。

总损失 L_total = κ·L_ICL + β·L_CCL + γ·L_rec 经 Adam 优化（lr=0.001, wd=0.001, cosine 退火，500 epochs）。推理时，Z_fuse 经 mclust（高斯混合模型）聚类，可选空间邻域标签精炼。

---

## 12. Introduction Material

**背景：** 空间转录组学能够在保留组织空间架构的前提下进行全基因组表达谱分析，空间域识别——将组织划分为空间连贯的功能区域——是该领域最基础且最关键的下游分析任务。

**零膨胀驱动的过平滑痛点：** ST 数据受限于测序深度，表现出极端的基因表达稀疏性和 dropout 效应。两个空间上相距极远的细胞仅因共享大量技术性零值即在特征空间中呈现虚假高相关。传统 GNN 缺乏物理距离约束，在此类伪边上盲目消息传递，引发特征过平滑，模糊组织边界。

**强制排斥带来的语义冲突：** 标准 InfoNCE 将所有非锚点样本等权排斥。在层状组织中，空间相邻的同类细胞被强制推开——这违背了生物组织的局部连续性，导致特征空间碎裂和聚类边界碎片化。

**Motivation：** 现有方法未能同时解决"远端伪边的拓扑污染"和"同质近邻的语义冲突"这两个耦合问题。SpatialGDC 通过非对称对比视图构建引入物理空间作为绝对锚点进行图拓扑纯化，同时以同质近邻保留的软约束替代暴力的强制排斥，从而在两个维度上同时突破性能瓶颈。

---

## 13. Potential Paper Claims

- Asymmetric contrastive view construction
- Spatial-prior guided dynamic graph topology optimization
- Zero-inflation-driven spurious distal correlations
- Topology contamination induced by spatial insensitivity
- Semantic collision of homologous neighborhoods
- Homology-preserving soft contrastive optimization
- Soft penalty as a "seeking common ground while preserving differences" mechanism
- Denoising purification (not data augmentation)
- Physical distance as an absolute anchor for graph topology refinement
- Over-smoothing prevention via distance-modulated information flow
- Two-level fusion disentangling perturbation variance from modality complementarity

---

## 14. Important Files

| 文件 | 职责 |
|------|------|
| `spCLUE/spCLUE.py` | 训练循环、模型初始化、超参数管理、早停与推理 |
| `spCLUE/network.py` | CCGCN 模型类：编码器、SP-DropEdge（get_dropped_adj）、两级融合、投影头 |
| `spCLUE/loss.py` | IntersectionContrastiveLoss（同质近邻软惩罚）、ClusterLoss、MSELoss |
| `spCLUE/preprocess.py` | 数据加载与预处理、双图构建、spatial keep_prob 计算 |
| `spCLUE/utils.py` | 稀疏矩阵转换、学习率余弦退火调度、随机种子固定 |
| `paper/spatialGDC.tex` | 完整 LaTeX 论文（IEEEtran 模板, ~375 行） |
| `paper/refs.bib` | BibTeX 参考文献（431 行） |
| `SpatialGDC_DLPFC_GridSearch_Summary.csv` | DLPFC 12 切片网格搜索结果 |
| `SpatialGDC_DLPFC_Ablation_v2.csv` | DLPFC 消融实验（3 变体 × 12 切片 × 5 种子） |
| `SpatialGDC_BRCA_GridSearch_v2.csv` | BRCA 增强网格搜索结果 |
| `SpatialGDC_BRCA_Ablation_v2.csv` | BRCA 消融实验（3 变体 × 5 种子） |
| `SpatialGDC_MOSTA_GridSearch_v2.csv` | MOSTA 网格搜索结果 |
| `SpatialGDC_MOB_SC_v2.txt` | MOB Silhouette Coefficient 搜索结果 |

---

## 15. Final Research Essence

SpatialGDC 在空间转录组学领域的本质创新哲学可以凝练为两条原则：

**原则一：打破对称加噪范式，引入绝对物理锚点。** 传统对比学习对所有视图施加同质扰动——这是对自然图像数据分布的无差别假设的延续。SpatialGDC 认识到在 ST 数据中，物理距离是唯一可用于区分"生物学信号"和"技术噪声"的客观参照系。通过在表达图上引入非对称的、以物理距离为锚点的差异化图拓扑优化，模型的信息流被从"盲目全局传播"重塑为"解剖学约束的局部流动"——这不是数据增强（加噪），而是去噪纯化。

**原则二：以温和的同质感知识别取代暴力的强制排斥。** 标准对比学习假设"非我即敌"，这在一个所有细胞天然嵌入连续组织空间的世界里是生物学错误的。SpatialGDC 识别出那些"不应被视为敌人"的同质近邻对，并以温和的软约束（轻推而非推开）替代暴力的强制排斥。这种"求同存异"的哲学在组织边界处尤为关键——它允许细胞在保持各自域身份的同时，不因过度排斥而产生虚假的解剖裂缝。
