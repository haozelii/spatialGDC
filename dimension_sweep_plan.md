# SpatialGDC 维度探索计划 — 2026-05-21

## 当前基线
- **架构**: 4-view 2orig+2aug, Attention fusion, dims=[200,64,24]
- **ICL**: IntersectionContrastiveLoss τ=0.2, fn_penalty=2.0
- **CL pairs**: 4对 (κ_cross=0.03 ×2 + κ_same=0.02 ×2 = 0.10)
- **Recon**: MSE(z_fuse → PCA), 2-step relu(z@W2.T)@W1.T ✓
- **LR**: cosine decay (lr→lr×0.001 over 500 epochs)
- **当前最佳**: R5(全DropEdge) ARI=0.5516, 2orig2aug-v2 ARI=0.5121

## 已探索过且失败的维度（不再重试）
- ✗ Fusion变体: CTCF(0.4581), ConcatProj, PerDimGate — bilinear不适合d=24
- ✗ 更多CL对(R7): V2↔V4 CL, 边际改善(0.558 vs 0.552)
- ✗ Multi-scale图(MS1/MS2): 0.522, 0.526
- ✗ Plan A (2×2 factorial): 0.5065
- ✗ FeatureMask替代NoiseLayer: −0.05 ARI
- ✗ Hierarchical fusion(R3): 0.521
- ✗ 不同增强类型叠加

## 本次4个实验 — 全是训练/正则化维度，不改架构

### E1: EMA (指数移动平均) — 稳定性维度
- **假设**: DLPFC 12个slice各自独立训练(3000-5000 spots/slice), 训练噪声大。EMA平滑参数更新，提升泛化。
- **修改**: train loop中维护shadow网络 (β=0.999), evaluation用EMA权重
- **预期**: +0.01-0.03 ARI
- **文献**: Semi-supervised GNNs常用(如GraphMAE, BGRL), 小批量训练标配

### E2: 温度调优 — 对比信号维度
- **假设**: τ=0.2可能导致对比过于激进(硬推远离邻域点)。4-view有更多正负信号，软对比(τ↑)可能更好。
- **修改**: 测试 τ ∈ {0.1, 0.2, 0.3, 0.5, 0.7}
- **预期**: τ=0.5附近可能最优(+0.01-0.02)
- **文献**: SimCLR发现τ需要与view数量协调整; 更多view→更软对比

### E3: Denoising Reconstruction — 表示鲁棒性维度
- **假设**: 在z_fuse上添加小噪声再重构，迫使编码器学习更鲁棒的表示
- **修改**: x_rec = decoder(z_fuse + σ·ε), σ=0.05
- **预期**: +0.01-0.02 ARI (自编码器经典正则化)
- **文献**: Denoising AE(Vincent 2008), MAE(He 2022)

### E4: α-CL混合 — 损失平衡维度
- **假设**: 当前ICL完全排斥交集邻域的假负样本(fn_penalty=2.0)。改为软混合：既做ICL又做标准InfoNCE，α控制混合比。
- **修改**: loss = α·ICL + (1-α)·standard_InfoNCE, α∈{0.0, 0.25, 0.5, 0.75, 1.0}
- **预期**: α=0.5可能最优(既利用拓扑信息又不过度排斥)
- **文献**: 纯InfoNCE在某些图上优于交叉对比; ICL的fn_penalty可能过于激进

## 实验设置
- DLPFC 12片, 每片单次运行(seed=0)
- 使用与fusion_ab_test相同的best_params
- 500 epochs, cosine LR
- 单GPU运行, 预计总耗时 ~8-12小时
