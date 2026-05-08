import pandas as pd
import scanpy as sc
import os

base_dir = "/home/bio/lhz/spatialGDC/dataset/Mouse_OB/"
sub_dir = os.path.join(base_dir, "Dataset1_LiuLongQi_MouseOlfactoryBulb")

print("📂 正在读取表达矩阵与坐标...")
counts = pd.read_csv(os.path.join(sub_dir, "RNA_counts.tsv.gz"), sep="\t", index_col=0)
counts.columns = counts.columns.astype(str)

pos = pd.read_csv(os.path.join(sub_dir, "position.tsv"), sep="\t")
pos['label'] = pos['label'].astype(int).astype(str)
pos = pos.set_index('label')

# 🌟 关键修改：直接用表达矩阵和坐标取交集，丢弃外层不匹配的 used_barcodes.txt
common_ids = list(set(counts.columns) & set(pos.index))
print(f"📊 两文件完美匹配的细胞数: {len(common_ids)}")

if len(common_ids) == 0:
    print("❌ 错误：如果还是 0，说明数据本身格式有异。")
    print(f"矩阵前5个ID: {counts.columns[:5].tolist()}")
    print(f"坐标前5个ID: {pos.index[:5].tolist()}")
    exit()

counts = counts[common_ids]
pos = pos.loc[common_ids]

print("🧬 正在构建 AnnData 对象...")
adata = sc.AnnData(counts.T)
adata.obsm['spatial'] = pos[['x', 'y']].values

sc.pp.filter_cells(adata, min_genes=1)
save_path = os.path.join(base_dir, "Mouse_Olfactory_Bulb.h5ad")
adata.write_h5ad(save_path)

print(f"🎉 任务完成！最终包含有效细胞数据的文件已生成：{save_path}")
print(f"📐 最终有效数据规模: {adata.n_obs} 个细胞 (Spots), {adata.n_vars} 个基因")