"""Test Full(fn=8) vs Full(fn=2) on all 12 DLPFC slices with optimal params"""
import os,sys,warnings,gc,pandas as pd,numpy as np,scanpy as sc,torch
from sklearn.metrics import adjusted_rand_score
os.environ['R_HOME']='/home/bio/miniconda3/envs/spCLUE/lib/R'
os.environ['OMP_NUM_THREADS']='8'
warnings.filterwarnings('ignore')
sys.path.insert(0,'/home/bio/lhz/spatialGDC')
import spCLUE

df=pd.read_csv('SpatialGDC_DLPFC_GridSearch_Summary.csv')
samples=['151507','151508','151509','151510','151669','151670','151671','151672','151673','151674','151675','151676']

for sample in samples:
    bp=df[df['Sample']==int(sample)].iloc[0]
    dp=f'./dataset/DLPFC/{sample}/'
    ar=spCLUE.load_and_preprocess_st(data_path=dp)
    nc=5 if sample in ['151669','151670','151671','151672'] else 7
    gs=spCLUE.prepare_graph(ar,'spatial');ge=spCLUE.prepare_graph(ar,'expr')
    gd={'spatial':gs,'expr':ge}
    sc_=ar.obsm['spatial'].copy();sc_=(sc_-sc_.min(axis=0))/(sc_.max(axis=0)-sc_.min(axis=0))
    
    for fn in [2.0,8.0]:
        al=[]
        for seed in [0,42,123,999,2024]:
            gc.collect()
            spCLUE.fix_seed(seed)
            a=ar.copy()
            ekp=spCLUE.compute_spatial_keep_prob(ge,sc_,sigma=bp['sigma'])
            m=spCLUE.spCLUE(input_data=a.obsm['X_pca'].copy(),graph_dict=gd,n_clusters=nc,expr_keep_prob=ekp,gamma=bp['gamma'],kappa=bp['kappa'],fn_penalty=fn)
            _,emb,_=m.train()
            a.obsm['emb']=emb
            spCLUE.clustering(a,nc,key='emb',refinement=True,cluster_methods='mclust')
            cc='mclust_refined' if 'mclust_refined' in a.obs.columns else 'mclust'
            ae=a[a.obs.Region.notna()]
            al.append(adjusted_rand_score(ae.obs['Region'],ae.obs[cc]))
        print(f'{sample} fn={fn:.0f}: {np.mean(al):.4f}')
    print()
