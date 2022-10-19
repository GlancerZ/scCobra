import os
os.environ['R_HOME'] = "/dssg/home/acct-clsdqw/clsdqw-user1/.conda/envs/scCobra/lib/R"

import scib
import scanpy as sc

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")


label_encoder = LabelEncoder()

def preprocess_adata(adata, batch_key='Batch', normalize=True, log1p=True, scale=True, hvg=None):
    
    if batch_key != 'Batch':
        adata.obs['Batch'] = adata.obs[batch_key].values
    
    if normalize:
        sc.pp.normalize_total(adata, target_sum=1e4)
    
    if log1p:
        sc.pp.log1p(adata)
    
    if hvg != None:
        adata = scib.preprocessing.hvg_batch(adata, batch_key=batch_key, target_genes=hvg, flavor='cell_ranger', n_bins=20, adataOut=True)
    
    if scale:
        adata = scib.preprocessing.scale_batch(adata, batch='Batch')
        
    label = label_encoder.fit_transform(adata.obs['Batch'])
    
    adata.obs['label'] = label
    
    adata.obs['domain_number'] = len(set(adata.obs.Batch))
        
    print('--------------------------')
        
    print('Preprocessing is finished')
        
    return adata