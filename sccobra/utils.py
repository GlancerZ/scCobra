import torch
from torch.utils.data import DataLoader, TensorDataset

def make_dataloader(adata, batch_size):
    feature = torch.tensor(adata.X)
    label = adata.obs['label']
    
    dataset = TensorDataset(feature, torch.tensor(label))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    
    return dataloader
    