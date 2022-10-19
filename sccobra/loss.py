import math
import torch
import torch.nn as nn

class InstanceLoss(nn.Module):
    def __init__(self, temperature):
        super(InstanceLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, z_i, z_j):
        
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        mask = self.mask_correlated_samples(batch_size)
        
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        z = torch.cat((z_i, z_j), dim=0)
        
        sim = torch.matmul(z, z.T) / self.temperature
       
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) 
        negative_samples = sim[mask].reshape(N, -1)
        
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)

        return loss
    
class ClusterLoss(nn.Module):
    def __init__(self, temperature):
        super(ClusterLoss, self).__init__()
        self.temperature = temperature

        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, c_i, c_j):
        
        p_i = c_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = c_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        ne_loss = ne_i + ne_j
        
        class_num = c_i.shape[1]
        N = 2 * class_num
        mask = self.mask_correlated_clusters(class_num)
        
        c_i = nn.functional.normalize(c_i.t(), dim=1)
        c_j = nn.functional.normalize(c_j.t(), dim=1)
        
        c = torch.cat((c_i, c_j), dim=0)

        sim = torch.matmul(c, c.T) / self.temperature
        
        sim_i_j = torch.diag(sim, class_num)
        sim_j_i = torch.diag(sim, class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)

        return loss + ne_loss
    
class Margin_InstanceLoss(nn.Module):
    def __init__(self, temperature, m):
        super(Margin_InstanceLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")
        self.m = m

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask
    
    def add_margin(self, batch_size, m):
        N = 2 * batch_size
        margin = torch.zeros((N, N))
        for i in range(batch_size):
            margin[i, batch_size + i] = m
            margin[batch_size + i, i] = m
        return margin
        
    def forward(self, z_i, z_j):
        
        batch_size = z_i.shape[0]
        N = 2 * batch_size
        mask = self.mask_correlated_samples(batch_size)
        
        z_i = nn.functional.normalize(z_i, dim=1)
        z_j = nn.functional.normalize(z_j, dim=1)
        z = torch.cat((z_i, z_j), dim=0)
        
        cos_sim = torch.matmul(z, z.T)
        cos_sim = torch.clip(cos_sim, -1+1e-6, 1-1e-6)
        #cos_sim = cos_sim.fill_diagonal_(0)#使对角线为0，是为了限制输出，否则arccos(1)会出现NaN
        
        arc_cos_sim = torch.acos(cos_sim) * 180 / torch.pi #转换成角度
        
        margin = self.add_margin(batch_size, self.m)
        
        sim = torch.cos((arc_cos_sim + margin) * torch.pi / 180) / self.temperature
       
        sim_i_j = torch.diag(sim, batch_size)
        sim_j_i = torch.diag(sim, -batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1) 
        negative_samples = sim[mask].reshape(N, -1)
        
        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)

        return loss