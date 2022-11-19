import torch
import torch.nn as nn
import torch.nn.functional as F

class feature_encoder(nn.Module):
    def __init__(self, input_dim, domin):
        super(feature_encoder, self).__init__()
        
        self.domin = domin
        self.droplayer = nn.Dropout(0.7)
        
        self.embnet = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512,256),
            nn.GELU(),
        )
        
        self.bn_list = nn.ModuleList()
        
        for i in range(domin):
            self.bn_list.append(nn.BatchNorm1d(256, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True))
        
        self.projector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        
    def forward(self, x, y):
        
        z = self.embnet(x)
        
        x1 = self.droplayer(x)
        x2 = self.droplayer(x)
        
        z1 = self.embnet(x1)
        z2 = self.embnet(x2)
        
        z1_ = []
        z2_ = []
        
        for i in range(self.domin):
            locals()['index_' + str(i)] = torch.argwhere(y==i).squeeze()
            locals()['z1_' + str(i)] = z1[locals()['index_' + str(i)]]
            if locals()['z1_' + str(i)].dim() == 2:
                z1_.append(self.bn_list[i](locals()['z1_' + str(i)]))
            
            
            locals()['z2_' + str(i)] = z2[locals()['index_' + str(i)]]
            if locals()['z2_' + str(i)].dim() == 2:
                z2_.append(self.bn_list[i](locals()['z2_' + str(i)]))
            
        z1 = torch.cat(z1_,0)
        z2 = torch.cat(z2_,0)
        
        h1 = self.projector(z1)
        
        h2 = self.projector(z2)
        
        return z, h1, h2
    
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction=='sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction=='mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)

class discriminator(nn.Module):
    def __init__(self, input_dim, domin_number):
        super(discriminator, self).__init__()
        
        self.loss = LabelSmoothingCrossEntropy()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.Linear(16, domin_number)
        )
    
    def forward(self, x, label):
        
        output = self.net(x)
        
        loss = self.loss(output, label.long())
        
        return loss
