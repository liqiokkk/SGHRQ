import torch
import torch.nn as nn
from torch.nn import functional as F

from .SGHRQ_utils import SpatialEncoding, sghrq_feat
from torch.nn.parameter import Parameter

def get_syncbn():
    #return nn.BatchNorm2d
    return nn.SyncBatchNorm


class SGHRQ_MLP(nn.Module):
    def __init__(self, ultra_pe=False, pos_dim=40, sync_bn=False, num_classes=19, local=False, unfold=False, stride=1, learn_pe=False, require_grad=False, num_layer=2):

        super(SGHRQ_MLP, self).__init__()
        self.pos_dim = pos_dim
        self.ultra_pe = ultra_pe
        self.local = local
        self.unfold = unfold
        self.stride = stride
        self.learn_pe = learn_pe
        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm1d
        if ultra_pe:
            self.pos1 = SpatialEncoding(2, self.pos_dim, require_grad=require_grad)
            self.pos2 = SpatialEncoding(2, self.pos_dim, require_grad=require_grad)
            self.pos3 = SpatialEncoding(2, self.pos_dim, require_grad=require_grad)
            self.pos4 = SpatialEncoding(2, self.pos_dim, require_grad=require_grad)
            self.pos_dim += 2
        
        in_dim = (256 + self.pos_dim) + 2*(num_classes + self.pos_dim)
       
        if num_layer == 2:
            self.imnet = nn.Sequential(
            nn.Conv1d(in_dim, 512, 1), norm_layer(512), nn.ReLU(), 
            nn.Conv1d(512, 256, 1), norm_layer(256), nn.ReLU(),
            nn.Conv1d(256, 256, 1), norm_layer(256), nn.ReLU(),
            nn.Conv1d(256, num_classes, 1)
            )                 

    def forward(self, x, size, level=0, after_cat=False):
        h, w = size
        if not after_cat:
            if not self.local:
                rel_coord, q_feat = sghrq_feat(x, [h, w])
                if self.ultra_pe or self.learn_pe:
                    rel_coord = eval('self.pos'+str(level))(rel_coord)
                x = torch.cat([rel_coord, q_feat], dim=-1)
            else:
                pass             
        else:
            x = self.imnet(x).view(x.shape[0], -1, h, w)
        return x

        
