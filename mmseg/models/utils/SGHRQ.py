import torch
import torch.nn as nn
from torch.nn import functional as F
from .SGHRQ_mlp import SGHRQ_MLP
import math
import numpy as np

def get_syncbn():
    #return nn.BatchNorm2d
    return nn.SyncBatchNorm


class SGHRQ(nn.Module):

    def __init__(self, in_planes, num_classes=19, inner_planes=256, sync_bn=False, dilations=(12, 24, 36), 
                pos_dim=24, ultra_pe=False, unfold=False, no_aspp=False,
                local=False, stride=1, learn_pe=False, require_grad=False, num_layer=2):

        super(SGHRQ, self).__init__()
        norm_layer = get_syncbn() if sync_bn else nn.BatchNorm2d

        self.head = nn.Sequential(nn.Conv2d(640, 256, kernel_size=1), norm_layer(256), nn.ReLU(inplace=True))  
        
        self.sghrq_mlp = SGHRQ_MLP(ultra_pe=ultra_pe, pos_dim=pos_dim, sync_bn=sync_bn, num_classes=num_classes, local=local, unfold=unfold, stride=stride, learn_pe=learn_pe, require_grad=require_grad, num_layer=num_layer)

    def forward(self, x):
        deep_feat, shallow_feat16, shallow_feat8 = x
        
        aspp_out = self.head(deep_feat)
        x1 = shallow_feat8
        x2 = shallow_feat16

        context = []
        h, w = x1.shape[-2], x1.shape[-1]

        target_feat = [x1, x2, aspp_out]

        for i, feat in enumerate(target_feat):
            context.append(self.sghrq_mlp(feat, size=[h, w], level=i+1))
        context = torch.cat(context, dim=-1).permute(0,2,1)

        res = self.sghrq_mlp(context, size=[h, w], after_cat=True)

        return res
