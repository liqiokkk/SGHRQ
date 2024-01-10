import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import copy

from mmcv.cnn import ConvModule


class Memory(nn.Module):
    def __init__(self, num_classes, feats_channels, out_channels, num_feats_per_cls=1, conv_cfg=None, norm_cfg=None, act_cfg=None):
        super(Memory, self).__init__()
        # set attributes
        self.num_classes = num_classes
        self.feats_channels = feats_channels
        self.out_channels = out_channels
        self.num_feats_per_cls = num_feats_per_cls
        
        # init memory
        self.memory = nn.Parameter(torch.zeros(num_classes, num_feats_per_cls, feats_channels, dtype=torch.float), requires_grad=False)
        
        self.S = nn.Sequential(nn.Conv1d(feats_channels, out_channels, 1), nn.BatchNorm1d(out_channels), nn.ReLU())
        
    '''forward'''
    def forward(self, feats):
        batch_size, num_channels, h, w = feats.size()

        feats_m = feats
        feats_m = feats_m.view(batch_size, num_channels, -1)
        feats_m = self.S(feats_m)
        feats_m = feats_m.permute(0, 2, 1) 

        selected_memory = self.memory[:, 0, :]
        selected_memory = selected_memory.unsqueeze(0)  
        selected_memory = selected_memory.expand(batch_size, -1, -1) 
        selected_memory = selected_memory.permute(0, 2, 1)  
        
        sim_map = torch.matmul(feats_m, selected_memory)
        sim_map = (self.feats_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1) 

        memory_output = sim_map.permute(0, 2, 1)
        memory_output = memory_output.reshape(batch_size, self.num_classes, h, w)
        
        return memory_output
    '''update'''
    def update(self, features, segmentation, ignore_index=255, strategy='cosine_similarity', momentum_cfg={
                'base_momentum': 0.1}, learning_rate=None):
        assert strategy in ['mean', 'cosine_similarity']
        batch_size, num_channels, h, w = features.size()
        momentum = momentum_cfg['base_momentum']
        # use features to update memory
        segmentation = segmentation.long()
        features = features.permute(0, 2, 3, 1).contiguous()
        features = features.view(batch_size * h * w, num_channels)
        clsids = segmentation.unique()
        for clsid in clsids:
            if clsid == ignore_index: continue
            seg_cls = segmentation.view(-1)
            feats_cls = features[seg_cls == clsid]
            need_update = True
            for idx in range(self.num_feats_per_cls):
                if (self.memory[clsid][idx] == 0).sum() == self.feats_channels:
                    self.memory[clsid][idx].data.copy_(feats_cls.mean(0))
                    need_update = False
                    break
            if not need_update: continue
            if self.num_feats_per_cls == 1:
                if strategy == 'mean':
                    feats_cls = feats_cls.mean(0)
                elif strategy == 'cosine_similarity':
                    similarity = F.cosine_similarity(feats_cls, self.memory[clsid].data.expand_as(feats_cls))
                    weight = (1 - similarity) / (1 - similarity).sum()
                    feats_cls = (feats_cls * weight.unsqueeze(-1)).sum(0)
                feats_cls = (1 - momentum) * self.memory[clsid].data + momentum * feats_cls.unsqueeze(0)
                self.memory[clsid].data.copy_(feats_cls)
            else:
                pass
        # syn the memory
        if dist.is_available() and dist.is_initialized():
            memory = self.memory.data.clone()
            dist.all_reduce(memory.div_(dist.get_world_size()))
            self.memory = nn.Parameter(memory, requires_grad=False)