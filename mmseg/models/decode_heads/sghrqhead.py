import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.cnn import ConvModule
from ..builder import HEADS
from .cascade_decode_head import BaseCascadeDecodeHead
from .stdc_head import ShallowNet
from ..utils.SGHRQ import SGHRQ
from ..utils.memory import Memory

class SegmentationHead(nn.Module):
    def __init__(self, conv_cfg, norm_cfg, act_cfg, in_channels, mid_channels, n_classes, *args, **kwargs):
        super(SegmentationHead, self).__init__()

        self.conv_bn_relu = ConvModule(in_channels, mid_channels, 3,
                                       stride=1,
                                       padding=1,
                                       conv_cfg=conv_cfg,
                                       norm_cfg=norm_cfg,
                                       act_cfg=act_cfg)

        self.conv_out = nn.Conv2d(mid_channels, n_classes, kernel_size=1, bias=True)

    def forward(self, x):
        x = self.conv_bn_relu(x)
        x = self.conv_out(x)
        return x


class Lap_Pyramid_Conv(nn.Module):
    def __init__(self, num_high=3, gauss_chl=3):
        super(Lap_Pyramid_Conv, self).__init__()

        self.num_high = num_high
        self.gauss_chl = gauss_chl
        self.kernel = self.gauss_kernel()

    def gauss_kernel(self, device=torch.device('cuda')):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(self.gauss_chl, 1, 1, 1)
        kernel = kernel.to(device)
        return kernel

    def downsample(self, x):
        return x[:, :, ::2, ::2]

    def upsample(self, x):
        cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
        cc = cc.permute(0, 1, 3, 2)
        cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
        cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
        x_up = cc.permute(0, 1, 3, 2)
        return self.conv_gauss(x_up, 4 * self.kernel)

    def conv_gauss(self, img, kernel):
        img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
        out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
        return out

    def pyramid_decom(self, img):
        current = img
        pyr = []
        for _ in range(self.num_high):
            filtered = self.conv_gauss(current, self.kernel)
            down = self.downsample(filtered)
            up = self.upsample(down)
            if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
                up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]))
            diff = current - up
            pyr.append(diff)
            current = down
        return pyr


class SRDecoder(nn.Module):
    # super resolution decoder
    def __init__(self, conv_cfg, norm_cfg, act_cfg, channels=128, up_lists=[2, 2, 2]):
        super(SRDecoder, self).__init__()
        self.conv1 = ConvModule(channels, channels // 2, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up1 = nn.Upsample(scale_factor=up_lists[0])
        self.conv2 = ConvModule(channels // 2, channels // 2, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up2 = nn.Upsample(scale_factor=up_lists[1])
        self.conv3 = ConvModule(channels // 2, channels, 3, stride=1, padding=1, conv_cfg=conv_cfg,
                                norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.up3 = nn.Upsample(scale_factor=up_lists[2])
        self.conv_sr = SegmentationHead(conv_cfg, norm_cfg, act_cfg, channels, channels // 2, 3, kernel_size=1)

    def forward(self, x, fa=False):
        x = self.up1(x)
        x = self.conv1(x)
        x = self.up2(x)
        x = self.conv2(x)
        x = self.up3(x)
        feats = self.conv3(x)
        outs = self.conv_sr(feats)
        if fa:
            return feats, outs
        else:
            return outs
        

class Reducer(nn.Module):
    # Reduce channel (typically to 128)
    def __init__(self, in_channels=512, reduce=128, bn_relu=True):
        super(Reducer, self).__init__()
        self.bn_relu = bn_relu
        self.conv1 = nn.Conv2d(in_channels, reduce, 1, bias=False)
        if self.bn_relu:
            self.bn1 = nn.BatchNorm2d(reduce)

    def forward(self, x):
        x = self.conv1(x)
        if self.bn_relu:
            x = self.bn1(x)
            x = F.relu(x)

        return x


@HEADS.register_module()
class SGHRQHead(BaseCascadeDecodeHead):
    def __init__(self, down_ratio, prev_channels, bn_type, reduce=False, **kwargs):
        super(SGHRQHead, self).__init__(**kwargs)
        self.sync_bn = bn_type == 'SyncBN'
        self.down_ratio = down_ratio
        self.sr_decoder = SRDecoder(self.conv_cfg, self.norm_cfg, self.act_cfg,
                                    channels=self.channels, up_lists=[4, 2, 2])
        # shallow branch
        self.stdc_net = ShallowNet(in_channels=9, pretrain_model="STDCNet813M_73.91.tar", sync_bn=self.sync_bn)
        self.lap_prymaid_conv = Lap_Pyramid_Conv(num_high=2)
        self.conv_seg_aux_16 = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, 512,
                                                self.channels // 2, self.num_classes, kernel_size=1)
        self.conv_seg_aux_8 = SegmentationHead(self.conv_cfg, self.norm_cfg, self.act_cfg, 256,
                                               self.channels // 2, self.num_classes, kernel_size=1)
        self.sghrq = SGHRQ(in_planes=None, num_classes=self.num_classes, ultra_pe=True, no_aspp=True, require_grad=True, sync_bn=self.sync_bn)

        self.reduce = Reducer() if reduce else None
        
        self.memory = Memory(self.num_classes, 512, 512, 1, self.conv_cfg, self.norm_cfg, self.act_cfg)

    def forward(self, inputs, prev_output, train_flag=True, gt=None):
        """Forward function."""
        prymaid_results = self.lap_prymaid_conv.pyramid_decom(inputs)
        high_residual_1 = prymaid_results[0]
        high_residual_2 = F.interpolate(prymaid_results[1], prymaid_results[0].size()[2:], mode='bilinear',
                                        align_corners=False)
        high_residual_input = torch.cat([inputs, high_residual_1, high_residual_2], dim=1)
        shallow_feat8, shallow_feat16 = self.stdc_net(high_residual_input)
        deep_feat = prev_output[0]
        if self.reduce is not None:
            deep_feat = self.reduce(deep_feat)
        if train_flag:
            img_size = prev_output[-1].size()[2:]
            with torch.no_grad():
                self.memory.update(
                features=prev_output[-1], 
                segmentation=F.interpolate(gt.float(), size=img_size, mode="nearest")
                )
            output_aux16 = self.conv_seg_aux_16(shallow_feat16)
            output_aux16_m = self.memory(shallow_feat16) * output_aux16 + output_aux16
            output_aux8 = self.conv_seg_aux_8(shallow_feat8)
            all_feats = [prev_output[2], output_aux16_m, output_aux8]
            output = self.sghrq(all_feats)
            
            feats, output_sr = self.sr_decoder(deep_feat, True)
            losses_re = self.image_recon_loss(high_residual_1 + high_residual_2, output_sr, re_weight=0.1)
            losses_fa = self.feature_affinity_loss(deep_feat, feats)
            return output, output_aux16, output_aux8, losses_re, losses_fa, output_aux16_m
        else:
            output_aux16 = self.conv_seg_aux_16(shallow_feat16)
            output_aux16_m = self.memory(shallow_feat16) * output_aux16 + output_aux16
            output_aux8 = self.conv_seg_aux_8(shallow_feat8)
            all_feats = [prev_output[2], output_aux16_m, output_aux8]
            output = self.sghrq(all_feats)
            return output

    def image_recon_loss(self, img, pred, re_weight=0.5):
        loss = dict()
        if pred.size()[2:] != img.size()[2:]:
            pred = F.interpolate(pred, img.size()[2:], mode='bilinear', align_corners=False)
        recon_loss = F.mse_loss(pred, img) * re_weight
        loss['recon_losses'] = recon_loss
        return loss

    def feature_affinity_loss(self, seg_feats, sr_feats, fa_weight=1., eps=1e-6):
        if seg_feats.size()[2:] != sr_feats.size()[2:]:
            sr_feats = F.interpolate(sr_feats, seg_feats.size()[2:], mode='bilinear', align_corners=False)
        loss = dict()
        # flatten:
        seg_feats_flatten = torch.flatten(seg_feats, start_dim=2)
        sr_feats_flatten = torch.flatten(sr_feats, start_dim=2)
        # L2 norm 
        seg_norm = torch.norm(seg_feats_flatten, p=2, dim=2, keepdim=True)
        sr_norm = torch.norm(sr_feats_flatten, p=2, dim=2, keepdim=True)
        # similiarity
        seg_feats_flatten = seg_feats_flatten / (seg_norm + eps)
        sr_feats_flatten = sr_feats_flatten / (sr_norm + eps)
        seg_sim = torch.matmul(seg_feats_flatten.permute(0, 2, 1), seg_feats_flatten)
        sr_sim = torch.matmul(sr_feats_flatten.permute(0, 2, 1), sr_feats_flatten)
        # L1 loss
        loss['fa_loss'] = F.l1_loss(seg_sim, sr_sim.detach()) * fa_weight
        return loss

    def forward_train(self, inputs, prev_output, img_metas, gt_semantic_seg, train_cfg):
        seg_logits, seg_logits_aux16, seg_logits_aux8, losses_recon, losses_fa, seg_logits_aux16_m = self.forward(inputs, prev_output,gt=gt_semantic_seg)
        losses = self.losses(seg_logits, gt_semantic_seg)
        losses_aux16 = self.losses(seg_logits_aux16, gt_semantic_seg)
        losses_aux8 = self.losses(seg_logits_aux8, gt_semantic_seg)
        losses_aux16_m = self.losses(seg_logits_aux16_m, gt_semantic_seg)
        return losses, losses_aux16, losses_aux8, losses_recon, losses_fa, losses_aux16_m

    def forward_test(self, inputs, prev_output, img_metas, test_cfg):
        return self.forward(inputs, prev_output, False)
