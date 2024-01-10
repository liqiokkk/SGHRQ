import torch
import time
from torch import nn
from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
from mmcv.runner import auto_fp16

@SEGMENTORS.register_module()
class EncoderDecoderRefine(EncoderDecoder):
    """Cascade Encoder Decoder segmentors.

    CascadeEncoderDecoder almost the same as EncoderDecoder, while decoders of
    CascadeEncoderDecoder are cascaded. The output of previous decoder_head
    will be the input of next decoder_head.
    """

    def __init__(self,
                 num_stages,
                 down_ratio,
                 backbone,
                 decode_head,
                 refine_input_ratio=1.,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 is_frequency=False,
                 pretrained=None,
                 init_cfg=None):
        self.num_stages = num_stages
        self.is_frequency = is_frequency
        self.down_scale = down_ratio
        self.refine_input_ratio = refine_input_ratio
        super(EncoderDecoderRefine, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        assert isinstance(decode_head, list)
        assert len(decode_head) == self.num_stages
        self.decode_head = builder.build_head(decode_head[0])
        self.refine_head = builder.build_head(decode_head[1])
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        img_os2 = nn.functional.interpolate(img, size=[img.shape[-2]//self.down_scale, img.shape[-1]//self.down_scale])
        x = self.extract_feat(img_os2)
        img_refine = img
        
        out_g, prev_outputs = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        out = self.refine_head.forward_test(img_refine, prev_outputs, img_metas, self.test_cfg)
        
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out
    
    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        img_os2 = nn.functional.interpolate(img, size=[img.shape[-2]//self.down_scale, img.shape[-1]//self.down_scale])
        x = self.extract_feat(img_os2)
        img_refine = img
        
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_refine, img_metas, gt_semantic_seg)
        losses.update(loss_decode)
        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    def _decode_head_forward_train(self, x, img, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode, prev_features = self.decode_head.forward_train(
            x, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_decode, 'decode'))

        loss_refine, *loss_contrsative_list = self.refine_head.forward_train(img, prev_features, img_metas, gt_semantic_seg, self.train_cfg)
        losses.update(add_prefix(loss_refine, 'refine'))

        j = 1
        for loss_aux in loss_contrsative_list:
            losses.update(add_prefix(loss_aux, 'aux_' + str(j)))
            j += 1
        return losses
