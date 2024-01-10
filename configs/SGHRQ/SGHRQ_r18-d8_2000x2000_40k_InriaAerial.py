_base_ = [
    '../_base_/models/SGHRQ_r50-d8.py', '../_base_/datasets/aeril_2000x2000.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_40k.py'
]
bn_type = 'BN'
norm_cfg = dict(type=bn_type, requires_grad=True)
data = dict(
    samples_per_gpu=4, # batch_size
    workers_per_gpu=8,)
model = dict(
    pretrained='open-mmlab://resnet18_v1c',
    down_ratio=4,
    backbone=dict(depth=18),
    decode_head=[
        dict(
            type='RefineASPPHead',
            in_channels=512,
            in_index=3,
            channels=128,
            dilations=(1, 12, 24, 36),
            dropout_ratio=0.1,
            num_classes=2,
            norm_cfg=norm_cfg,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
        dict(
            type='SGHRQHead',
            in_channels=3,
            prev_channels=128,
            down_ratio=4,
            channels=128,
            num_classes=2,
            dropout_ratio=0.1,
            norm_cfg=norm_cfg,
            bn_type=bn_type,
            align_corners=False,
            loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    ],
    auxiliary_head=dict(in_channels=256, channels=64, num_classes=2),#)
    test_cfg=dict(mode='slide', crop_size=(1280, 1280), stride=(960, 960)))
