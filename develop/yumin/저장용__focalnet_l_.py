# _base_ = [
#     #'./models/__sparse_rcnn_focal_custom.py',
#     '../_base_/datasets/__coco_detection_custom.py',
#     '../_base_/schedules/__schedule_custom.py',
#     '../_base_/default_runtime.py'
# ]

##### 모델 구현 #####

#pretrained = 'https://github.com/microsoft/FocalNet/releases/download/v1.0.0/focalnet_large_lrf_384_fl4.pth'  # noqa
pretrained = '/data/ephemeral/home/level2-objectdetection-cv-10/develop/yumin/UniverseNet/configs/focalnet/epoch_36.pth'

num_proposals = 300
  
# model settings
num_stages = 6
num_proposals = 100
model = dict(
    type='SparseRCNN',
    backbone=dict(
        type='FocalNet',
        embed_dim=192,
        depths=[2, 2, 6, 2],
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False,
        focal_windows=[3, 5, 7, 9],
        focal_levels=[4, 4, 4, 4],
        use_conv_embed=False,
        pretrained=None),
    neck=dict(
        type='FPN',
        in_channels=[192, 384, 768, 1536],
        out_channels=256,
        start_level=0,
        add_extra_convs='on_input',
        num_outs=4),
    rpn_head=dict(
        type='EmbeddingRPNHead',
        num_proposals=num_proposals,
        proposal_feature_channel=256),
    roi_head=dict(
        type='SparseRoIHead',
        num_stages=num_stages,
        stage_loss_weights=[1] * num_stages,
        proposal_feature_channel=256,
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=2),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='DIIHead',
                num_classes=10,
                num_ffn_fcs=2,
                num_heads=8,
                num_cls_fcs=1,
                num_reg_fcs=3,
                feedforward_channels=2048,
                in_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                dynamic_conv_cfg=dict(
                    type='DynamicConv',
                    in_channels=256,
                    feat_channels=64,
                    out_channels=256,
                    input_feat_shape=7,
                    act_cfg=dict(type='ReLU', inplace=True),
                    norm_cfg=dict(type='LN')),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                loss_cls=dict(
                    type='FocalLoss',
                    use_sigmoid=True,
                    gamma=2.0,
                    alpha=0.25,
                    loss_weight=2.0),
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    clip_border=False,
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.5, 0.5, 1., 1.])) for _ in range(num_stages)
        ]),
    # training and testing settings
    train_cfg=dict(
        rpn=None,
        rcnn=[
            dict(
                assigner=dict(
                    type='HungarianAssigner',
                    cls_cost=dict(type='FocalLossCost', weight=2.0),
                    reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                    iou_cost=dict(type='IoUCost', iou_mode='giou',
                                  weight=2.0)),
                sampler=dict(type='PseudoSampler'),
                pos_weight=1) for _ in range(num_stages)
        ]),
    test_cfg=dict(rpn=None, rcnn=dict(max_per_img=num_proposals)))



##### 데이터셋 #####

dataset_type = 'CocoDataset'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    #dict(type='Resize', img_scale=(1024,1024)),
    dict(
        type='AutoAugment',
        policies=[[
            dict(
                type='Resize',
                img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                           (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                           (736, 1333), (768, 1333), (800, 1333), (830, 830),
                           (860, 860), (900,900), (960, 960), (1024, 1024)],
                multiscale_mode='value',
                keep_ratio=True)
        ]]),
             
    #dict(type='CutOut', n_holes=3,cutout_shape=(20, 20)),
    dict(type='RandomFlip', flip_ratio=0.5),
    #dict(type='RandomAffine', max_rotate_degree=30 ),
    #dict(
    #    type='OneOf',
    #    transforms=[
    #        dict(type='Blur', blur_limit=3, p=0.2),
    #        dict(type='MedianBlur', blur_limit=3, p=0.2)
    #    ],
    #    p=0.1),
    #dict(type='PhotoMetricDistortion', brightness_delta=15, hue_delta=2),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file= ' json file address ',
        img_prefix= ' image folder address/ ',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file= ' json file address ',
        img_prefix=' image folder address/ ',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=' json file address ',
        img_prefix=' image folder address/ ',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')


##### optimizer #####

optimizer = dict(type='AdamW', lr=0.00025, weight_decay=0.0001) # for focalnet
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))


##### scheduler #####

lr_config = dict(
    policy='cyclic',    # Policy를 'clr'로 설정
    by_epoch = False,
    target_ratio = (10, 1e-4),
    cyclic_times = 2,
    step_ratio_up = 0.4,
    anneal_strategy = 'cos',
    gamma = 0.5
)

checkpoint_config = dict(max_keep_ckpts=3, interval=1)
runner = dict(type='EpochBasedRunner', max_epochs=15)



##### default_runtime #####
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)


