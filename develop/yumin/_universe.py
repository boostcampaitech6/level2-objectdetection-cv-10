
pretrained_ckpt = 'open-mmlab://res2net101_v1d_26w_4s'

##### 모델 선언 #####
model = dict(
    type='GFL',
    backbone=dict(
        type='Res2Net',
        depth=101,
        scales=4,
        base_width=26,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        dcn=dict(type='DCN', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=pretrained_ckpt)),
    neck=[
        dict(
            type='FPN',
            in_channels=[256, 512, 1024, 2048],
            out_channels=256,
            start_level=1,
            add_extra_convs='on_output',
            num_outs=5),
        dict(
            type='SEPC',
            out_channels=256,
            stacked_convs=4,
            pconv_deform=True,
            lcconv_deform=True,
            ibn=False,  # please set imgs/gpu >= 4
            lcconv_padding=1)
    ],
    bbox_head=dict(
        type='GFLSEPCHead',
        num_classes=10,
        in_channels=256,
        stacked_convs=0,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
        reg_max=16,
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=100))


fp16 = dict(loss_scale=512.)

#### 데이터셋 선언 #####
# dataset settings
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
#optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer = dict(type='AdamW', lr=0.00025, weight_decay=0.0001) # for focalnet
optimizer_config = dict(grad_clip=dict(max_norm=15, norm_type=2))

#####  scheduler ##### 


lr_config = dict(
    policy='cyclic',
    by_epoch = False,
    target_ratio = (10, 1e-4),
    cyclic_times = 2,
    step_ratio_up = 0.4,
    anneal_strategy = 'cos',
    gamma = 0.5,
    warmup_iters=1000
)
# # learning policy
# lr_config = dict(
#     policy='step',
#     warmup='linear',
#     warmup_iters=500,
#     warmup_ratio=0.001,
#     step=[27, 33])
# runner = dict(type='EpochBasedRunner', max_epochs=36)

runner = dict(type='EpochBasedRunner', max_epochs=30)



##### default #####

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
