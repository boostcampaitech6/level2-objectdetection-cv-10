_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/schedules/schedule_custom.py',
    '../_base_/default_runtime.py',
]
pretrained = './co_dino_5scale_r50_1x_coco.pth'  # noqa

model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')))


