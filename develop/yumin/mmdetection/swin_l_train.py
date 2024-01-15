# 모듈 import
# python cascade_train.py

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device
import wandb

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

wandb.init(project='cascade', name='cascade_swin_multi_scale')
# config file 들고오기

cfg = Config.fromfile('/data/ephemeral/home/level2-objectdetection-cv-10/develop/yumin/mmdetection/configs/_custom_/swin_l.py')
#cfg = Config.fromfile('./configs/cascade_rcnn/cascade_rcnn_x101_64x4d_fpn_20e_coco.py')

root='/data/ephemeral/home/level2-objectdetection-cv-10/dataset/'

# dataset config 수정
cfg.data.train.classes = classes
cfg.data.train.img_prefix = root
cfg.data.train.ann_file = root + 'updated_train.json' # train json 정보

cfg.data.test.classes = classes
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = root + 'test.json' # test json 정보

cfg.data.samples_per_gpu = 2

cfg.seed = 2022
cfg.gpu_ids = [0]
cfg.work_dir = '/data/ephemeral/home/level2-objectdetection-cv-10/develop/yumin/mmdetection/work_dirs/swin_b_trash'

#cfg.model.roi_head.bbox_head.num_classes = 10
cfg.model.roi_head.bbox_head[0].num_classes = 10
cfg.model.roi_head.bbox_head[1].num_classes = 10
cfg.model.roi_head.bbox_head[2].num_classes = 10

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
cfg.device = get_device()

cfg.log_config.hooks = [
    dict(type='TextLoggerHook'),
    dict(type='MMDetWandbHook',
         init_kwargs={'project':'cascade'},
         interval=10,
         log_checkpoint=True,
         log_checkpoint_metadata=True,
         num_eval_images=100)
]


# build_dataset
datasets = [build_dataset(cfg.data.train)]

# 모델 build 및 pretrained network 불러오기
model = build_detector(cfg.model)
model.init_weights()

# 모델 학습
train_detector(model, datasets[0], cfg, distributed=False, validate=False)