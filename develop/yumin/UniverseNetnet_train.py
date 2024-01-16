# 모듈 import

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device

import wandb
#wandb.init(project='cascade', name='focal_schedule')

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

# config file 들고오기
cfg = Config.fromfile('./develop/yumin/UniverseNet/configs/universenet/_universe.py')

root='./dataset/'

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
cfg.work_dir = './work_dirs/focal_saved_trash'

cfg.device = get_device()

# cfg.log_config.hooks = [
#     dict(type='TextLoggerHook'),
#     dict(type='MMDetWandbHook',
#          init_kwargs={'project':'cascade'},
#          interval=10,
#          log_checkpoint=True,
#          log_checkpoint_metadata=True,
#          num_eval_images=100)
# ]


meta = dict()
meta['fp16'] = cfg.fp16

# build_dataset
datasets = [build_dataset(cfg.data.train)]

model = build_detector(cfg.model)
model.init_weights()


# 모델 학습
train_detector(model, datasets[0], cfg, distributed=False, validate=False, meta=meta)
