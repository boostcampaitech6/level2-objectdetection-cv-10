# 모듈 import

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device

classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

cfg = Config.fromfile('./configs/efficientnet/retinanet_effb3_fpn_crop896_8x4_1x_coco.py')

root='../../../dataset/'

# dataset config 수정
cfg.data.train.classes = classes
cfg.data.train.img_prefix = root
cfg.model.bbox_head.num_classes=10

cfg.data.train.ann_file = root + 'train.json' # train json 정보
# cfg.data.train.pipeline[2]['img_scale'] = (512,512) # Resize

cfg.data.test.classes = classes
cfg.data.test.img_prefix = root
cfg.data.test.ann_file = root + 'test.json' # test json 정보
# cfg.data.test.pipeline[1]['img_scale'] = (512,512) # Resize

cfg.data.samples_per_gpu = 4

cfg.seed = 2022
cfg.gpu_ids = [0]
cfg.work_dir = './work_dirs/retinanet_effb3_fpn_crop896_8x4_1x_trash'
# cfg.work_dir = './work_dirs/'+model_name+'_trash'
# cfg.model.roi_head.bbox_head.nu

# cfg.model.roi_head.bbox_head[0].num_classes = 10
# cfg.model.roi_head.bbox_head[1].num_classes = 10
# cfg.model.roi_head.bbox_head[2].num_classes = 10
# cfg.model.roi_head.mask_head.num_classes=10

cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
cfg.device = get_device()
cfg.runner.max_epochs=100
datasets = [build_dataset(cfg.data.train)]

model = build_detector(cfg.model)
model.init_weights()

train_detector(model, datasets[0], cfg, distributed=False, validate=False)