# 모듈 import

from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.utils import get_device
import wandb

def train(data_dir, args, train_logger):
    train_logger.info(args)
    run = wandb.init(
        project="jjh_test", 
        entity = "level2-cv-10-detection",
        # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
        name=args.name,
        # Track hyperparameters and run metadata
        config = { 
            "epochs" : args.epochs,
            "dataset" : args.dataset,
            "augmentation" : args.augmentation,
            "resize" : args.resize,
            "batch_size" : args.batch_size,
            "valid_batch_size" : args.valid_batch_size,
            "model" : args.model,
            "optimizer" : args.optimizer,
            "lr" : args.lr,
            "val_ratio" : args.val_ratio,
            "criterion" : args.criterion,
            "lr_decay_step" : args.lr_decay_step,
            "log_interval" : args.log_interval,
            "data_dir" : args.data_dir,
            "model_dir" : args.model_dir,
            "k_fold": args.k_fold,
            "category_train": args.category_train
        }
    )
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

    # config file 들고오기
    cfg = Config.fromfile('/data/ephemeral/home/level2-objectdetection-cv-10/baseline/mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py')

    root=data_dir

    # dataset config 수정
    cfg.data.train.pipeline[4]['mean'] # [123.675, 116.28, 103.53]
    cfg.data.train.pipeline[4]['std'] # [58.395, 57.12, 57.375]
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + 'train.json' # train json 정보
    cfg.data.train.pipeline[2]['img_scale'] = (512, 512) # Resize

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json' # test json 정보
    cfg.data.test.pipeline[1]['img_scale'] = (512, 512) # Resize

    cfg.data.samples_per_gpu = 4

    cfg.seed = args.seed
    cfg.gpu_ids = [0]
    cfg.work_dir = './work_dirs/faster_rcnn_r50_fpn_1x_trash'

    cfg.model.roi_head.bbox_head.num_classes = 10

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()
    # build_dataset
    datasets = [build_dataset(cfg.data.train)]
    # 모델 build 및 pretrained network 불러오기
    model = build_detector(cfg.model)
    model.init_weights()
    # 모델 학습
    train_detector(model, datasets[0], cfg, distributed=False, validate=False)