from torchvision import transforms
from utils import *
from easydict import EasyDict as edict
from datetime import datetime

setting_config = edict()

# Basic settings
setting_config.work_dir = './results/0519/'
setting_config.gpu_id = '0'
setting_config.seed = 42
setting_config.epochs = 200
setting_config.batch_size = 16
setting_config.num_workers = 4
setting_config.amp = False
setting_config.distributed = False
setting_config.local_rank = -1
setting_config.world_size = None
setting_config.rank = None

# Dataset settings
setting_config.datasets = 'isic2017'
if setting_config.datasets == 'isic18':
    setting_config.data_path = '/hy-tmp/PCViM/datasets/isic2018/'
elif setting_config.datasets == 'isic2017':
    setting_config.data_path = '/hy-tmp/PCViM/datasets/isic2017/'
else:
    raise Exception('datasets is not right!')

# Model settings
setting_config.network = 'transunet'

setting_config.model_config = {
    'num_classes': 1,
    'input_channels': 3,
    'depths': [2, 2, 2, 2],
    'depths_decoder': [2, 2, 2, 1],
    'drop_path_rate': 0.2,
    'load_ckpt_path': "/hy-tmp/PCViM/configs/pre_trained_weights/local_vssm_small.ckpt",
}

# Image settings
setting_config.input_size_h = 256
setting_config.input_size_w = 256
setting_config.input_channels = 3
setting_config.num_classes = 1
setting_config.threshold = 0.5

# Training settings
setting_config.criterion = BceDiceLoss(wb=1, wd=1)
setting_config.print_interval = 20
setting_config.val_interval = 1
setting_config.save_interval = 10

# Data augmentation
setting_config.train_transformer = transforms.Compose([
    myNormalize(setting_config.datasets, train=True),
    myToTensor(),
    myRandomHorizontalFlip(p=0.5),
    myRandomVerticalFlip(p=0.5),
    myRandomRotation(p=0.5, degree=[0, 360]),
    myResize(setting_config.input_size_h, setting_config.input_size_w)
])

setting_config.test_transformer = transforms.Compose([
    myNormalize(setting_config.datasets, train=False),
    myToTensor(),
    myResize(setting_config.input_size_h, setting_config.input_size_w)
])

# Optimizer settings
setting_config.opt = 'AdamW'
setting_config.lr = 0.001
setting_config.betas = (0.9, 0.999)
setting_config.eps = 1e-8
setting_config.weight_decay = 1e-2
setting_config.amsgrad = False

# Scheduler settings
setting_config.sch = 'CosineAnnealingLR'
setting_config.T_max = 50
setting_config.eta_min = 0.00001
setting_config.last_epoch = -1

# Warmup settings
setting_config.warmup_epochs = 5
setting_config.min_lr = 1e-6
