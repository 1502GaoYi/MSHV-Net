import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *

# from models.UltraLight_VM_UNet import MALUNet
from engine import *
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings

warnings.filterwarnings("ignore")


# torch.backends.cudnn.deterministic = True


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    set_seed(config.seed)
    gpu_ids = [0]  # [0, 1, 2, 3]
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = isic_loader(path_Data=config.data_path, train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    val_dataset = isic_loader(path_Data=config.data_path, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=config.num_workers,
                            drop_last=True)
    test_dataset = isic_loader(path_Data=config.data_path, train=False, Test=True)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             shuffle=False,
                             pin_memory=True,
                             num_workers=config.num_workers,
                             drop_last=True)

    print('#----------Prepareing Models----------#')
    model_cfg = config.model_config
    from models.UltraLight_VM_UNet_MSFN_RFA import MALUNet
    model = MALUNet(num_classes=model_cfg['num_classes'],
                    input_channels=model_cfg['input_channels'],
                    c_list=model_cfg['c_list'],
                    split_att=model_cfg['split_att'],
                    bridge=model_cfg['bridge'])
    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

    # best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
    'results/MALUNet_ISIC2017_Thursday_25_July_2024_10h_18m_52s/checkpoints/best-epoch98-loss0.3141.pth'
    'results/MALUNet_ISIC2018_Friday_26_July_2024_10h_06m_22s/checkpoints/best-epoch92-loss0.3090.pth'
    best_weight = torch.load('results/ISIC2017/best-epoch33-loss0.2913.pth', map_location=torch.device('cpu'))

    model.module.load_state_dict(best_weight)


    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    print('#----------Testing----------#')

    loss = test_one_epoch(
        test_loader,
        model,
        criterion,
        logger,
        config,
    )



if __name__ == '__main__':
    config = setting_config
    main(config)