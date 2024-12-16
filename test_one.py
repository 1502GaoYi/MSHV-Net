import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *

# from models.UltraLight_VM_UNet import MALUNet
# from engine import *
import os
import sys


import numpy as np
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix, accuracy_score
from utils import save_imgs
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import mean_absolute_error
import openpyxl
import os
from openpyxl import Workbook, load_workbook
from openpyxl.worksheet import worksheet
from configs.config_setting import setting_config

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings

warnings.filterwarnings("ignore")


# torch.backends.cudnn.deterministic = True
from openpyxl import Workbook

import os
import sys
import argparse
import importlib.util

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils import *
import warnings

warnings.filterwarnings("ignore")

# 从配置文件加载默认参数
def load_config(config_path):
    spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description="Training script")

    # 配置文件参数：现在默认指向 `configs/config_setting_CGNet_DySample_GSConvns_CBAM.py`
    parser.add_argument('--config', type=str, default='configs/config_setting.py', help='Path to the config file (Python script)')

    # 数据集参数：如果没有传递，使用配置文件中的默认值
    parser.add_argument('--dataset', type=str, help='Dataset name')

    # 训练轮数参数：如果没有传递，使用配置文件中的默认值
    parser.add_argument('--epochs', type=int, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, help='Number of batch_size for training')

    parser.add_argument('--network', type=str, help='network')
    parser.add_argument('--M', type=int, help='M')
    parser.add_argument('--N', type=int, help='N')

    return parser.parse_args()


def main():
    # 解析命令行参数
    args = parse_args()

    # 打印命令行参数，确保解析工作正常
    print(f"Command line arguments: {args}")


    # 加载配置文件
    config = load_config(args.config)
    if args.network is not None:
        config.setting_config.network = args.network
        from datetime import datetime
        config.setting_config.work_dir = 'results/' + args.network + str(args.batch_size)+ '_' + args.dataset  + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    if args.M is not None:
        config.setting_config.M =   int(args.M )
        print(config.setting_config.M,'------------------------')
    if args.N is not None:
        config.setting_config.N = int(args.N )
        print(config.setting_config.N,'------------------------')

    # 覆盖配置文件中的数据集参数
    if args.dataset is not None:
        # 直接覆盖配置中的 dataset 设置，确保命令行优先
        config.setting_config.datasets = args.dataset

        print('config.setting_config.datasets ',config.setting_config.datasets )
        if args.dataset == 'ISIC2017':
            config.setting_config.data_path = 'data/dataset_isic17/'
        elif args.dataset == 'ISIC2018':
            config.setting_config.data_path = 'data/dataset_isic18/'
        elif args.dataset == 'PH2':
            config.setting_config.data_path = 'data/dataset_ph2/'
        else:
            raise ValueError("Unsupported dataset!")
    else:
        # 如果命令行没有传入 datasets，则使用默认设置
        print(f"Using default dataset from setting_config: {config.setting_config.datasets}")


    # 覆盖 epochs 参数，如果命令行没有传递，则使用配置文件中的默认值
    if args.epochs is not None:
        config.setting_config.epochs = args.epochs
    else:
        print(f"Using default epochs: {config.setting_config.epochs}")

    # 覆盖 batch_size 参数，如果命令行没有传递，则使用配置文件中的默认值
    if args.batch_size is not None:
        config.setting_config.batch_size = args.batch_size
    else:
        print(f"Using default epochs: {config.setting_config.batch_size}")




    # 输出确认
    print(f"Final config: dataset={config.setting_config.datasets}, data_path={config.setting_config.data_path}")
    print(f"epochs={config.setting_config.epochs}, batch_size={config.setting_config.batch_size}")

    # 继续调用训练流程
    train(config.setting_config)



def test_one_epoch(test_loader, model, criterion, logger, config, model_name='Model'):
    # 创建一个新的工作簿
    wb = Workbook()

    # 创建一个工作表并命名为模型名称
    ws = wb.active
    ws.title = f'{model_name} Results'  # 工作表名称为模型名称

    # 添加表头
    ws.append([
        'Image', 'Loss', 'Accuracy', 'Sensitivity', 'Specificity', 'F1 or DSC',
        'MIoU', 'IoU', 'Precision', 'Recall', 'F1', 'MAE', 'Kappa', 'OA'
    ])

    # 切换到评估模式
    model.eval()
    loss_list = []

    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            # 使用模型进行推理
            out = model(img)

            # 计算损失
            loss = criterion(out, msk)
            loss_list.append(loss.item())

            # 将评估结果处理为一维数组
            msk = msk.squeeze(1).cpu().detach().numpy()
            out = out.squeeze(1).cpu().detach().numpy()




            # 计算评价指标
            y_pre = np.where(out >= config.threshold, 1, 0)
            y_true = np.where(msk >= 0.5, 1, 0)
            confusion = confusion_matrix(y_true.flatten(), y_pre.flatten())
            TN, FP, FN, TP = confusion[0, 0], confusion[0, 1], confusion[1, 0], confusion[1, 1]

            accuracy = float(TN + TP) / float(np.sum(confusion)) if np.sum(confusion) != 0 else 0
            sensitivity = float(TP) / float(TP + FN) if (TP + FN) != 0 else 0
            specificity = float(TN) / float(TN + FP) if (TN + FP) != 0 else 0
            f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if (2 * TP + FP + FN) != 0 else 0
            miou = float(TP) / float(TP + FP + FN) if (TP + FP + FN) != 0 else 0

            # 将评估结果写入Excel
            log_info = [
                f'{i + 1}/{len(test_loader)}',
                f'{loss.item():.4f}',
                f'{accuracy:.4f}',
                f'{sensitivity:.4f}',
                f'{specificity:.4f}',
                f'{f1_or_dsc:.4f}',
                f'{miou:.4f}',
                f'{accuracy:.4f}',
                f'{accuracy:.4f}',
                f'{accuracy:.4f}',
                f'{accuracy:.4f}',
                f'{accuracy:.4f}',
                f'{accuracy:.4f}',
                f'{accuracy:.4f}'
            ]
            ws.append(log_info)

    # 保存Excel文件
    wb.save(os.path.join(config.work_dir, f'{model_name}_test_results.xlsx'))
    print(os.path.join(config.work_dir, f'{model_name}_test_results.xlsx'))


def train(config):
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



    #-------------------------------------------------------------------------------------------------------------------


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



    if config.network == 'UltraLight_VM_UNet_MSFN_CSPHet':
        from models.models_RFAConv.UltraLight_VM_UNet_assemFormer_EB_CSPHet import MALUNet
        input = torch.rand(1,3,256,256).cuda()
        model = MALUNet(num_classes=model_cfg['num_classes'],
                        input_channels=model_cfg['input_channels'],
                        c_list=model_cfg['c_list'],
                        split_att=model_cfg['split_att'],
                        bridge=model_cfg['bridge'])
        model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])
        best_weight = torch.load('results/UltraLight_VM_UNet_MSFN_CSPHet8_ISIC2018_Sunday_24_November_2024_08h_53m_19s/checkpoints/best-epoch24-loss0.3080.pth')
        model.module.load_state_dict(best_weight)


        output = model(input)
        print(output)

    #-------------------------------------------------------------------------------------------------------------------




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
    main()