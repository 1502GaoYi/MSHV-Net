import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from loader import *

from engine import *
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0, 1, 2, 3"

from utils import *
from configs.config_setting import setting_config

import warnings

warnings.filterwarnings("ignore")


# torch.backends.cudnn.deterministic = True

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


def train(config):
    print('#----------Creating logger----------#')
    # sys.path.append(config.work_dir + '/')
    sys.path.append(config.work_dir + '/')

    log_dir = os.path.join(config.work_dir, 'log')
    print(log_dir,config.work_dir,'------------------------------------------')
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
    #本人网络---------------------------------------------------------------------------------------

    if config.network == 'UltraLight_VM_UNet_RFA_assemFormer_EB':
        from models.UltraLight_VM_UNet_RFA_assemFormer_EB import MALUNet



        # 模板
        model_weight_path = "results/ISIC2018/best-epoch18-loss0.2874.pth"  # 预训练权重的路径
        if config.datasets == 'ISIC2018':
            model_weight_path = "results/ISIC2018/best-epoch98-loss0.2846.pth"  # 预训练权重的路径
            print(model_weight_path,'# 预训练权重的路径')
        if config.datasets == 'ISIC2017':
            model_weight_path = "results/ISIC2017/best-epoch33-loss0.2913.pth"  # 预训练权重的路径

        ckpt = torch.load(model_weight_path)  # 加载预训练权重
        # 实例化模型
        model = MALUNet(num_classes=model_cfg['num_classes'],
                        input_channels=model_cfg['input_channels'],
                        c_list=model_cfg['c_list'],
                        split_att=model_cfg['split_att'],
                        bridge=model_cfg['bridge'])



        model_dict = model.state_dict()  # 获取我们模型的参数

        # 判断预训练模型中网络的模块是否在修改后的网络中也存在，并且shape相同，如果相同则取出
        pretrained_dict = {k: v for k, v in ckpt.items() if k in model_dict and (v.shape == model_dict[k].shape)}

        # 更新修改之后的model_dict
        model_dict.update(pretrained_dict)

        # 加载我们真正需要的state_dict
        model.load_state_dict(model_dict, strict=True)


    #对比实验模型包括以下，cmd上注意名字
    #Attention_UNet  HSH_UNet  MALUNet  MHA_UNet  MHorUNet
    #SegNet  UnetPlusPlus  UNet  Unext_S UCM_Net
    # CMUNeXt  DFANet  ENet
    if config.network == 'Attention_UNet':
        from models.models.Attention_UNet import AttU_Net
        model = AttU_Net(in_channel=3,
                         num_classes=1,
                         channel_list=[64, 128, 256, 512, 1024],
                         checkpoint=False,
                         convTranspose=True).cuda()
    if config.network == 'HSH_UNet':
        from models.models.HSH_UNet import HSH_UNet
        model = HSH_UNet(num_classes=1, input_channels=3,
                         pretrained=None,
                         use_checkpoint=False, c_list=[32, 64, 128, 256, 512],
                         split_att='fc', bridge=True)


    if config.network == 'MALUNet':
        from models.models.MALUNet import MALUNet
        model = MALUNet(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                        bridge=True).cuda()
    if config.network == 'MACUNet':
        from models.models.MACUNet import MACUNet
        model = MACUNet(3,1).cuda()
    if config.network == 'HF_UNet':
        from models.models.HF_UNet import HFUNet
        model = HFUNet(1,3).cuda()


    if config.network == 'MHA_UNet':
        from models.models.MHA_UNet import MHA_UNet

        model = MHA_UNet().cuda()
    if config.network == 'MHorUNet':
        from models.models.MHorUNet import MHorunet
        model =  MHorunet(num_classes=1, input_channels=3, layer_scale_init_value=1e-6,
                 pretrained=None,
                 use_checkpoint=False, c_list=[8, 16, 32, 64, 128, 256], depths=[2, 3, 18, 2], drop_path_rate=0.,
                 split_att='fc', bridge=True).cuda()
    if config.network == 'SegNet':
        from models.models.SegNet import SegNet

        model = SegNet(1)
    if config.network == 'UnetPlusPlus':   #unet++
        from models.models.UnetPlusPlus import UnetPlusPlus

        model = UnetPlusPlus(num_classes=1, deep_supervision=False)
    if config.network == 'UNet':
        from models.models.UNet import Unet

        model = Unet(num_classes=1)
    if config.network == 'CENet':
        from models.models.CENet import CE_Net_

        model = CE_Net_()
    if config.network == 'Unext_S':
        from models.models.UNext_S import UNext_S

        model = UNext_S(num_classes=1, input_channels=3, deep_supervision=False, img_size=256, patch_size=16, in_chans=3,
                    embed_dims=[32, 64, 128, 512],
                    num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                    attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                    depths=[1, 1, 1], sr_ratios=[8, 4, 2, 1])

    if config.network == 'UCM_Net':
        from models.models.UCM_Net import UCM_Net

        model = UCM_Net(num_classes=1)

    if config.network == 'CMUNet':
        from models.models.CMUNet import CMUNet

        model = CMUNet(3,1)

    if config.network == 'CMUNeXt':
        from models.models.CMUNeXt import CMUNeXt
        model = CMUNeXt(num_classes=1).cuda()
    if config.network == 'DFANet':
        from models.models.DFANet import DFANet
        model = DFANet(num_classes=1).cuda()
    if config.network == 'ENet':
        from models.models.ENet import ENet
        model = ENet(n_classes=1).cuda()
    if config.network == 'SCSONet':
        from models.models.SCSONet import SCSONet
        model = SCSONet().cuda()

    if config.network == 'Rolling_Unet':
        from models.models.Rolling_Unet import Rolling_Unet_S
        model = Rolling_Unet_S(1,3).cuda()

    if config.network == 'A2FPN':
        from models.models.A2FPN import A2FPN
        model = A2FPN(3,1).cuda()

    if config.network == 'ABCNet':
        from models.models.ABCNet import ABCNet
        model = ABCNet(3,1).cuda()

    if config.network == 'MANet':
        from models.models.MANet import MANet
        model = MANet(3,1).cuda()

    if config.network == 'MAResUNet':
        from models.models.MAResUNet import MAResUNet
        model = MAResUNet(3,1).cuda()

    if config.network == 'MSHV_Net':
        from models.MSHV_Net import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                                   bridge=True).cuda()
    if config.network == 'H_vmunet':
        from models.models.vmunet.H_vmunet import H_vmunet
        model = H_vmunet(num_classes=1,
                         input_channels=3,
                         c_list=[8, 16, 32, 64, 128, 256],
                         split_att='fc',
                         bridge=True,
                         drop_path_rate=0).cuda()
    if config.network == 'vmunet':
        from models.models.vmunet.vmunet import VMUNet
        model = VMUNet().cuda()
    if config.network == 'vmunet_v2':
        from models.models.vmunet.vmunet_v2 import VMUNetV2
        model = VMUNetV2(load_ckpt_path=None, deep_supervision=True).cuda()

    if config.network == 'UltraLight_VM_UNet':
        from models.models.UltraLight_VM_UNet import UltraLight_VM_UNet
        model = UltraLight_VM_UNet(num_classes = 1, input_channels = 3,
                                   c_list = [8, 16, 24, 32, 48, 64],
                                   split_att = 'fc', bridge = True).cuda()



    #注意力桥ACF的消融实验
    # MSHV_Net_RFM_MSWM_CAFM
    # MSHV_Net_RFM_MSWM_CBAM
    # MSHV_Net_RFM_MSWM_DSAM
    # MSHV_Net_RFM_MSWM_GAM_Attention
    # MSHV_Net_RFM_MSWM_GCT
    # MSHV_Net_RFM_MSWM_GLSA
    # MSHV_Net_RFM_MSWM_LKA
    # MSHV_Net_RFM_MSWM_MDTA
    # MSHV_Net_RFM_MSWM_SCSA

    if config.network == 'MSHV_Net_RFM_MSWM_CAFM':
        from models.att_ACF.MSHV_Net_RFM_MSWM_CAFM import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                                   bridge=True).cuda()
    if config.network == 'MSHV_Net_RFM_MSWM_CBAM':
        from models.att_ACF.MSHV_Net_RFM_MSWM_CBAM import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                                   bridge=True).cuda()
    if config.network == 'MSHV_Net_RFM_MSWM_DSAM':
        from models.att_ACF.MSHV_Net_RFM_MSWM_DSAM import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                                   bridge=True).cuda()
    if config.network == 'MSHV_Net_RFM_MSWM_GAM_Attention':
        from models.att_ACF.MSHV_Net_RFM_MSWM_GAM_Attention import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                                   bridge=True).cuda()

    if config.network == 'MSHV_Net_RFM_MSWM_GCT':
        from models.att_ACF.MSHV_Net_RFM_MSWM_GCT import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()
    if config.network == 'MSHV_Net_RFM_MSWM_GLSA':
        from models.att_ACF.MSHV_Net_RFM_MSWM_GLSA import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()

    if config.network == 'MSHV_Net_RFM_MSWM_LKA':
        from models.att_ACF.MSHV_Net_RFM_MSWM_LKA import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()
    if config.network == 'MSHV_Net_RFM_MSWM_MDTA':
        from models.att_ACF.MSHV_Net_RFM_MSWM_MDTA import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()
    if config.network == 'MSHV_Net_RFM_MSWM_SCSA':
        from models.att_ACF.MSHV_Net_RFM_MSWM_SCSA import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()


    #上面卷积RFM的消融实验

    # MSHV_Net_ACF_MSWM_AKConv
    # MSHV_Net_ACF_MSWM_C2F_Dual
    # MSHV_Net_ACF_MSWM_C2F_MSBlock
    # MSHV_Net_ACF_MSWM_CondConv
    # MSHV_Net_ACF_MSWM_CSPHet
    # MSHV_Net_ACF_MSWM_DOConv
    # MSHV_Net_ACF_MSWM_GhostModule
    # MSHV_Net_ACF_MSWM_GhostModuleV2
    # MSHV_Net_ACF_MSWM_RefConv
    if config.network == 'MSHV_Net_ACF_MSWM_AKConv':
        from models.conv1_RFM.MSHV_Net_ACF_MSWM_AKConv import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()
    if config.network == 'MSHV_Net_ACF_MSWM_C2F_Dual':
        from models.conv1_RFM.MSHV_Net_ACF_MSWM_C2F_Dual import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()
    if config.network == 'MSHV_Net_ACF_MSWM_C2F_MSBlock':
        from models.conv1_RFM.MSHV_Net_ACF_MSWM_C2F_MSBlock import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()

    if config.network == 'MSHV_Net_ACF_MSWM_CondConv':
        from models.conv1_RFM.MSHV_Net_ACF_MSWM_CondConv import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()
    if config.network == 'MSHV_Net_ACF_MSWM_CSPHet':
        from models.conv1_RFM.MSHV_Net_ACF_MSWM_CSPHet import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()
    if config.network == 'MSHV_Net_ACF_MSWM_DOConv':
        from models.conv1_RFM.MSHV_Net_ACF_MSWM_DOConv import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()
    if config.network == 'MSHV_Net_ACF_MSWM_GhostModule':
        from models.conv1_RFM.MSHV_Net_ACF_MSWM_GhostModule import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()
    if config.network == 'MSHV_Net_ACF_MSWM_GhostModuleV2':
        from models.conv1_RFM.MSHV_Net_ACF_MSWM_GhostModuleV2 import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()

    if config.network == 'MSHV_Net_ACF_RFM_ConvEncoder':
        # from models.conv2_MSWM.MSHV_Net_ACF_RFM_ConvEncoder import MSHV_Net
        from models.conv2_MSWM.MSHV_Net_ACF_RFM_ConvEncoder import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()

    if config.network == 'MSHV_Net_ACF_MSWM_RefConv':
        from models.conv1_RFM.MSHV_Net_ACF_MSWM_RefConv import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()


    #下面卷积MSWM的消融实验

    # MSHV_Net_ACF_RFM_ConvEncoder
    # MSHV_Net_ACF_RFM_CoordAtt
    # MSHV_Net_ACF_RFM_CPCA
    # MSHV_Net_ACF_RFM_EVC
    # MSHV_Net_ACF_RFM_GLSA
    # MSHV_Net_ACF_RFM_MDCR
    # MSHV_Net_ACF_RFM_MLA
    # MSHV_Net_ACF_RFM_RAB
    # MSHV_Net_ACF_RFM_ScConv
    # MSHV_Net_ACF_RFM_SwiftFormerEncoder
    if config.network == 'MSHV_Net_ACF_RFM_ConvEncoder':
        # from models.conv2_MSWM.MSHV_Net_ACF_RFM_ConvEncoder import MSHV_Net
        from models.conv2_MSWM.MSHV_Net_ACF_RFM_ConvEncoder import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()
    if config.network == 'MSHV_Net_ACF_RFM_CoordAtt':
        from models.conv2_MSWM.MSHV_Net_ACF_RFM_CoordAtt import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()
    if config.network == 'MSHV_Net_ACF_RFM_CPCA':
        from models.conv2_MSWM.MSHV_Net_ACF_RFM_CPCA import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()
    if config.network == 'MSHV_Net_ACF_RFM_EVC':
        from models.conv2_MSWM.MSHV_Net_ACF_RFM_EVC import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()
    if config.network == 'MSHV_Net_ACF_RFM_GLSA':
        from models.conv2_MSWM.MSHV_Net_ACF_RFM_GLSA import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()
    if config.network == 'MSHV_Net_ACF_RFM_MDCR':
        from models.conv2_MSWM.MSHV_Net_ACF_RFM_MDCR import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()
    if config.network == 'MSHV_Net_ACF_RFM_MLA':
        from models.conv2_MSWM.MSHV_Net_ACF_RFM_MLA import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()
    if config.network == 'MSHV_Net_ACF_RFM_RAB':
        from models.conv2_MSWM.MSHV_Net_ACF_RFM_RAB import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()
    if config.network == 'MSHV_Net_ACF_RFM_ScConv':
        from models.conv2_MSWM.MSHV_Net_ACF_RFM_ScConv import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()
    if config.network == 'MSHV_Net_ACF_RFM_SwiftFormerEncoder':
        from models.conv2_MSWM.MSHV_Net_ACF_RFM_SwiftFormerEncoder import MSHV_Net
        model = MSHV_Net(num_classes=1, input_channels=3, c_list=[8, 16, 24, 32, 48, 64], split_att='fc',
                         bridge=True).cuda()






    model = torch.nn.DataParallel(model.cuda(), device_ids=gpu_ids, output_device=gpu_ids[0])

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)
    scaler = GradScaler()

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.module.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']

        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

        train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            logger,
            config,
            scaler=scaler
        )

        loss = val_one_epoch(
            val_loader,
            model,
            criterion,
            epoch,
            logger,
            config
        )

        if loss < min_loss:
            torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save(
            {
                'epoch': epoch,
                'min_loss': min_loss,
                'min_epoch': min_epoch,
                'loss': loss,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.module.load_state_dict(best_weight)
        loss = test_one_epoch(
            test_loader,
            model,
            criterion,
            logger,
            config,
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )


# if __name__ == '__main__':
#     config = setting_config
#     main(config)

if __name__ == '__main__':
    main()