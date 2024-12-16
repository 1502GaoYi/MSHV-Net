<div align="center">
<h1>MSHV-Net: A Multi-Scale Hybrid Vision Network for Skin Image Segmentation </h1>

Haicheng Qu<sup>1</sup>\*, Yi Gao<sup>1</sup>, Qingling Jiang<sup>2</sup>, Ying Wang<sup>1</sup>

<sup>1</sup>  Liaoning Technical University, <sup>2</sup>  Tieling Normal College

</div>


**1、本文数据集是isic2017与isic2018** </br>

数据集的论文来源：
```
https://conferences.miccai.org/2023/papers/237-Paper1980.html
https://arxiv.org/pdf/2307.08473
```
原百度网盘地址：
```
https://pan.baidu.com/s/1Y0YupaH21yDN5uldl7IcZA?pwd=dybm
```
**2、本文项目借鉴开源项目** </br>
```
https://github.com/JCruan519/EGE-UNet?tab=readme-ov-file
https://github.com/wurenkai/UltraLight-VM-UNet
```


**3、数据集处理（windows）** </br>
注意绝对路径与相对路径，自己设好绝对路径与相对路径</br>
运行：  dataprepare/data批量转npy.py


**4、本文运行环境设置（linux）** </br>
注意cuda11.8以上</br>
***(1)torch安装***</br>
进入encironment/torch
```
pip install torch-2.0.0+cu118-cp38-cp38-linux_x86_64.whl 
pip install torchaudio-2.0.1+cu118-cp38-cp38-linux_x86_64.whl
pip install torchvision-0.15.1+cu118-cp38-cp38-linux_x86_64.whl
```
***(2)mamba安装***</br>
进入encironment/mamba</br>
```
pip install causal_conv1d-1.0.0+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm-1.0.1+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```
***(3)其它安装***</br>
进入encironment</br>
```
pip install -r environments.txt
```

**5、本文对比实验与消融实验公开** </br>

下面给出几个实例，注意自己设定网络名称、数据集以及其它参数</br>

#这是我的模型</br>
```
python train.py  --dataset ISIC2018 --epochs  75 --batch_size 8 --network MSHV_Net
```
#注意力桥ACF替换  消融实验</br>
```
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_RFM_MSWM_CAFM
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_RFM_MSWM_CBAM
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_RFM_MSWM_DSAM
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_RFM_MSWM_GAM_Attention
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_RFM_MSWM_GCT
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_RFM_MSWM_GLSA
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_RFM_MSWM_LKA
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_RFM_MSWM_MDTA
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_RFM_MSWM_SCSA
```
#上面卷积RFM替换  消融实验</br>
```
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_ACF_MSWM_AKConv
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_ACF_MSWM_C2F_Dual
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_ACF_MSWM_C2F_MSBlock
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_ACF_MSWM_CondConv
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_ACF_MSWM_CSPHet
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_ACF_MSWM_DOConv
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_ACF_MSWM_GhostModule
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_ACF_MSWM_GhostModuleV2
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_ACF_MSWM_RefConv
```

#下面卷积MSWM替换  消融实验</br>
```
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_ACF_RFM_ConvEncoder
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_ACF_RFM_CoordAtt
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MSHV_Net_ACF_RFM_CPCA
python train.py  --dataset ISIC2017 --epochs  1 --batch_size 32 --network MSHV_Net_ACF_RFM_EVC
python train.py  --dataset ISIC2017 --epochs  1 --batch_size 32 --network MSHV_Net_ACF_RFM_GLSA
python train.py  --dataset ISIC2017 --epochs  1 --batch_size 32 --network MSHV_Net_ACF_RFM_MDCR
python train.py  --dataset ISIC2017 --epochs  1 --batch_size 32 --network MSHV_Net_ACF_RFM_MLA
python train.py  --dataset ISIC2017 --epochs  1 --batch_size 32 --network MSHV_Net_ACF_RFM_RAB
python train.py  --dataset ISIC2017 --epochs  1 --batch_size 32 --network MSHV_Net_ACF_RFM_ScConv
python train.py  --dataset ISIC2017 --epochs  1 --batch_size 32 --network MSHV_Net_ACF_RFM_SwiftFormerEncoder
```

#对比实验</br>
```
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network UNet
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network SegNet
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network UnetPlusPlus
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network Attention_UNet
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network UNext_S
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network MALUNet
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network vmunet
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 32 --network vmunet_v2
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 32 --network H_vmunet
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network HSH_UNet
python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network UltraLight_VM_UNet
```
```
# windows上循环
for /L %i in (1,1,5) do (
    python train.py  --dataset ISIC2018 --epochs  75 --batch_size 8 --network MSHV_Net
    python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network UltraLight_VM_UNet
)

#------------------------------------------------------------------
# linux上循环
for ((i=1; i<=5; i++)); do
    python train.py  --dataset ISIC2018 --epochs  75 --batch_size 8 --network MSHV_Net
    python train.py  --dataset ISIC2018 --epochs  1 --batch_size 8 --network UltraLight_VM_UNet
done
shutdown

```

**6、测试** </br>
注意自己改模型名称与权重</br>
```
python test.py  #整个数据集
python test_one.py  #一张图片
```

