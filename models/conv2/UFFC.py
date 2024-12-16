import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""ICCV2023
最近提出的图像修复方法 LaMa 以快速傅里叶卷积 (FFC) 为基础构建了其网络，该网络最初是为图像分类等高级视觉任务而提出的。
FFC 使全卷积网络在其早期层中拥有全局感受野。得益于 FFC 模块的独特特性，LaMa 能够生成稳健的重复纹理，
这是以前的修复方法无法实现的。但是，原始 FFC 模块是否适合图像修复等低级视觉任务？
在本文中，我们分析了在图像修复中使用 FFC 的基本缺陷，即 1) 频谱偏移、2) 意外的空间激活和 3) 频率感受野有限。
这些缺陷使得基于 FFC 的修复框架难以生成复杂纹理并执行完美重建。
基于以上分析，我们提出了一种新颖的无偏快速傅里叶卷积 (UFFC) 模块，该模块通过
 1) 范围变换和逆变换、2) 绝对位置嵌入、3) 动态跳过连接和 4) 自适应剪辑对原始 FFC 模块进行了修改，以克服这些缺陷，
实现更好的修复效果。在多个基准数据集上进行的大量实验证明了我们方法的有效性，在纹理捕捉能力和表现力方面均优于最先进的方法。
"""

class FourierUnit_modified(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit_modified, self).__init__()
        self.groups = groups

        self.input_shape = 64  # change!!!!!it!!!!!!manually!!!!!!
        self.in_channels = in_channels

        self.locMap = nn.Parameter(torch.rand(self.input_shape, self.input_shape // 2 + 1))

        self.lambda_base = nn.Parameter(torch.tensor(0.), requires_grad=True)

        self.conv_layer_down55 = torch.nn.Conv2d(in_channels=in_channels * 2 + 1,  # +1 for locmap
                                                 out_channels=out_channels * 2,
                                                 kernel_size=1, stride=1, padding=0, dilation=1, groups=self.groups,
                                                 bias=False, padding_mode='reflect')
        self.conv_layer_down55_shift = torch.nn.Conv2d(in_channels=in_channels * 2 + 1,  # +1 for locmap
                                                       out_channels=out_channels * 2,
                                                       kernel_size=3, stride=1, padding=2, dilation=2,
                                                       groups=self.groups, bias=False, padding_mode='reflect')

        self.norm = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

        self.img_freq = None
        self.distill = None

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode,
                              align_corners=False)

        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        locMap = self.locMap.expand_as(ffted[:, :1, :, :])  # B 1 H' W'
        ffted_copy = ffted.clone()

        cat_img_mask_freq = torch.cat((ffted[:, :self.in_channels, :, :],
                                       ffted[:, self.in_channels:, :, :],
                                       locMap), dim=1)

        ffted = self.conv_layer_down55(cat_img_mask_freq)
        ffted = torch.fft.fftshift(ffted, dim=-2)

        ffted = self.relu(ffted)

        locMap_shift = torch.fft.fftshift(locMap, dim=-2)  ## ONLY IF NOT SHIFT BACK

        # REPEAT CONV
        cat_img_mask_freq1 = torch.cat((ffted[:, :self.in_channels, :, :],
                                        ffted[:, self.in_channels:, :, :],
                                        locMap_shift), dim=1)

        ffted = self.conv_layer_down55_shift(cat_img_mask_freq1)
        ffted = torch.fft.fftshift(ffted, dim=-2)

        lambda_base = torch.sigmoid(self.lambda_base)

        ffted = ffted_copy * lambda_base + ffted * (1 - lambda_base)

        # irfft
        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        epsilon = 0.5
        output = output - torch.mean(output) + torch.mean(x)
        output = torch.clip(output, float(x.min() - epsilon), float(x.max() + epsilon))

        self.distill = output  # for self perc
        return output

if __name__ == '__main__':
    in_channels = 64
    out_channels = 64

    block = FourierUnit_modified(in_channels=in_channels, out_channels=out_channels)

    input_tensor = torch.rand(1, in_channels, 64, 64)

    output = block(input_tensor)

    print("Input size:", input_tensor.size())
    print("Output size:", output.size())