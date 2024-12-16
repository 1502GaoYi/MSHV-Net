import torch
import torch.nn as nn
import torch.nn.functional as F



# https://github.com/nibuiro/CondConv-pytorch/blob/master/condconv/condconv.py
#https://arxiv.org/pdf/1904.04971
#CondConv: Conditionally Parameterized Convolutions for Efficient Inference
"""
_routing: 这是一个辅助模块，定义了用于计算条件卷积权重的路由机制。它应用线性变换，然后使用 sigmoid 激活函数产生基于输入的路由权重。

CondConv2D: 这是主要的模块类。它继承自 _ConvNd，这是PyTorch中用于卷积层的基类。

初始化：

__init__ 方法初始化了 CondConv2D 层。它接受类似于标准卷积层的参数，但还包括额外的参数，如 num_experts（每层的专家数）和 dropout_rate（丢弃率）。
它将权重初始化为形状为 (num_experts, out_channels, in_channels // groups, *kernel_size) 的张量。
还初始化了 _avg_pooling 和 _routing_fn。
_conv_forward: 这是一个执行卷积操作的辅助方法。它处理除了'zeros'之外的填充模式。

forward: 这个方法定义了 CondConv2D 层的前向传播。

它使用 _routing_fn 计算每个输入样本的路由权重。
然后，根据路由权重以及专家核的加权和来计算卷积核。
最后，使用 _conv_forward 对这些计算出的卷积核执行卷积。
它将所有输入样本的结果连接起来并返回输出张量。
这里的条件卷积操作允许网络根据输入条件动态调整卷积核，从而可能增强模型适应数据中各种模式的能力。
"""
import functools
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter


class _routing(nn.Module):

    def __init__(self, in_channels, num_experts, dropout_rate):
        super(_routing, self).__init__()

        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        x = torch.flatten(x)
        x = self.dropout(x)
        x = self.fc(x)
        return torch.sigmoid(x)


class CondConv2D(_ConvNd):
    r"""Learn specialized convolutional kernels for each example.
    As described in the paper
    `CondConv: Conditionally Parameterized Convolutions for Efficient Inference`_ ,
    conditionally parameterized convolutions (CondConv),
    which challenge the paradigm of static convolutional kernels
    by computing convolutional kernels as a function of the input.
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        padding_mode (string, optional): ``'zeros'``, ``'reflect'``, ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        num_experts (int): Number of experts per layer
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          .. math::
              H_{out} = \left\lfloor\frac{H_{in}  + 2 \times \text{padding}[0] - \text{dilation}[0]
                        \times (\text{kernel\_size}[0] - 1) - 1}{\text{stride}[0]} + 1\right\rfloor
          .. math::
              W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] - \text{dilation}[1]
                        \times (\text{kernel\_size}[1] - 1) - 1}{\text{stride}[1]} + 1\right\rfloor
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}},`
                         :math:`\text{kernel\_size[0]}, \text{kernel\_size[1]})`.
                         The values of these weights are sampled from
                         :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
        bias (Tensor):   the learnable bias of the module of shape (out_channels). If :attr:`bias` is ``True``,
                         then the values of these weights are
                         sampled from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                         :math:`k = \frac{groups}{C_\text{in} * \prod_{i=0}^{1}\text{kernel\_size}[i]}`
    .. _CondConv: Conditionally Parameterized Convolutions for Efficient Inference:
       https://arxiv.org/abs/1904.04971
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros', num_experts=3, dropout_rate=0.2):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(CondConv2D, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

        self._avg_pooling = functools.partial(F.adaptive_avg_pool2d, output_size=(1, 1))
        self._routing_fn = _routing(in_channels, num_experts, dropout_rate)

        self.weight = Parameter(torch.Tensor(
            num_experts, out_channels, in_channels // groups, *kernel_size))

        self.reset_parameters()

    def _conv_forward(self, input, weight):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._padding_repeated_twice, mode=self.padding_mode),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, inputs):
        b, _, _, _ = inputs.size()
        res = []
        for input in inputs:
            input = input.unsqueeze(0)
            pooled_inputs = self._avg_pooling(input)
            routing_weights = self._routing_fn(pooled_inputs)
            kernels = torch.sum(routing_weights[:, None, None, None, None] * self.weight, 0)
            out = self._conv_forward(input, kernels)
            res.append(out)
        return torch.cat(res, dim=0)


# if __name__ == '__main__':
#     batch_size = 1
#     channels = 64
#     height, width = 64, 64
#     input_tensor = torch.randn(batch_size, channels, height, width)
#
#     layer = CondConv2D(64, 128)
#
#     # 前向传播
#     output_tensor = layer(input_tensor)
#
#     # 打印输入和输出的形状
#     print("输入张量形状:", input_tensor.shape)
#     print("输出张量形状:", output_tensor.shape)

#https://blog.csdn.net/yumaomi/article/details/124898858#:~:text=DFANet%EF%BC%9A
class depthwise_separable_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                                   bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class XceptionABlock(nn.Module):
    """
    Base Block for XceptionA mentioned in DFANet paper.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(XceptionABlock, self).__init__()
        self.conv1 = nn.Sequential(
            depthwise_separable_conv(in_channels, out_channels // 4, stride=stride),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            depthwise_separable_conv(out_channels // 4, out_channels // 4),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
        )
        self.conv3 = nn.Sequential(
            depthwise_separable_conv(out_channels // 4, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)
        identity = self.skip(x)

        return residual + identity


class enc(nn.Module):
    """
    encoder block
    """

    def __init__(self, in_channels, out_channels, stride=2, num_repeat=3):
        super(enc, self).__init__()
        stacks = [XceptionABlock(in_channels, out_channels, stride=2)]
        for x in range(num_repeat - 1):
            stacks.append(XceptionABlock(out_channels, out_channels))
        self.build = nn.Sequential(*stacks)

    def forward(self, x):
        x = self.build(x)
        return x


class ChannelAttention(nn.Module):
    """
        channel attention module
    """

    def __init__(self, in_channels, out_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 1000, bias=False),
            nn.ReLU(),
            nn.Linear(1000, out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SubBranch(nn.Module):
    """
        create 3 Sub Branches in DFANet
        channel_cfg: the chnnels of each enc stage
        branch_index: the index of each sub branch
    """

    def __init__(self, channel_cfg, branch_index):
        super(SubBranch, self).__init__()
        self.enc2 = enc(channel_cfg[0], 48, num_repeat=3)
        self.enc3 = enc(channel_cfg[1], 96, num_repeat=6)
        self.enc4 = enc(channel_cfg[2], 192, num_repeat=3)
        self.fc_atten = ChannelAttention(192, 192)
        self.branch_index = branch_index

    def forward(self, x0, *args):
        out0 = self.enc2(x0)
        if self.branch_index in [1, 2]:
            out1 = self.enc3(torch.cat([out0, args[0]], 1))
            out2 = self.enc4(torch.cat([out1, args[1]], 1))
        else:
            out1 = self.enc3(out0)
            out2 = self.enc4(out1)
        out3 = self.fc_atten(out2)
        return [out0, out1, out2, out3]


class XceptionA(nn.Module):
    """
    channel_cfg: the all channels in each enc blocks
    """

    def __init__(self, channel_cfg, num_classes=33):
        super(XceptionA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.branch = SubBranch(channel_cfg, branch_index=0)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Linear(192, 1000),
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )

    def forward(self, x):
        b, c, _, _ = x.szie()
        x = self.conv1(x)
        _, _, _, x = self.branch(x)
        x = self.avg_pool(x).view(b, -1)
        x = self.classifier(x)

        return x


class DFA_Encoder(nn.Module):
    def __init__(self, channel_cfg):
        super(DFA_Encoder, self).__init__()
        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
            CondConv2D(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU()
        )
        self.branch0 = SubBranch(channel_cfg[0], branch_index=0)
        self.branch1 = SubBranch(channel_cfg[1], branch_index=1)
        self.branch2 = SubBranch(channel_cfg[2], branch_index=2)

    def forward(self, x):
        x = self.conv1(x)
        x0, x1, x2, x5 = self.branch0(x)
        x3 = F.interpolate(x5, x0.size()[2:], mode='bilinear', align_corners=True)

        x1, x2, x3, x6 = self.branch1(torch.cat([x0, x3], 1), x1, x2)
        x4 = F.interpolate(x6, x1.size()[2:], mode='bilinear', align_corners=True)

        x2, x3, x4, x7 = self.branch2(torch.cat([x1, x4], 1), x2, x3)

        return [x0, x1, x2, x5, x6, x7]


class DFA_Decoder(nn.Module):
    """
        the DFA decoder
    """

    def __init__(self, decode_channels, num_classes):
        super(DFA_Decoder, self).__init__()

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=decode_channels, kernel_size=1, bias=False),
            # CondConv2D(in_channels=48, out_channels=decode_channels, kernel_size=1, bias=False),

            nn.BatchNorm2d(decode_channels),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_channels=48, out_channels=decode_channels, kernel_size=1, bias=False),
            CondConv2D(in_channels=48, out_channels=decode_channels, kernel_size=1, bias=False),

            nn.BatchNorm2d(decode_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=48, out_channels=decode_channels, kernel_size=1, bias=False),
            # CondConv2D(in_channels=48, out_channels=decode_channels, kernel_size=1, bias=False),

            nn.BatchNorm2d(decode_channels),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=decode_channels, kernel_size=1, bias=False),
            # CondConv2D(in_channels=48, out_channels=decode_channels, kernel_size=1, bias=False),

            nn.BatchNorm2d(decode_channels),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=decode_channels, kernel_size=1, bias=False),
            # CondConv2D(in_channels=192, out_channels=decode_channels, kernel_size=1, bias=False),

            nn.BatchNorm2d(decode_channels),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=decode_channels, kernel_size=1, bias=False),
            # CondConv2D(in_channels=192, out_channels=decode_channels, kernel_size=1, bias=False),

            nn.BatchNorm2d(decode_channels),
            nn.ReLU()
        )
        self.conv_add = nn.Sequential(
            nn.Conv2d(in_channels=decode_channels, out_channels=decode_channels, kernel_size=3, padding=1, bias=False),
            # CondConv2D(in_channels=decode_channels, out_channels=decode_channels, kernel_size=1, bias=False),

            nn.BatchNorm2d(decode_channels),
            nn.ReLU()
        )
        self.conv_cls = nn.Conv2d(in_channels=decode_channels, out_channels=num_classes, kernel_size=3, padding=1,
                                  bias=False)

    def forward(self, x0, x1, x2, x3, x4, x5):
        x0 = self.conv0(x0)
        x1 = F.interpolate(self.conv1(x1), x0.size()[2:], mode='bilinear', align_corners=True)
        x2 = F.interpolate(self.conv2(x2), x0.size()[2:], mode='bilinear', align_corners=True)
        x3 = F.interpolate(self.conv3(x3), x0.size()[2:], mode='bilinear', align_corners=True)
        x4 = F.interpolate(self.conv5(x4), x0.size()[2:], mode='bilinear', align_corners=True)
        x5 = F.interpolate(self.conv5(x5), x0.size()[2:], mode='bilinear', align_corners=True)

        x_shallow = self.conv_add(x0 + x1 + x2)

        x = self.conv_cls(x_shallow + x3 + x4 + x5)
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        return x


class DFANet(nn.Module):
    def __init__(self, channel_cfg = [[8, 48, 96],[240, 144, 288], [240, 144, 288]], decoder_channel=64, num_classes=1):
        super(DFANet, self).__init__()
        self.encoder = DFA_Encoder(channel_cfg)
        self.decoder = DFA_Decoder(decoder_channel, num_classes)

    def forward(self, x):
        x0, x1, x2, x3, x4, x5 = self.encoder(x)
        x = self.decoder(x0, x1, x2, x3, x4, x5)
        # return x
        return torch.sigmoid(x)


# 可以用以下代码进行测试
import torch

ch_cfg = [[8, 48, 96],[240, 144, 288], [240, 144, 288]]

device = torch.device("cpu")
model = DFANet(ch_cfg, 64, 1).cuda()
# model = model.to(device)
# a = torch.ones([1, 3, 256, 256])
a = torch.randn(1, 3, 256, 256).cuda()
# a = a.to(device)
out = model(a)
print(out.shape)
#https://blog.csdn.net/yumaomi/article/details/124898858#:~:text=DFANet%EF%BC%9A


from thop import profile		 ## 导入thop模块
if __name__ == "__main__":

    input = torch.randn(1, 3, 256, 256).cuda()
    # input = torch.randn(1, 3, size, size)
    # model = DFANet(ch_cfg, 64, 1).cuda()
    model = DFANet(num_classes=1).cuda()

    flops, params = profile(model, inputs=(input,))
    print('flops',flops/1e9)			## 打印计算量
    print('params',params/1e6)			## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))