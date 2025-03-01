import torch.nn as nn
import torch.nn.functional as F
import math
import torch.utils.model_zoo as model_zoo
import torch
import numpy as np

affine_par = True  # True: BN has learnable affine parameters, False: without learnable affine parameters of BatchNorm Layer


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):

    def __init__(self, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(2048, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class Residual_Covolution(nn.Module):
    def __init__(self, icol, ocol, num_classes):
        super(Residual_Covolution, self).__init__()
        self.conv1 = nn.Conv2d(icol, ocol, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.conv2 = nn.Conv2d(ocol, num_classes, kernel_size=3, stride=1, padding=12, dilation=12, bias=True)
        self.conv3 = nn.Conv2d(num_classes, ocol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.conv4 = nn.Conv2d(ocol, icol, kernel_size=1, stride=1, padding=0, dilation=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        dow1 = self.conv1(x)
        dow1 = self.relu(dow1)
        seg = self.conv2(dow1)
        inc1 = self.conv3(seg)
        add1 = dow1 + self.relu(inc1)
        inc2 = self.conv4(add1)
        out = x + self.relu(inc2)
        return out, seg


class PSPModule(nn.Module):
    """Ref: Pyramid Scene Parsing Network,CVPR2017, http://arxiv.org/abs/1612.01105 """

    def __init__(self, inChannel, midReduction=4, outChannel=512, sizes=(1, 2, 3, 6)):
        super(PSPModule, self).__init__()
        self.midChannel = int(inChannel / midReduction)  # 1x1Conv channel num, defalut=512
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(inChannel, self.midChannel, size) for size in sizes])  # pooling->conv1x1
        self.bottleneck = nn.Conv2d((inChannel + self.midChannel * 4), outChannel,
                                    kernel_size=3)  # channel: 4096->512 1x1
        self.bn = nn.BatchNorm2d(outChannel)
        self.prelu = nn.PReLU()

    def _make_stage(self, inChannel, midChannel, size):
        pooling = nn.AdaptiveAvgPool2d(output_size=(size, size))
        Conv = nn.Conv2d(inChannel, midChannel, kernel_size=1, bias=False)
        return nn.Sequential(pooling, Conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        mulBranches = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [
            feats]  # four parallel baranches
        out = self.bottleneck(torch.cat((mulBranches[0], mulBranches[1], mulBranches[2], mulBranches[3], feats), 1))
        out = self.bn(out)
        out = self.prelu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, midReduction=4):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)

        # PSPModule
        self.pspmodule = PSPModule(inChannel=512 * block.expansion, midReduction=midReduction, outChannel=512,
                                   sizes=(1, 2, 3, 6))
        self.spatial_drop = nn.Dropout2d(p=0.1)

        # auxiliary classifier followd res4, I think the auxiliary is not neccesary!!!
        # main classifier
        self.main_classifier = nn.Conv2d(512, num_classes, kernel_size=1)
        # self.softmax = nn.Softmax()

        # init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        #        for i in m.parameters():
        #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.conv1(x)  # 7x7Conv
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)  # res2
        x = self.layer2(x)  # res3
        x = self.layer3(x)  # res4
        x = self.layer4(x)  # res5

        # PSPModule
        # print("before PSPModule, tensor size:", x.size())
        x = self.pspmodule(x)

        # print("after PSPModule, tensor size:", x.size())
        x = self.spatial_drop(x)

        # classifier 
        x = self.main_classifier(x)

        # print("before upsample, tensor size:", x.size())
        x = F.upsample(x, input_size, mode='bilinear')  # upsample to the size of input image, scale=8
        # print("after upsample, tensor size:", x.size())
        # x = self.softmax(x)
        return x  # shape: NxCxHxW, (H,W) is equal to that of input image


def PSPNet(num_classes=1):
    """ """
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model


# if __name__ == "__main__":
#     network = PSPNet()
#     print(network)



from thop import profile		 ## 导入thop模块
if __name__ == "__main__":

    input = torch.randn(1, 3, 256, 256).cuda()
    # input = torch.randn(1, 3, size, size)
    model = PSPNet().cuda()

    flops, params = profile(model, inputs=(input,))
    print('flops',flops/1e9)			## 打印计算量
    print('params',params/1e6)			## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))

#https://github.com/wutianyiRosun/CGNet/blob/master/model/PSPNet.py