import torch
import torch.nn as nn

class OCRNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OCRNet, self).__init__()
        # 编码器部分
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 对象上下文模块
        self.object_context = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        # 编码器
        x1 = self.encoder(x)
        # 对象上下文
        x2 = self.object_context(x1)
        # 解码器
        x3 = self.decoder(x2)
        # return x3
        return torch.sigmoid(x3)


# 创建模型实例
model = OCRNet(in_channels=3, out_channels=1)
print(model)

from thop import profile		 ## 导入thop模块
if __name__ == "__main__":

    input = torch.randn(1, 3, 256, 256)
    # input = torch.randn(1, 3, size, size)
    # model = UnetPlusPlus(num_classes=1, deep_supervision=False)
    model = OCRNet(in_channels=3, out_channels=1)

    flops, params = profile(model, inputs=(input,))
    print('flops',flops/1e9)			## 打印计算量
    print('params',params/1e6)			## 打印参数量

    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total/1e6))