# models/siamese_res_unet.py
import torch
import torch.nn as nn
import torchvision.models as models

class DoubleConv(nn.Module):
    """标准的 U-Net 式连续两次卷积，用于替换 ResNet 原生的 7x7 卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, fusion_mode='diff'):
        super(ResUpBlock, self).__init__()
        self.fusion_mode = fusion_mode
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        conv_in = in_channels + skip_channels if fusion_mode == 'diff' else in_channels + skip_channels * 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(conv_in, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip_A, skip_B):
        x = self.up(x)
        if self.fusion_mode == 'diff':
            skip_fused = torch.abs(skip_A - skip_B)
            x = torch.cat([x, skip_fused], dim=1)
        elif self.fusion_mode == 'concat':
            x = torch.cat([x, skip_A, skip_B], dim=1)
        return self.conv(x)

class SiameseResUNet(nn.Module):
    def __init__(self, n_classes=1, pretrained=False, fusion_mode='concat'): # 默认关闭预训练
        super(SiameseResUNet, self).__init__()
        self.fusion_mode = fusion_mode
        
        # 不加载预训练权重，从零开始学习遥感特征
        resnet = models.resnet34(weights=None)
        
        # --- 核心改动：前端 U-Net 化 ---
        # 替换原本的 7x7 卷积，使用 3x3 DoubleConv，保持输入分辨率 (256x256)
        self.encoder0 = DoubleConv(3, 64) 
        
        # 将池化层单独剥离，放在 encoder1 之前，实现第一次下采样
        self.pool = nn.MaxPool2d(2, 2)
        
        # --- 后续保持 ResNet 残差结构 ---
        self.encoder1 = resnet.layer1 # 输入 128x128，输出 128x128，通道 64
        self.encoder2 = resnet.layer2 # 输入 128x128，输出 64x64，通道 128
        self.encoder3 = resnet.layer3 # 输入 64x64，输出 32x32，通道 256
        self.encoder4 = resnet.layer4 # 输入 32x32，输出 16x16，通道 512
        
        bottleneck_in = 512 if fusion_mode == 'diff' else 1024
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(bottleneck_in, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.up4 = ResUpBlock(512, 256, 256, fusion_mode) # 16x16 升到 32x32
        self.up3 = ResUpBlock(256, 128, 128, fusion_mode) # 32x32 升到 64x64
        self.up2 = ResUpBlock(128, 64, 64, fusion_mode)   # 64x64 升到 128x128
        self.up1 = ResUpBlock(64, 64, 64, fusion_mode)    # 128x128 升到 256x256
        
        # 注意：因为前端分辨率没有被 7x7 卷积减半，up1 输出的特征图已经是 256x256 了。
        # 所以我们这里直接去掉了原来代码里多余的 self.out_up！直接用 1x1 卷积输出概率图。
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward_one(self, x):
        e0 = self.encoder0(x)             # 256x256
        e1 = self.encoder1(self.pool(e0)) # 128x128
        e2 = self.encoder2(e1)            # 64x64
        e3 = self.encoder3(e2)            # 32x32
        e4 = self.encoder4(e3)            # 16x16
        return e0, e1, e2, e3, e4

    def forward(self, img_A, img_B):
        a0, a1, a2, a3, a4 = self.forward_one(img_A)
        b0, b1, b2, b3, b4 = self.forward_one(img_B)
        
        if self.fusion_mode == 'diff':
            bottleneck = torch.abs(a4 - b4)
        else:
            bottleneck = torch.cat([a4, b4], dim=1)
            
        x = self.bottleneck_conv(bottleneck)
        
        x = self.up4(x, a3, b3)
        x = self.up3(x, a2, b2)
        x = self.up2(x, a1, b1)
        x = self.up1(x, a0, b0) # 此时 x 已经是 256x256
        
        logits = self.out_conv(x)
        return logits