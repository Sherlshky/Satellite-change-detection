# models/siamese_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
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

class UpFusion(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, fusion_mode='concat'):
        super().__init__()
        self.fusion_mode = fusion_mode
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 根据融合策略计算输入通道数
        if self.fusion_mode == 'diff':
            # 只接收 upsampled_feature + |skip_A - skip_B|
            conv_in = in_channels + skip_channels
        elif self.fusion_mode == 'concat':
            # 接收 upsampled_feature + skip_A + skip_B
            conv_in = in_channels + skip_channels * 2
            
        self.conv = DoubleConv(conv_in, out_channels)

    def forward(self, x_deep, skip_A, skip_B):
        x_up = self.up(x_deep)
        
        if self.fusion_mode == 'diff':
            skip_fused = torch.abs(skip_A - skip_B)
            x = torch.cat([x_up, skip_fused], dim=1)
        elif self.fusion_mode == 'concat':
            x = torch.cat([x_up, skip_A, skip_B], dim=1)
            
        return self.conv(x)

class SiameseUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, fusion_mode='diff'):
        super(SiameseUNet, self).__init__()
        self.fusion_mode = fusion_mode
        
        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))

        # Decoder Bottleneck 处理
        bottleneck_out = 1024 if fusion_mode == 'diff' else 2048
        self.bottleneck_conv = DoubleConv(bottleneck_out, 1024)

        # Decoder
        self.up1 = UpFusion(1024, 512, 512, fusion_mode)
        self.up2 = UpFusion(512, 256, 256, fusion_mode)
        self.up3 = UpFusion(256, 128, 128, fusion_mode)
        self.up4 = UpFusion(128, 64, 64, fusion_mode)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward_one(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5

    def forward(self, x_A, x_B):
        x1_A, x2_A, x3_A, x4_A, x5_A = self.forward_one(x_A)
        x1_B, x2_B, x3_B, x4_B, x5_B = self.forward_one(x_B)

        # 修复了原来直接 max() 导致的差异丢失问题
        if self.fusion_mode == 'diff':
            bottleneck = torch.abs(x5_A - x5_B)
        else:
            bottleneck = torch.cat([x5_A, x5_B], dim=1)
            
        x = self.bottleneck_conv(bottleneck)
        
        x = self.up1(x, x4_A, x4_B)
        x = self.up2(x, x3_A, x3_B)
        x = self.up3(x, x2_A, x2_B)
        x = self.up4(x, x1_A, x1_B)
        
        logits = self.outc(x)
        return logits