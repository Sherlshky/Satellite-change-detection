# models/siamese_res_unet.py
import torch
import torch.nn as nn
import torchvision.models as models

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
    def __init__(self, n_classes=1, pretrained=False, fusion_mode='concat'): 
        super(SiameseResUNet, self).__init__()
        self.fusion_mode = fusion_mode
        

        resnet = models.resnet34(weights=None)
        

        self.encoder0 = DoubleConv(3, 64) 
        

        self.pool = nn.MaxPool2d(2, 2)
        

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2 #  64x64，c 128
        self.encoder3 = resnet.layer3 #  64x64， 32x32，c 256
        self.encoder4 = resnet.layer4 #  32x32， 16x16，c 512
        
        bottleneck_in = 512 if fusion_mode == 'diff' else 1024
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(bottleneck_in, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.up4 = ResUpBlock(512, 256, 256, fusion_mode) # 16x16  32x32
        self.up3 = ResUpBlock(256, 128, 128, fusion_mode) # 32x32  64x64
        self.up2 = ResUpBlock(128, 64, 64, fusion_mode)   # 64x64  128x128
        self.up1 = ResUpBlock(64, 64, 64, fusion_mode)    # 128x128  256x256
        

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
        x = self.up1(x, a0, b0) 
        
        logits = self.out_conv(x)
        return logits