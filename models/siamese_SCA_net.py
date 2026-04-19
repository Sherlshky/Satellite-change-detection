# models/siamese_SCA_net.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerModel

# ==========================================
# 1. 基础注意力机制 (CBAM & PPM)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return x * self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(x_cat))

class CBAM(nn.Module):
    """即文档中的 CSAGate"""
    def __init__(self, planes):
        super().__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
    def forward(self, x):
        return self.sa(self.ca(x))

class PPM(nn.Module):
    """金字塔池化模块"""
    def __init__(self, channels):
        super().__init__()
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1), nn.AdaptiveAvgPool2d(2),
            nn.AdaptiveAvgPool2d(3), nn.AdaptiveAvgPool2d(6)
        ])
        self.conv = nn.Conv2d(channels, channels // 4, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        h, w = x.shape[2:]
        feats = [x]
        for pool in self.pools:
            p = F.interpolate(self.conv(pool(x)), size=(h, w), mode='bilinear', align_corners=True)
            feats.append(p)
        return self.fuse(torch.cat(feats, dim=1))

# ==========================================
# 2. 自适应多尺度处理 (高分/低分)
# ==========================================
class HighResEnhance(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=1, groups=channels)
        self.pw = nn.Conv2d(channels, channels, 1)
        self.ca = ChannelAttention(channels)
        self.bn_relu = nn.Sequential(nn.BatchNorm2d(channels), nn.ReLU(inplace=True))

    def forward(self, x):
        res = x
        x = self.bn_relu(self.pw(self.dw(x)))
        x = self.ca(x)
        return x + res

class MultiScaleShapeModule(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.d1 = nn.Conv2d(channels, channels // 4, 3, padding=1, dilation=1)
        self.d2 = nn.Conv2d(channels, channels // 4, 3, padding=2, dilation=2)
        self.d3 = nn.Conv2d(channels, channels // 4, 3, padding=3, dilation=3)
        self.dir_h = nn.Conv2d(channels, channels // 8, (1, 5), padding=(0, 2))
        self.dir_v = nn.Conv2d(channels, channels // 8, (5, 1), padding=(2, 0))
        self.fuse = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.BatchNorm2d(channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = torch.cat([self.d1(x), self.d2(x), self.d3(x), self.dir_h(x), self.dir_v(x)], dim=1)
        return self.fuse(out)

# ==========================================
# 3. 增强型 Bi-temporal Interaction 层
# ==========================================
class EnhancedLPE(nn.Module):
    """增强型局部感知增强: 多尺度并行卷积 + SE注意力"""
    def __init__(self, channels):
        super().__init__()
        self.conv3x3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv5x1 = nn.Conv2d(channels, channels, (5, 1), padding=(2, 0))
        self.conv1x5 = nn.Conv2d(channels, channels, (1, 5), padding=(0, 2))
        
        # SE Attention
        self.se_avg = nn.AdaptiveAvgPool2d(1)
        self.se_fc = nn.Sequential(
            nn.Linear(channels * 3, (channels * 3) // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear((channels * 3) // 16, channels * 3, bias=False),
            nn.Sigmoid()
        )
        self.proj = nn.Conv2d(channels * 3, channels, 1)

    def forward(self, x):
        x1, x2, x3 = self.conv3x3(x), self.conv5x1(x), self.conv1x5(x)
        out = torch.cat([x1, x2, x3], dim=1)
        
        # Apply SE
        b, c, _, _ = out.size()
        y = self.se_avg(out).view(b, c)
        y = self.se_fc(y).view(b, c, 1, 1)
        out = out * y.expand_as(out)
        
        return x + self.proj(out)

class StabilizedGDFA(nn.Module):
    """稳定化全局差异融合注意力"""
    def __init__(self, channels):
        super().__init__()
        # 当通道数 > 128 时，内部降维以避免数值不稳定
        inter_channels = channels // 2 if channels > 128 else channels
        
        self.proj = nn.Conv2d(channels * 2, inter_channels, 1)
        self.attn = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_channels, channels, 1),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout2d(0.1) # 加入 Dropout 防止过拟合

    def forward(self, fa, fb):
        concat = torch.cat([fa, fb], dim=1)
        attn_map = self.attn(self.proj(concat))
        diff = torch.abs(fa - fb)
        return self.dropout(diff * attn_map)

# ==========================================
# 4. 差异金字塔块 (Top-Down DPB)
# ==========================================
class TopDownDPB(nn.Module):
    def __init__(self, dims):
        super().__init__()
        # 1x1 卷积初步细化
        self.refine1x1 = nn.ModuleList([nn.Conv2d(dim, dim, 1) for dim in dims])
        # 3x3 卷积平滑
        self.smooth3x3 = nn.ModuleList([nn.Conv2d(dims[i], dims[i], 3, padding=1) for i in range(len(dims)-1)])
        # 上采样通道对齐
        self.up_projs = nn.ModuleList([nn.Conv2d(dims[i+1], dims[i], 1) for i in range(len(dims)-1)])

    def forward(self, diffs):
        # 1. 初步细化
        refined = [self.refine1x1[i](diffs[i]) for i in range(len(diffs))]
        out = [None] * len(diffs)
        out[-1] = refined[-1] # 最深层保持不变
        
        # 2. 从最深层开始，将细化后的差异图上采样后与上一层差异图相加融合
        for i in range(len(diffs)-2, -1, -1):
            up = F.interpolate(out[i+1], size=refined[i].shape[2:], mode='bilinear', align_corners=True)
            up = self.up_projs[i](up)
            fused = refined[i] + up
            out[i] = self.smooth3x3[i](fused) # 融合后平滑
            
        return out

# ==========================================
# 5. SCA-Net 主网络
# ==========================================
class SCANet(nn.Module):
    def __init__(self, num_classes=1, ablation='full'):
        """
        ablation 模式说明:
        - 'base': 仅 SegFormer 骨干 + 直接做绝对差异(Diff) + 解码器
        - 'dpb': base + 引入自顶向下的 DPB (差异金字塔融合)
        - 'bi3': base + DPB + 引入 BI³ (Enhanced LPE + Stabilized GDFA)
        - 'full': 完整版 SCA-Net (包含多尺度自适应处理、CBAM和PPM)
        """
        super().__init__()
        self.ablation = ablation
        
        # 1. Siamese 编码器 (SegFormer-B1)
        self.backbone = SegformerModel.from_pretrained("nvidia/mit-b1")
        dims = [64, 128, 320, 512]
        
        # 按需初始化模块，节省显存
        if self.ablation == 'full':
            self.hr_enhance = nn.ModuleList([HighResEnhance(dims[0]), HighResEnhance(dims[1])])
            self.shape_mod = nn.ModuleList([MultiScaleShapeModule(dims[2]), MultiScaleShapeModule(dims[3])])
            self.cbam_gates = nn.ModuleList([CBAM(dim) for dim in dims])
            self.ppm = PPM(dims[3])
            
        if self.ablation in ['bi3', 'full']:
            self.lpe = nn.ModuleList([EnhancedLPE(dim) for dim in dims])
            self.gdfa = nn.ModuleList([StabilizedGDFA(dim) for dim in dims])
            
        if self.ablation in ['dpb', 'bi3', 'full']:
            self.dpb = TopDownDPB(dims)
        
        # 6. 解码器 (全模式通用)
        self.up_conv3 = nn.Conv2d(dims[3] + dims[2], dims[2], 3, padding=1)
        self.up_conv2 = nn.Conv2d(dims[2] + dims[1], dims[1], 3, padding=1)
        self.up_conv1 = nn.Conv2d(dims[1] + dims[0], dims[0], 3, padding=1)

        self.out_up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.out_conv = nn.Conv2d(dims[0], num_classes, kernel_size=1)

    def forward(self, img_A, img_B):
        # 1. 骨干特征提取
        outputs_A = self.backbone(img_A, output_hidden_states=True).hidden_states
        outputs_B = self.backbone(img_B, output_hidden_states=True).hidden_states
        
        # --- 消融开关: 自适应多尺度处理 ---
        if self.ablation == 'full':
            feats_A = [self.hr_enhance[i](outputs_A[i]) if i<2 else self.shape_mod[i-2](outputs_A[i]) for i in range(4)]
            feats_B = [self.hr_enhance[i](outputs_B[i]) if i<2 else self.shape_mod[i-2](outputs_B[i]) for i in range(4)]
        else:
            feats_A, feats_B = outputs_A, outputs_B
            
        # --- 消融开关: BI³ (LPE + GDFA) ---
        if self.ablation in ['bi3', 'full']:
            lpe_A = [self.lpe[i](feats_A[i]) for i in range(4)]
            lpe_B = [self.lpe[i](feats_B[i]) for i in range(4)]
            D_raw = [self.gdfa[i](lpe_A[i], lpe_B[i]) for i in range(4)]
        else:
            # 未开启 BI³ 时，退化为简单的绝对差异 (Diff)
            D_raw = [torch.abs(feats_A[i] - feats_B[i]) for i in range(4)]
            
        # --- 消融开关: DPB 差异金字塔融合 ---
        if self.ablation in ['dpb', 'bi3', 'full']:
            D_fused = self.dpb(D_raw)
        else:
            D_fused = D_raw # 未开启 DPB 时，直接穿透
            
        # --- 消融开关: 多级注意力 (CBAM + PPM) ---
        if self.ablation == 'full':
            F_refined = [self.cbam_gates[i](D_fused[i]) for i in range(4)]
            F_refined[3] = self.ppm(F_refined[3])
        else:
            F_refined = D_fused

        # 解码器自顶向下融合
        x = F_refined[3]
        x = F_refined[2] + self.up_conv3(torch.cat([F_refined[2], F.interpolate(x, size=F_refined[2].shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x = F_refined[1] + self.up_conv2(torch.cat([F_refined[1], F.interpolate(x, size=F_refined[1].shape[2:], mode='bilinear', align_corners=True)], dim=1))
        x = F_refined[0] + self.up_conv1(torch.cat([F_refined[0], F.interpolate(x, size=F_refined[0].shape[2:], mode='bilinear', align_corners=True)], dim=1))
        
        logits = self.out_conv(self.out_up(x))
        return logits