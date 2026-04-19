# utils/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F

def lovasz_grad(gt_sorted):
    """
    计算 Lovász 扩展的梯度 (基于预测误差的排序)
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge_flat(logits, labels):
    """
    二分类 Lovász hinge loss 核心计算
    """
    if len(labels) == 0:
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * signs)
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm_ground_truth = labels[perm]
    grad = lovasz_grad(perm_ground_truth)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

class DynamicCompositeLoss(nn.Module):
    def __init__(self):
        super(DynamicCompositeLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, logits, labels):
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        labels = labels.view(-1)
        intersection = (probs * labels).sum()
        # 加 1e-6 防止分母为 0
        return 1.0 - (2. * intersection + 1e-6) / (probs.sum() + labels.sum() + 1e-6)

    def lovasz_loss(self, logits, labels):
        # 展平 tensor
        logits = logits.view(-1)
        labels = labels.view(-1)
        return lovasz_hinge_flat(logits, labels)

    def forward(self, logits, labels, w_bce, w_dice, w_lovasz):
        """
        根据传入的动态权重组合三种 Loss
        """
        loss = 0.0
        
        # 为了提高计算效率，如果某个 loss 的权重为 0，则直接跳过该计算
        if w_bce > 0:
            loss += w_bce * self.bce(logits, labels)
        if w_dice > 0:
            loss += w_dice * self.dice_loss(logits, labels)
        if w_lovasz > 0:
            loss += w_lovasz * self.lovasz_loss(logits, labels)
            
        return loss