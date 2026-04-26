# utils/metrics.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class Evaluator:
    def __init__(self, num_class=2):
        self.num_class = num_class
        self.confusion_matrix = torch.zeros((self.num_class,)*2)

    def reset(self):
        self.confusion_matrix = torch.zeros((self.num_class,)*2)

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        pre_image = pre_image.flatten()
        gt_image = gt_image.flatten()

        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask].long() + pre_image[mask].long()
        count = torch.bincount(label, minlength=self.num_class**2)
        self.confusion_matrix += count.reshape(self.num_class, self.num_class).cpu()

    def get_metrics(self):
        cm = self.confusion_matrix
        TN = cm[0, 0]
        FP = cm[0, 1]
        FN = cm[1, 0]
        TP = cm[1, 1]

        precision = TP / (TP + FP + 1e-6)
        recall = TP / (TP + FN + 1e-6)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-6)
        iou = TP / (TP + FP + FN + 1e-6)
        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-6) 

        return {
            "IoU": iou.item(),
            "F1": f1_score.item(),
            "Precision": precision.item(),
            "Recall": recall.item(),
            "Accuracy": accuracy.item()
        }
    
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.9, dice_weight=0.1):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        
        bce_loss = self.bce(inputs, targets)
        
        
        inputs_sigmoid = torch.sigmoid(inputs)
        
        
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)
        
        
        intersection = (inputs_flat * targets_flat).sum()
        dice_score = (2. * intersection + 1e-6) / (inputs_flat.sum() + targets_flat.sum() + 1e-6)
        dice_loss = 1.0 - dice_score
        
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss