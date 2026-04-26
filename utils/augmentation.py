# utils/augmentation.py
import torch
import numpy as np

def rand_bbox(size, lam):
    """
    bounding box
    """
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)


    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def mixup_data(img_A, img_B, label, alpha=0.2):

    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = img_A.size()[0]
    

    index = torch.randperm(batch_size).to(img_A.device)


    mixed_A = lam * img_A + (1 - lam) * img_A[index, :]
    mixed_B = lam * img_B + (1 - lam) * img_B[index, :]
    
    mixed_label = lam * label + (1 - lam) * label[index, :]
    
    return mixed_A, mixed_B, mixed_label

def cutmix_data(img_A, img_B, label, alpha=0.2):

    lam = np.random.beta(alpha, alpha)
    batch_size = img_A.size()[0]
    index = torch.randperm(batch_size).to(img_A.device)

    bbx1, bby1, bbx2, bby2 = rand_bbox(img_A.size(), lam)
    
    mixed_A = img_A.clone()
    mixed_B = img_B.clone()
    mixed_label = label.clone()
    

    mixed_A[:, :, bbx1:bbx2, bby1:bby2] = img_A[index, :, bbx1:bbx2, bby1:bby2]
    mixed_B[:, :, bbx1:bbx2, bby1:bby2] = img_B[index, :, bbx1:bbx2, bby1:bby2]
    mixed_label[:, :, bbx1:bbx2, bby1:bby2] = label[index, :, bbx1:bbx2, bby1:bby2]
    
    return mixed_A, mixed_B, mixed_label

def apply_bitemporal_aug(img_A, img_B, label, alpha=0.2, prob=0.5):

    if np.random.rand() < prob:
        if np.random.rand() < 0.5:
            return mixup_data(img_A, img_B, label, alpha)
        else:
            return cutmix_data(img_A, img_B, label, alpha)
    
    return img_A, img_B, label