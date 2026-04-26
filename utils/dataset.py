# utils/dataset.py
import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class LevirCDDataset(Dataset):
    def __init__(self, data_root, split='train', crop_size=256):
        """
        :param data_root: 
        :param split: 'train', 'val', or 'test'
        :param crop_size:  256
        """
        self.data_root = data_root
        self.split = split
        self.crop_size = crop_size
        
        self.img_dir_A = os.path.join(data_root, split, 'A')
        self.img_dir_B = os.path.join(data_root, split, 'B')
        self.label_dir = os.path.join(data_root, split, 'label')
        
        self.image_files = sorted(os.listdir(self.img_dir_A))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        

        img_A = Image.open(os.path.join(self.img_dir_A, img_name)).convert('RGB')
        img_B = Image.open(os.path.join(self.img_dir_B, img_name)).convert('RGB')
        label = Image.open(os.path.join(self.label_dir, img_name)).convert('L')
        

        if self.split == 'train':
            w, h = img_A.size
            th, tw = self.crop_size, self.crop_size
            

            if w == tw and h == th:
                i, j = 0, 0
            else:
                i = random.randint(0, h - th)
                j = random.randint(0, w - tw)
                

            img_A = TF.crop(img_A, i, j, th, tw)
            img_B = TF.crop(img_B, i, j, th, tw)
            label = TF.crop(label, i, j, th, tw)

            

        img_A = TF.to_tensor(img_A)
        img_B = TF.to_tensor(img_B)
        label = TF.to_tensor(label)
        

        label = (label > 0.5).float()
        
        return img_A, img_B, label