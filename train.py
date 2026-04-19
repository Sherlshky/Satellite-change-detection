# train.py
import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.dataset import LevirCDDataset
from utils.metrics import Evaluator
from utils.loss import DynamicCompositeLoss
import matplotlib.pyplot as plt

# 导入上一轮编写的数据增强和滑窗推理模块
from utils.augmentation import apply_bitemporal_aug
from utils.sliding_window import sliding_window_predict

# --- Import Models ---
from models.siamese_unet import SiameseUNet
from models.siamese_res_unet import SiameseResUNet  
from models.siamese_SCA_net import SCANet

# --- Config ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_ROOT = './data/Levir-cd' 

def get_args():
    parser = argparse.ArgumentParser(description='Train Change Detection Model')
    # 清理了多余模型，只保留三个核心
    parser.add_argument('--model', type=str, default='sca_net', choices=['unet', 'resunet', 'sca_net'])
    # 基线模型的融合策略
    parser.add_argument('--fusion_mode', type=str, default='diff', choices=['diff', 'concat'])
    # SCA-Net 专用的消融参数
    parser.add_argument('--ablation', type=str, default='full', choices=['base', 'dpb', 'bi3', 'full'], 
                        help="Ablation mode strictly for SCA-Net")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8) 
    parser.add_argument('--lr', type=float, default=1e-4)
    return parser.parse_args()

def train():
    args = get_args()
  
    if args.model == 'sca_net':
        run_name = f"run_{args.model}_{args.ablation}"
    else:
        run_name = f"run_{args.model}_{args.fusion_mode}"
        
    # 2. 第二步：然后再根据 run_name 生成路径并创建文件夹
    checkpoint_dir = os.path.join('./checkpoint', run_name)
    results_dir = os.path.join('./results', run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Initializing Training: Model=[{args.model.upper()}] on {DEVICE}...")
    
    try:
        train_dataset = LevirCDDataset(data_root=DATA_ROOT, split='train')
        val_dataset = LevirCDDataset(data_root=DATA_ROOT, split='val')
    except Exception as e:
        print(f"Dataset Error: {e}")
        return

    # 训练集使用设定 batch_size，验证集强制为 1（因为输出 1024x1024 大图防止 OOM）
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    # 2. Model Selection Logic
    if args.model == 'sca_net':
        run_name = f"run_{args.model}_{args.ablation}"
    else:
        run_name = f"run_{args.model}_{args.fusion_mode}"
        
    checkpoint_dir = os.path.join('./checkpoint', run_name)
    results_dir = os.path.join('./results', run_name)
    # ...

    # 模型选择逻辑精简
    if args.model == 'unet':
        model = SiameseUNet(fusion_mode=args.fusion_mode).to(DEVICE)
    elif args.model == 'resunet':
        model = SiameseResUNet(fusion_mode=args.fusion_mode).to(DEVICE)
    elif args.model == 'sca_net':
        model = SCANet(ablation=args.ablation).to(DEVICE)

    # 3. Optimization
    criterion = DynamicCompositeLoss().to(DEVICE) 
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    evaluator = Evaluator(num_class=2)
    
    history = {'train_loss': [], 'val_iou': [], 'val_f1': [], 'val_precision': [], 'val_recall': [], 'val_accuracy': []}

    print(f"Start Training ({args.epochs} epochs)...")
    best_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        
        w_bce, w_dice, w_lovasz = get_dynamic_loss_weights(epoch, args.epochs)
        
        for i, (img_A, img_B, label) in enumerate(train_loader):
            img_A, img_B, label = img_A.to(DEVICE), img_B.to(DEVICE), label.to(DEVICE)
            
            # --- 加入数据增强 (MixUp/CutMix) ---
            img_A, img_B, label = apply_bitemporal_aug(img_A, img_B, label, alpha=0.2, prob=0.5)
            label = label.float() # 防止增强产生软标签报错
            
            optimizer.zero_grad()
            outputs = model(img_A, img_B)
            
            loss = criterion(outputs, label, w_bce, w_dice, w_lovasz)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        
        # --- Validation (大图滑窗推理) ---
        metrics = validate(model, val_loader, evaluator, epoch, results_dir)
        
        history['train_loss'].append(avg_loss)
        history['val_iou'].append(metrics['IoU'])
        history['val_f1'].append(metrics['F1'])
        history['val_precision'].append(metrics['Precision'])
        history['val_recall'].append(metrics['Recall'])
        history['val_accuracy'].append(metrics['Accuracy'])
        
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.4f} | F1: {metrics['F1']:.4f} | P: {metrics['Precision']:.4f} | R: {metrics['Recall']:.4f}")
        
        if metrics['F1'] > best_f1:
            best_f1 = metrics['F1']
            # 这里统一用 run_name，防止名字错乱
            best_name = f'best_{run_name}.pth' 
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, best_name))
        if (epoch + 1) % 10 == 0:
            # 这里也统一用 run_name
            epoch_name = f'epoch_{epoch+1}_{run_name}.pth'
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, epoch_name))
    
    history_path = os.path.join(results_dir, 'history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f)
    
    plot_single_curve(history, args.model, results_dir)

def get_dynamic_loss_weights(epoch, total_epochs):
    ratio = epoch / total_epochs
    if ratio < 0.2:
        return 1.0, 0.0, 0.0 
    elif ratio < 0.4:
        return 0.5, 0.5, 0.0
    elif ratio < 0.6:
        return 0.2, 0.2, 0.6
    else:
        return 0.34, 0.33, 0.33
    
def validate(model, loader, evaluator, epoch, save_dir):
    model.eval()
    evaluator.reset()
    with torch.no_grad():
        for i, (img_A, img_B, label) in enumerate(loader):
            img_A, img_B, label = img_A.to(DEVICE), img_B.to(DEVICE), label.to(DEVICE)
            
            # --- 使用滑窗推理处理 1024x1024 验证集图片 ---
            preds = sliding_window_predict(model, img_A, img_B, crop_size=256, stride=256)
            
            evaluator.add_batch(label.squeeze(1), preds.squeeze(1))
            
    return evaluator.get_metrics()

def plot_single_curve(history, model_name, save_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.title(f'{model_name.upper()} Loss')
    plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_f1'], label='Val F1-Score', color='orange')
    plt.title(f'{model_name.upper()} F1 Score')
    plt.xlabel('Epochs'); plt.ylabel('F1'); plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_curves.png'))
    plt.close()

if __name__ == '__main__':
    train()