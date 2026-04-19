# compare.py
import json
import matplotlib.pyplot as plt
import os

def load_history(run_name):
    path = f'./results/{run_name}/history.json'
    if not os.path.exists(path):
        print(f"Warning: History not found for {run_name} at {path}")
        return None
    with open(path, 'r') as f:
        return json.load(f)

def compare():
    # 配置你的 4 组消融实验
    # 采用同色系对比：红色系给 U-Net，蓝色系给 ResUNet；实线给 Diff，虚线给 Concat
    models_config = {
        'run_unet_diff':    {'label': 'U-Net (Absolute Diff)', 'style': '-',  'color': '#d62728'}, # 红色实线
        'run_unet_concat':  {'label': 'U-Net (Concat)',        'style': '--', 'color': '#ff7f0e'}, # 橙色虚线
        'run_resunet_diff': {'label': 'ResUNet (Absolute Diff)','style': '-',  'color': '#1f77b4'}, # 蓝色实线
        'run_resunet_concat':{'label': 'ResUNet (Concat)',      'style': '--', 'color': '#2ca02c'}  # 绿色虚线
    }

    # 创建 2x3 的六宫格画布
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Ablation Study: Feature Fusion Strategies (Diff vs Concat)', fontsize=18, fontweight='bold', y=0.98)

    metrics_to_plot = [
        ('train_loss', 'Training Loss', axes[0, 0]),
        ('val_iou', 'Validation IoU', axes[0, 1]),
        ('val_f1', 'Validation F1-Score', axes[0, 2]),
        ('val_precision', 'Validation Precision', axes[1, 0]),
        ('val_recall', 'Validation Recall', axes[1, 1]),
        ('val_accuracy', 'Validation Accuracy', axes[1, 2])
    ]

    valid_models_count = 0

    for run_name, config in models_config.items():
        hist = load_history(run_name)
        if hist is None:
            continue

        valid_models_count += 1
        epochs = range(1, len(hist['train_loss']) + 1)

        # 绘图
        for metric_key, title, ax in metrics_to_plot:
            if metric_key in hist:
                ax.plot(epochs, hist[metric_key], linestyle=config['style'], 
                        color=config['color'], label=config['label'], linewidth=2.5, alpha=0.9)

    if valid_models_count == 0:
        print("No training histories found. Please run the training scripts first.")
        return

    # 设置网格、标题和图例
    for metric_key, title, ax in metrics_to_plot:
        ax.set_title(title, fontsize=14)
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('Score' if 'loss' not in title.lower() else 'Loss', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # 调整图例位置
        if 'loss' in title.lower():
            ax.legend(loc='upper right', fontsize=10)
        else:
            ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = './results/ablation_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nAblation comparison plot saved to {save_path}")

if __name__ == '__main__':
    compare()