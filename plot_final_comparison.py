# plot_final_comparison.py
import json
import matplotlib.pyplot as plt
import os

def resolve_history_path(base_name):

    paths_to_try = [
        f'./results/{base_name}/history.json',
        f'./results/{base_name}_diff/history.json',
        f'./results/{base_name}_concat/history.json'
    ]
    for p in paths_to_try:
        if os.path.exists(p):
            return p
    print(f"error")
    return None

def compare():
    
    models_config = {
        'run_unet_diff':      {'label': 'Siam-UNet (Baseline)', 'style': '--', 'color': '#1f77b4'}, 
        'run_resunet_concat': {'label': 'Siam-ResUNet (Strong)', 'style': '-.', 'color': '#ff7f0e'}, 
        'run_sca_net_full':   {'label': 'SCA-Net (Ours)',       'style': '-',  'color': '#d62728'}
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Overall Model Performance Comparison (Test Set)', fontsize=20, fontweight='bold', y=0.98)

    metrics_to_plot = [
        ('train_loss', 'Training Loss', axes[0, 0]),
        ('val_iou', 'Validation IoU', axes[0, 1]),
        ('val_f1', 'Validation F1-Score', axes[0, 2]),
        ('val_precision', 'Validation Precision', axes[1, 0]),
        ('val_recall', 'Validation Recall', axes[1, 1]),
        ('val_accuracy', 'Validation Accuracy', axes[1, 2])
    ]

    valid_models_count = 0

    for base_name, config in models_config.items():
        hist_path = resolve_history_path(base_name)
        if hist_path is None:
            continue
            
        with open(hist_path, 'r') as f:
            hist = json.load(f)

        valid_models_count += 1
        epochs = range(1, len(hist['train_loss']) + 1)
        lw = 2.5 if 'Ours' in config['label'] else 2.0 

        for metric_key, title, ax in metrics_to_plot:
            if metric_key in hist:
                ax.plot(epochs, hist[metric_key], linestyle=config['style'], 
                        color=config['color'], label=config['label'], linewidth=lw, alpha=0.9)

    if valid_models_count == 0:
        print("no config")
        return

    for metric_key, title, ax in metrics_to_plot:
        ax.set_title(title, fontsize=15, fontweight='bold')
        ax.set_xlabel('Epochs', fontsize=13)
        ax.set_ylabel('Score' if 'loss' not in title.lower() else 'Loss', fontsize=13)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.tick_params(axis='both', which='major', labelsize=11)
        
        if 'loss' in title.lower():
            ax.legend(loc='upper right', fontsize=11)
        else:
            ax.legend(loc='lower right', fontsize=11)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = './results/final_thesis_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nsaved to: {save_path}")

if __name__ == '__main__':
    compare()