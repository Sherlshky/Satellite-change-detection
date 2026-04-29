# Automatic Detection of Coastal and Landmark Changes in Satellite Imagery Using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-ffea00)

##  Overview

Change detection in high-resolution satellite imagery is challenging due to complex semantic representations and varied object scales. To address these issues, we propose **CA-Net**, a novel Siamese network leveraging a Transformer-based encoder and advanced interaction mechanisms.

###  Features
* **Siamese SegFormer-B1 Encoder:** Extracts robust, hierarchical feature representations with shared weights.
* **Enhanced Bi-temporal Interaction (BI³):** Integrates **Enhanced LPE** (Parallel multi-scale convolutions + SE attention) and **Stabilized GDFA** to effectively fuse temporal differences and prevent overfitting.
* **Difference Pyramid Block (DPB):** A top-down cascade structure that refines absolute difference maps progressively.
* **Adaptive Multi-scale Processing:** Utilizes `HighResEnhance` for fine details (L1, L2) and `MultiScaleShapeModule` for strong semantics (L3, L4).
* **Multi-level Attention:** Incorporates Channel and Spatial Attention (CBAM) and Pyramid Pooling Modules (PPM) to capture global context.
* **Dynamic Composite Loss:** A multi-stage training strategy combining BCE, Dice, and Lovász-Softmax losses to tackle extreme class imbalance.
* **Data Augmentation Consistency:** Synchronized Bi-temporal `MixUp` and `CutMix` tailored for change detection pairs.

---
## Repository Structure
```text
Project_Root/
├── checkpoint/               # Directory for saved model weights
├── data/
│   └── Levir-cd/             # LEVIR-CD Dataset directory
│       ├── train/            # Train set (A/, B/, label/)
│       ├── val/              # Validation set (A/, B/, label/)
│       └── test/             # Test set (A/, B/, label/)
├── models/                   # Model architectures
│   ├── siamese_SCA_net.py    # Proposed SCA-Net
│   ├── siamese_unet.py       # Siamese U-Net (Baseline)
│   └── siamese_res_unet.py   # Siamese ResUNet (Baseline)
├── results/                  
├── utils/                    # Utility scripts
│   ├── dataset.py            # Custom Dataset & Online Cropping
│   ├── augmentation.py       # Bi-temporal MixUp & CutMix
│   ├── sliding_window.py     # Large-image sliding window inference
│   ├── loss.py               # Dynamic Composite Loss
│   └── metrics.py            # Evaluation metrics (IoU, F1, etc.)
├── train.py                  # Main training script
└── requirements.txt          # Python dependencies
```
## Installation
1. Clone the repository
2. Create a virtual environment (Recommended)
3. Install dependencies
```python
pip install -r requirements.txt
```
## Usage

### 1. Data Preparation
Please download the dataset from the link below:
* **[Download LEVIR-CD Dataset (Google Drive)](https://drive.google.com/drive/folders/1dLuzldMRmbBNKPpUkX8Z53hi6NHLrWim)**

Alternatively, we are planning to host the dataset on HuggingFace soon for easier access. Once downloaded, extract and organize the data under the `./data/Levir-cd/` directory as shown in the Repository Structure above.

1. Training
We provide an automated batch script run.bat that executes the full training pipeline, including baseline models and the SCA-Net ablation study.
```python
run.bat
```
Alternatively, you can run individual models using train.py:
```python
python train.py --model unet --fusion_mode diff
python train.py --model ca_net --ablation full
```
1. Evaluation
```python
python plot_final_comparison.py
```