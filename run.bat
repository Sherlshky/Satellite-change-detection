@echo off
title LEVIR-CD Training Pipeline
color 0A

echo ===================================================
echo     LEVIR-CD Deep Learning Training Pipeline
echo ===================================================
echo.
echo Press ANY KEY to start the training sequence...
pause >nul


echo.
echo ===================================================
echo   STAGE 1: Baseline Models Comparison
echo ===================================================
echo.

echo [1/6] Training Siam-UNet (Diff)...
python train.py --model unet --fusion_mode diff
echo.

echo [2/6] Training Siam-UNet (Concat)...
python train.py --model unet --fusion_mode concat
echo.

echo [3/6] Training Siam-ResUNet (Concat)...
python train.py --model resunet --fusion_mode concat
echo.

echo.
echo ===================================================
echo   STAGE 2: SCA-Net Ablation Study
echo ===================================================
echo.

echo [4/6] Training SCA-Net (Base: Segformer Siam + Diff)...
python train.py --model sca_net --ablation base
echo.

echo [5/6] Training SCA-Net (+ DPB + BI3)...
python train.py --model sca_net --ablation bi3
echo.

echo [6/6] Training SCA-Net (Full: + DPB + BI3 + Multiscale + Attn)...
python train.py --model sca_net --ablation full
echo.

echo ===================================================
echo     ALL TRAINING TASKS COMPLETED SUCCESSFULLY!
echo ===================================================
pause