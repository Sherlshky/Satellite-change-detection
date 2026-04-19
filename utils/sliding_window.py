# utils/sliding_window.py
import torch

def sliding_window_predict(model, img_A, img_B, crop_size=256, stride=256):
    """
    对大分辨率图像进行滑窗切片推理并拼接。
    LEVIR-CD 中原图为 1024，crop_size=256, stride=256 时为完美无重叠切片 (4x4=16块)
    """
    b, c, h, w = img_A.shape
    device = img_A.device
    
    # 初始化完整的预测概率图和计数图 (用于重叠部分求均值，即便无重叠写全逻辑也更严谨)
    preds = torch.zeros((b, 1, h, w), device=device)
    count = torch.zeros((b, 1, h, w), device=device)
    
    for i in range(0, h, stride):
        for j in range(0, w, stride):
            # 处理边界防溢出
            i_end = min(i + crop_size, h)
            j_end = min(j + crop_size, w)
            i_start = max(0, i_end - crop_size)
            j_start = max(0, j_end - crop_size)
            
            # 截取当前窗口的特征块
            patch_A = img_A[:, :, i_start:i_end, j_start:j_end]
            patch_B = img_B[:, :, i_start:i_end, j_start:j_end]
            
            # 推理
            with torch.no_grad():
                output = model(patch_A, patch_B)
                pred_prob = torch.sigmoid(output) # 转换为 0~1 的概率
                
            # 累加预测值和次数
            preds[:, :, i_start:i_end, j_start:j_end] += pred_prob
            count[:, :, i_start:i_end, j_start:j_end] += 1
            
    # 计算平均概率并二值化
    final_prob = preds / count
    final_pred = (final_prob > 0.5).float()
    
    return final_pred