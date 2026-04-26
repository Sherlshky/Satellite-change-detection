# utils/sliding_window.py
import torch

def sliding_window_predict(model, img_A, img_B, crop_size=256, stride=256):

    b, c, h, w = img_A.shape
    device = img_A.device
    

    preds = torch.zeros((b, 1, h, w), device=device)
    count = torch.zeros((b, 1, h, w), device=device)
    
    for i in range(0, h, stride):
        for j in range(0, w, stride):

            i_end = min(i + crop_size, h)
            j_end = min(j + crop_size, w)
            i_start = max(0, i_end - crop_size)
            j_start = max(0, j_end - crop_size)
            

            patch_A = img_A[:, :, i_start:i_end, j_start:j_end]
            patch_B = img_B[:, :, i_start:i_end, j_start:j_end]
            

            with torch.no_grad():
                output = model(patch_A, patch_B)
                pred_prob = torch.sigmoid(output) 

            preds[:, :, i_start:i_end, j_start:j_end] += pred_prob
            count[:, :, i_start:i_end, j_start:j_end] += 1
            

    final_prob = preds / count
    final_pred = (final_prob > 0.5).float()
    
    return final_pred