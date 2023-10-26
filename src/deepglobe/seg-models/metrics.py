# -----------------------------------------------------------------------------------------------------------------------
# metrics.py: This is the code containing metrics to train base networks on deepglobe
# -----------------------------------------------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.autograd import Variable as V
import numpy as np
from tqdm import tqdm

class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()
        
    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def focal_loss(self, y_true, y_pred, alpha=0.8, gamma=2, smooth=1):
        BCE = self.bce_loss(y_true, y_pred)
        BCE_EXP = torch.exp(-BCE)
        loss = alpha * (1 - BCE_EXP) ** gamma * BCE
        return loss
        
    def __call__(self, y_true, y_pred):
        a =  self.bce_loss(y_pred, y_true)
        b =  self.soft_dice_loss(y_true, y_pred)
        c = self.focal_loss(y_true, y_pred)
        return a + b

def int_uni(gt, pred, DEVICE, smooth=1):
    pred_th = torch.zeros(pred.shape).to(DEVICE)
    pred_th[pred >= 0.5] = 1
    pred_th[pred < 0.5] = 0
    intersection = torch.sum(gt * pred_th)
    union = torch.sum(gt + pred_th) - intersection
    iou = torch.mean((intersection + smooth) / (union + smooth))
    return iou

def threshold_predictions(predictions, th):
    pred_th = predictions.copy()
    pred_th[pred_th <= th] = 0
    pred_th[pred_th > th] = 1
    return pred_th

def int_uni_numpy(gt, pred, smooth=1):
    intersection = np.sum(gt * pred)
    union = np.sum(gt + pred) - intersection
    iou = np.mean((intersection + smooth) / (union + smooth))
    return iou

def patchify_image(im, patch_size):
    im_patches = []
    for i in range(im.shape[0]//patch_size):
        for j in range(im.shape[1]//patch_size):
            im_temp = im[patch_size*i:patch_size*i + patch_size, patch_size*j:patch_size*j + patch_size, :]
            im_patches.append(im_temp)
    return im_patches

def merge_patches(im_patches, orig_size = 512):
    num_cols = orig_size // im_patches[0].shape[0]
    final = np.zeros((orig_size, orig_size, 1))
    im_patches = np.array(im_patches)
    im_patches2 = im_patches.reshape(num_cols, num_cols, im_patches[0].shape[0], im_patches[0].shape[0], 1)
    for i in range(num_cols):
        for j in range(num_cols):
            final[im_patches[0].shape[0]*i:im_patches[0].shape[0]*i + im_patches[0].shape[0], im_patches[0].shape[0]*j:im_patches[0].shape[0]*j + im_patches[0].shape[0], :] = im_patches2[i,j,:,:,:]
    return final

def calculate_rst_iou(gt_list, pred_list, patch_size, key):
    th_distribution = []
    iou_distribution = []
    best_pred_th = []
    best_iou_list = []
    for i in tqdm(range(len(pred_list)), leave=True, position=0):
        raw_pred_patches = patchify_image(pred_list[i], patch_size)
        gt_patches = patchify_image(gt_list[i], patch_size)
        patch_iou = []
        patch_th = []
        patch_pred = []
        for j in range(len(gt_patches)):
            temp = []
            for t in range(1, 21):
                th = t * 0.05
                pred_th = threshold_predictions(raw_pred_patches[j], th)
                iou = int_uni_numpy(gt_patches[j], pred_th)
                temp.append(iou)
            th_index = np.argmax(temp) + 1
            best_th =  round(th_index * 0.05, 2)
            best_iou = np.max(temp)
            pred_th = threshold_predictions(raw_pred_patches[j], best_th)
            iou = int_uni_numpy(gt_patches[j], pred_th)
            patch_th.append(best_th)
            patch_iou.append(best_iou)
            patch_pred.append(pred_th)
        merged = merge_patches(patch_pred)
        best_patch_iou = int_uni_numpy(gt_list[i], merged)
        th_distribution.append(patch_th)
        iou_distribution.append(patch_iou)
        best_pred_th.append(merged)
        best_iou_list.append(best_patch_iou)

        return best_iou_list, th_distribution, best_pred_th

# IST iou calculation
def calculate_ist_iou(gt_list, pred_list, key):
    th_distribution = []
    iou_distribution = []
    best_pred_th = []
    for i in tqdm(range(len(pred_list)), leave=True, position=0):
        temp_iou = []
        for t in range(1, 21):
            th = t * 0.05
            pred_th = threshold_predictions(pred_list[i], th)
            iou = int_uni_numpy(gt_list[i], pred_th)
            temp_iou.append(iou)
        th_index = np.argmax(temp_iou) + 1
        best_th =  round(th_index * 0.05, 2)
        best_iou = np.max(temp_iou)
        pred_th = threshold_predictions(pred_list[i], best_th)
        th_distribution.append(best_th)
        iou_distribution.append(best_iou)
        best_pred_th.append(pred_th)
    print('[INFO] IST mIoU on {} set = {:.4f}'.format(key, np.mean(iou_distribution)))
    return iou_distribution, th_distribution, best_pred_th