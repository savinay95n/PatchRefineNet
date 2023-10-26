# -----------------------------------------------------------------------------------------------------------------------
# metrics.py: This is the code for metrics used for PRN

# Usage: python src/DUTS/refine-models/metrics.py

# Note: To independently run, always run this file from PatchRefineNetwork (root) directory.
# -----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

def int_uni(gt, pred, DEVICE, smooth=1):
    pred_th = torch.zeros(pred.shape).to(DEVICE)
    pred_th[pred >= 0.5] = 1
    pred_th[pred < 0.5] = 0
    intersection = torch.sum(gt * pred_th)
    union = torch.sum(gt + pred_th) - intersection
    iou = torch.mean((intersection + smooth) / (union + smooth))
    return iou

def int_uni_numpy(gt, pred, smooth=1):
    intersection = np.sum(gt * pred)
    union = np.sum(gt + pred) - intersection
    iou = np.mean((intersection + smooth) / (union + smooth))
    return iou

def threshold_predictions(predictions, th):
    pred_th = predictions.copy()
    pred_th[pred_th <= th] = 0
    pred_th[pred_th > th] = 1
    return pred_th

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

def calculate_rst_iou(gt_list, pred_list, patch_size, key=None):
    th_distribution = []
    iou_distribution = []
    best_pred_th = []
    best_iou_list = []
    kernel = np.ones((5,5), np.uint8)
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
def calculate_ist_iou(gt_list, pred_list, key=None):
    th_distribution = []
    iou_distribution = []
    best_pred_th = []
    kernel = np.ones((5,5), np.uint8)
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

    return iou_distribution, th_distribution, best_pred_th

def mae(gt, pred):
    return torch.mean(torch.abs(pred - gt))