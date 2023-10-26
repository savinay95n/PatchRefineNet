# -----------------------------------------------------------------------------------------------------------------------
# utils.py: This is the code for Utility functions for PRN
# Usage: python src/deepglobe/refine-models/utils.py

# Note: To independently run, always run this file from PatchRefineNetwork (root) directory.
# -----------------------------------------------------------------------------------------------------------------------
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
import cv2
import os


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

def calculate_iou(gt_list, pred_list, patch_size):
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
    
    return best_iou_list, best_pred_th

def iou_calculate(gt_list, pred_list, th):
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
    return best_pred_th

def plot(DATA_DIR_SEG, test_input, pred_base_th, pred_list, test_dataframe, pred_th, test_orig_gt):
    frames = random.sample(range(len(test_input)), 100)

    for i, frame in enumerate(frames):
        im = cv2.imread(os.path.join(DATA_DIR_SEG, 'DUTS-TE', 'DUTS-TE', 'DUTS-TE-Image', test_dataframe[frame]))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (512, 512), interpolation = cv2.INTER_AREA)

        fig, ax = plt.subplots(1, 5, figsize=(25,20))
        im1 = ax[0].imshow(test_input[frame], cmap = cm.jet)
        ax[0].axis('off')
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='2%', pad=0.05)
        fig.colorbar(im1, cax=cax, orientation='vertical')

        ax[1].imshow(im)
        cmap1 = colors.ListedColormap(['tomato'])
        cmap2 = colors.ListedColormap(['red'])
        masked1 = np.ma.masked_where(pred_base_th[frame][:,:,0] == 0, pred_base_th[frame][:,:,0])
        ax[1].imshow(masked1, cmap = cmap1,interpolation='none', alpha=1)
        ax[1].axis('off')

        im2 = ax[2].imshow(pred_list[frame], cmap = cm.jet)
        ax[2].axis('off')
        divider = make_axes_locatable(ax[2])
        fig.colorbar(im1, cax=cax, orientation='vertical')

        ax[3].imshow(im)
        cmap1 = colors.ListedColormap(['cyan'])
        cmap2 = colors.ListedColormap(['blue'])
        masked1 = np.ma.masked_where(pred_th[frame][:,:,0] == 0, pred_th[frame][:,:,0])
        ax[3].imshow(masked1, cmap = cmap1,interpolation='none', alpha=1)
        ax[3].axis('off')

        ax[4].imshow(im)
        cmap1 = colors.ListedColormap(['lime'])
        cmap2 = colors.ListedColormap(['green'])
        masked1 = np.ma.masked_where(test_orig_gt[frame][:,:,0] == 0, test_orig_gt[frame][:,:,0])
        ax[4].imshow(masked1, cmap = cmap1,interpolation='none', alpha=1)
        ax[4].axis('off')
        plt.tight_layout(pad=0, w_pad=0.5)
        plt.savefig('test_{}.png'.format(frame))
