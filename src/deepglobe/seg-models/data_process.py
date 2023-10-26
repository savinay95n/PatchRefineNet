# -----------------------------------------------------------------------------------------------------------------------
# dataprocess.py: This is the code to preproces data for DeepGlobe base network training.

# Usage: python src/deepglobe/seg-models/data_process.py

# Note: To independently run, always run this file from PatchRefineNetwork (root) directory.
# -----------------------------------------------------------------------------------------------------------------------

import torch
import torch.utils.data as data
from torch.autograd import Variable as V
import pandas as pd
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import colors
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid


# Working Directory (root)
WORKING_DIR = os.getcwd()
# Base path of dataset
DATA_DIR = os.path.join(WORKING_DIR, 'data', 'deepglobe', 'seg-models')

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, mask,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-0.0, 0.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, mask

def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask=np.rot90(mask)

    return image, mask

def default_loader(id, sat_root, mask_root):
    img = cv2.imread(os.path.join(sat_root,'{}_sat.jpg').format(id))
    mask = cv2.imread(os.path.join(mask_root,'{}_mask.png').format(id), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_AREA)
    mask = cv2.resize(mask, (512, 512), interpolation = cv2.INTER_AREA)
    img = randomHueSaturationValue(img,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))
    
    img, mask = randomShiftScaleRotate(img, mask,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-0, 0))
    img, mask = randomHorizontalFlip(img, mask)
    img, mask = randomVerticleFlip(img, mask)
    img, mask = randomRotate90(img, mask)
    
    mask = np.expand_dims(mask, axis=2)
    orig_img = img.copy()
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
    mask[mask>=0.5] = 1
    mask[mask<0.5] = 0
    return img, mask, orig_img

def dataloader_for_predictions(id, sat_root, mask_root):
    img = cv2.imread(os.path.join(sat_root,'{}_sat.jpg').format(id))
    mask = cv2.imread(os.path.join(mask_root,'{}_mask.png').format(id), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512, 512), interpolation = cv2.INTER_AREA)
    orig_img = img.copy()
    orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    mask = cv2.resize(mask, (512, 512), interpolation = cv2.INTER_AREA)
    mask = np.expand_dims(mask, axis=2)
    img = np.array(img, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6
    mask = np.array(mask, np.float32).transpose(2,0,1)/255.0
    mask[mask>=0.5] = 1
    mask[mask<=0.5] = 0
    return img, mask, orig_img

class ImageFolder(data.Dataset):

    def __init__(self, trainlist, sat_root, mask_root):
        self.ids = trainlist
        self.loader = default_loader
        self.sat_root = sat_root
        self.mask_root = mask_root

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask, orig_img = self.loader(id, self.sat_root, self.mask_root)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask, orig_img

    def __len__(self):
        return len(self.ids)

class ImageFolderPredictions(data.Dataset):

    def __init__(self, trainlist, sat_root, mask_root):
        self.ids = trainlist
        self.loader = dataloader_for_predictions
        self.sat_root = sat_root
        self.mask_root = mask_root

    def __getitem__(self, index):
        id = self.ids[index]
        img, mask, orig_img = self.loader(id, self.sat_root, self.mask_root)
        img = torch.Tensor(img)
        mask = torch.Tensor(mask)
        return img, mask, orig_img

    def __len__(self):
        return len(self.ids)

def sample_plots(img, mask, pred, mask_edges, pred_edges, iou, iou_edge, args, index, key, dir):
    cmap = colors.ListedColormap(['blue'])
    fig, ax = plt.subplots(1, 5, figsize=(26,5))
    ax[0].imshow(img)
    ax[0].axis('off')
    ax[0].set_title('Input Image')
    ax[1].imshow(img)
    masked = np.ma.masked_where(mask[:,:,0] == 0, mask[:,:,0])
    ax[1].imshow(masked, cmap = cmap,interpolation='none', alpha=1)
    ax[1].axis('off')
    ax[1].set_title('Ground Truth Mask')
    im1 = ax[2].imshow(pred, cmap=cm.jet)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size='2%', pad=0.05)
    ax[2].axis('off')
    ax[2].set_title('Confidence Map')
    fig.colorbar(im1, cax=cax, orientation='vertical')
    ax[3].imshow(img)
    masked = np.ma.masked_where(mask_edges[:,:,0] == 0, mask_edges[:,:,0])
    ax[3].imshow(masked, cmap = cmap,interpolation='none', alpha=1)
    ax[3].axis('off')
    ax[3].set_title('Ground Truth Boundary Mask')
    im2 = ax[4].imshow(pred_edges, cmap=cm.jet)
    divider = make_axes_locatable(ax[4])
    cax = divider.append_axes('right', size='2%', pad=0.05)
    ax[4].axis('off')
    ax[4].set_title('Boundary Confidence Map')
    fig.colorbar(im2, cax=cax, orientation='vertical')
    plt.suptitle('{} segmentation model prediction: IoU = {:.2f}, mBA = {:.2f}'.format(args.model, iou, iou_edge))
    plt.savefig(os.path.join(dir, 'key'+'_'+str(index)+'.png'))
    plt.close('all')

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
    return iou_distribution, th_distribution, best_pred_th