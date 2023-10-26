# -----------------------------------------------------------------------------------------------------------------------
# dataprocess.py: This is the code to preprocess data for training PRN on DUTS data.

# Usage: python src/DUTS/refine-models/dataprocess.py

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

def randomHorizontalFlip(image, mask1, mask2, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask1 = cv2.flip(mask1, 1)
        mask2 = cv2.flip(mask2, 1)

    return image, mask1, mask2

def randomVerticleFlip(image, mask1, mask2, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask1 = cv2.flip(mask1, 0)
        mask2 = cv2.flip(mask2, 0)

    return image, mask1, mask2

def randomRotate90(image, mask1, mask2, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        mask1=np.rot90(mask1)
        mask2=np.rot90(mask2)

    return image, mask1, mask2


class ImageFolder(data.Dataset):
    def transform(self, img, mask1, mask2):
        img, mask1, mask2 = randomHorizontalFlip(img, mask1, mask2)
        img, mask1, mask2 = randomVerticleFlip(img, mask1, mask2)
        img, mask1, mask2 = randomRotate90(img, mask1, mask2)
        return img, mask1, mask2

    def __init__(self, input_arr, mask1_arr, mask2_arr):
        self.input = input_arr
        self.mask1 = mask1_arr
        self.mask2 = mask2_arr

    def __getitem__(self, index):
        img, mask1, mask2 = self.input[index][:,:,0], self.mask1[index][:,:,0], self.mask2[index][:,:,0]
        img, mask1, mask2 = self.transform(img, mask1, mask2)
        img, mask1, mask2 = np.expand_dims(img, axis=-1), np.expand_dims(mask1, axis=-1), np.expand_dims(mask2, axis=-1)
        img = torch.from_numpy(img.copy())
        mask1 = torch.from_numpy(mask1.copy())
        mask2 = torch.from_numpy(mask2.copy())
        return img, mask1, mask2

    def __len__(self):
        return len(self.input)

class ImageFolderPredictions(data.Dataset):

    def __init__(self, input_arr, mask1_arr, mask2_arr):
        self.input = input_arr
        self.mask1 = mask1_arr
        self.mask2 = mask2_arr

    def __getitem__(self, index):
        img, mask1, mask2 = self.input[index], self.mask1[index], self.mask2[index]
        img = torch.from_numpy(img.copy())
        mask1 = torch.from_numpy(mask1.copy())
        mask2 = torch.from_numpy(mask2.copy())
        return img, mask1, mask2

    def __len__(self):
        return len(self.input)

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


def plot(sample_input, sample_mask, sample_orig_mask):
    fig, ax = plt.subplots(4, 3, figsize=(12,8))
    ax[0][0].imshow(sample_input[0])
    ax[0][1].imshow(sample_mask[0])
    ax[0][2].imshow(sample_orig_mask[0])

    ax[1][0].imshow(sample_input[1])
    ax[1][1].imshow(sample_mask[1])
    ax[1][2].imshow(sample_orig_mask[1])

    ax[2][0].imshow(sample_input[2])
    ax[2][1].imshow(sample_mask[2])
    ax[2][2].imshow(sample_orig_mask[2])

    ax[3][0].imshow(sample_input[3])
    ax[3][1].imshow(sample_mask[3])
    ax[3][2].imshow(sample_orig_mask[3])

    plt.savefig('test.png')