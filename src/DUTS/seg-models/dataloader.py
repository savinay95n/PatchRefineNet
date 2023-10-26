# ------------------------------------------------------------------------------------------------
# dataloader.py : Custom dataloader with augmentations for DUTS dataset
# ------------------------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import sys
import cv2
import numpy as np
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm

def pad_resize_image(inp_img, out_img=None, target_size=None):
    """
    Function to pad and resize images to a given size.
    out_img is None only during inference. During training and testing
    out_img is NOT None.
    :param inp_img: A H x W x C input image.
    :param out_img: A H x W input image of mask.
    :param target_size: The size of the final images.
    :return: Re-sized inp_img and out_img
    """
    h, w, c = inp_img.shape
    size = max(h, w)

    padding_h = (size - h) // 2
    padding_w = (size - w) // 2

    if out_img is None:
        # For inference
        temp_x = cv2.copyMakeBorder(inp_img, top=padding_h, bottom=padding_h, left=padding_w, right=padding_w,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        if target_size is not None:
            temp_x = cv2.resize(temp_x, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return temp_x
    else:
        # For training and testing
        temp_x = cv2.copyMakeBorder(inp_img, top=padding_h, bottom=padding_h, left=padding_w, right=padding_w,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        temp_y = cv2.copyMakeBorder(out_img, top=padding_h, bottom=padding_h, left=padding_w, right=padding_w,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # print(inp_img.shape, temp_x.shape, out_img.shape, temp_y.shape)

        if target_size is not None:
            temp_x = cv2.resize(temp_x, (target_size, target_size), interpolation=cv2.INTER_AREA)
            temp_y = cv2.resize(temp_y, (target_size, target_size), interpolation=cv2.INTER_AREA)
        return temp_x, temp_y


def random_crop_flip(inp_img, out_img):
    """
    Function to randomly crop and flip images.
    :param inp_img: A H x W x C input image.
    :param out_img: A H x W input image.
    :return: The randomly cropped and flipped image.
    """
    h, w = out_img.shape

    rand_h = np.random.randint(h/8)
    rand_w = np.random.randint(w/8)
    offset_h = 0 if rand_h == 0 else np.random.randint(rand_h)
    offset_w = 0 if rand_w == 0 else np.random.randint(rand_w)
    p0, p1, p2, p3 = offset_h, h+offset_h-rand_h, offset_w, w+offset_w-rand_w

    rand_flip = np.random.randint(10)
    if rand_flip >= 5:
        inp_img = inp_img[::, ::-1, ::]
        out_img = out_img[::, ::-1]

    return inp_img[p0:p1, p2:p3], out_img[p0:p1, p2:p3]


def random_rotate(inp_img, out_img, max_angle=25):
    """
    Function to randomly rotate images within +max_angle to -max_angle degrees.
    This algorithm does NOT crops the edges upon rotation.
    :param inp_img: A H x W x C input image.
    :param out_img: A H x W input image.
    :param max_angle: Maximum angle an image can be rotated in either direction.
    :return: The randomly rotated image.
    """
    angle = np.random.randint(-max_angle, max_angle)
    h, w = out_img.shape
    center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # Compute new dimensions of the image and adjust the rotation matrix
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(inp_img, M, (new_w, new_h)), cv2.warpAffine(out_img, M, (new_w, new_h))


def random_rotate_lossy(inp_img, out_img, max_angle=25):
    """
    Function to randomly rotate images within +max_angle to -max_angle degrees.
    This algorithm crops the edges upon rotation.
    :param inp_img: A H x W x C input image.
    :param out_img: A H x W input image.
    :param max_angle: Maximum angle an image can be rotated in either direction.
    :return: The randomly rotated image.
    """
    angle = np.random.randint(-max_angle, max_angle)
    h, w = out_img.shape
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(inp_img, M, (w, h)), cv2.warpAffine(out_img, M, (w, h))


def random_brightness(inp_img):
    """
    Function to randomly perturb the brightness of the input images.
    :param inp_img: A H x W x C input image.
    :return: The image with randomly perturbed brightness.
    """
    contrast = np.random.rand(1) + 0.5
    light = np.random.randint(-20, 20)
    inp_img = contrast * inp_img + light

    return np.clip(inp_img, 0, 255)

class SODLoader(Dataset):
    """
    DataLoader for DUTS dataset (for training and testing).
    """
    def __init__(self, input_root, mask_root, csv_root, input_csv, augment_data=False, target_size=512):
        self.augment_data = augment_data
        self.target_size = target_size
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            self.normalize,
        ])  # Not used

        self.img_files = list(pd.read_csv(os.path.join(csv_root, input_csv))['img'])
        self.mask_files = list(pd.read_csv(os.path.join(csv_root, input_csv))['target'])
        self.img_files = [os.path.join(input_root, x) for x in self.img_files]
        self.mask_files = [os.path.join(mask_root, x) for x in self.mask_files]

    def __getitem__(self, idx):
        inp_img = cv2.imread(self.img_files[idx])
        inp_img = cv2.cvtColor(inp_img, cv2.COLOR_BGR2RGB)
        inp_img = inp_img.astype('float32')

        mask_img = cv2.imread(self.mask_files[idx], 0)
        mask_img = mask_img.astype('float32')
        mask_img /= np.max(mask_img)

        if self.augment_data:
            inp_img, mask_img = random_crop_flip(inp_img, mask_img)
            inp_img, mask_img = random_rotate(inp_img, mask_img)
            inp_img = random_brightness(inp_img)

        # Pad images to target size
        inp_img, mask_img = pad_resize_image(inp_img, mask_img, self.target_size)
        inp_img /= 255.0
        inp_img = np.transpose(inp_img, axes=(2, 0, 1))
        inp_img = torch.from_numpy(inp_img).float()
        inp_img = self.normalize(inp_img)

        mask_img = np.expand_dims(mask_img, axis=0)

        return inp_img, torch.from_numpy(mask_img).float()

    def __len__(self):
        return len(self.img_files)

if __name__ == '__main__':
    # Test Dataloader
    img_size = 512
    bs = 8
    WORKING_DIR = os.getcwd()
    csv_root = os.path.join(WORKING_DIR, 'data', 'DUTS', 'seg-models')
    train_input_root = os.path.join(csv_root, 'DUTS-TR', 'DUTS-TR', 'DUTS-TR-Image')
    train_mask_root = os.path.join(csv_root, 'DUTS-TR', 'DUTS-TR', 'DUTS-TR-Mask')
    val_input_root = os.path.join(csv_root, 'DUTS-TE', 'DUTS-TE', 'DUTS-TE-Image')
    val_mask_root = os.path.join(csv_root, 'DUTS-TE', 'DUTS-TE', 'DUTS-TE-Mask')
    test_input_root = os.path.join(csv_root, 'DUTS-TE', 'DUTS-TE', 'DUTS-TE-Image')
    test_mask_root = os.path.join(csv_root, 'DUTS-TE', 'DUTS-TE', 'DUTS-TE-Mask')

    train_data = SODLoader(train_input_root, train_mask_root, csv_root, 'train.csv', augment_data=True, target_size=img_size)
    val_data = SODLoader(val_input_root, val_mask_root, csv_root, 'val.csv', augment_data=False, target_size=img_size)
    test_data = SODLoader(test_input_root, test_mask_root, csv_root, 'test.csv', augment_data=False, target_size=img_size)

    train_dataloader = DataLoader(train_data, batch_size=bs, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=bs, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=bs, shuffle=False)

    print("Train Dataloader :")
    for batch_idx, (inp_imgs, gt_masks) in enumerate(train_dataloader):
        print('Loop :', batch_idx, inp_imgs.size(), gt_masks.size())
        if batch_idx == 3:
            break

    sample_img = inp_imgs.permute(0,2,3,1).cpu().detach().numpy()[0]
    sample_mask = gt_masks.permute(0,2,3,1).cpu().detach().numpy()[0]

    plt.imshow(sample_img)
    cmap = colors.ListedColormap(['green'])
    masked = np.ma.masked_where(sample_mask[:,:,0] == 0, sample_mask[:,:,0])
    plt.imshow(masked, cmap = cmap,interpolation='none', alpha=0.6)
    plt.savefig('sample_img.png')
    

    print("Val Dataloader :")
    for batch_idx, (inp_imgs, gt_masks) in enumerate(val_dataloader):
        print('Loop :', batch_idx, inp_imgs.size(), gt_masks.size())
        if batch_idx == 3:
            break

    print("\nTest Dataloader :")
    for batch_idx, (inp_imgs, gt_masks) in enumerate(test_dataloader):
        print('Loop :', batch_idx, inp_imgs.size(), gt_masks.size())
        if batch_idx == 3:
            break

    