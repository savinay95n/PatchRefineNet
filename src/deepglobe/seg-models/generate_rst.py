# -----------------------------------------------------------------------------------------------------------------------
# generate_rst.py: This is the code to run generate region-specific ground truths for deepglobe validation dataset.

# Usage: python src/deepglobe/seg-models/generate_rst.py --model dlinknet

# Note: To independently run the model, always run this file from PatchRefineNetwork (root) directory.
# -----------------------------------------------------------------------------------------------------------------------

import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from data_process import calculate_rst_iou, calculate_ist_iou, threshold_predictions
import logging
import argparse
import cv2

# Command line arguments for which model to train
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Model Name: Choose one of [unet, dkinknet]', type=str, required=True)
args = parser.parse_args()

#CONSTANTS
# Working Directory (root)
WORKING_DIR = os.getcwd()
# Base path of dataset
DATA_DIR = os.path.join(WORKING_DIR, 'data', 'deepglobe', 'seg-models')
DATA_DIR2 = os.path.join(WORKING_DIR, 'data', 'deepglobe', 'refine-models', args.model)

train_input = np.load(os.path.join(DATA_DIR2, 'val_pred.npy'))
train_orig_gt = np.load(os.path.join(DATA_DIR2, 'val_orig_gt.npy'))

def generate_rst_gt(train_input, train_gt):
    patch_sizes = [64]
    rst_dict = {}
    for p in patch_sizes:
        rst_dict[p] = {}
        rst_dict[p]['train'] = {}
        rst_dict[p]['test'] = {}
        if p == 512:
            train_iou_dist_ist, train_th_dist_ist, train_ist_pred = calculate_ist_iou(train_gt, train_input, 'Train')
            np.save(os.path.join(DATA_DIR2, 'train_aux_gt_p_{}'.format(p)), train_ist_pred)

        else:
            train_iou_dist_rst, train_th_dist_rst, train_rst_pred = calculate_rst_iou(train_gt, train_input, p, 'Train')
            np.save(os.path.join(DATA_DIR2, 'train_aux_gt_p_{}'.format(p)), train_rst_pred)

if __name__ == "__main__":
    generate_rst_gt(train_input, train_orig_gt)
    