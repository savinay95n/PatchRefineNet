# -----------------------------------------------------------------------------------------------------------------------
# genearte_rst.py: This is the code to generate region-specific ground truth for base segmentation network 
# on kvasir-seg dataset.

# Usage: python src/kvasir/seg-models/genearte_rst.py --model resunetplusplus

# Note: To independently run, always run this file from PatchRefineNetwork (root) directory.
# -----------------------------------------------------------------------------------------------------------------------
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
from data_process import calculate_rst_iou, threshold_predictions
import logging
import argparse
import cv2

# Command line arguments for which model to train
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Model Name: Choose one of [resunetplusplus]', type=str, required=True)
args = parser.parse_args()

#CONSTANTS
# Working Directory (root)
WORKING_DIR = os.getcwd()
# Base path of dataset
DATA_DIR = os.path.join(WORKING_DIR, 'data', 'kvasir', 'seg-models')
DATA_DIR2 = os.path.join(WORKING_DIR, 'data', 'kvasir', 'refine-models', args.model)

train_input = np.load(os.path.join(DATA_DIR2, 'val_pred.npy'))
train_orig_gt = np.load(os.path.join(DATA_DIR2, 'val_orig_gt.npy'))

def generate_rst_gt(train_input, train_gt):
    # patch sizes between 512 and 16
    patch_sizes = [64]
    rst_dict = {}
    for p in patch_sizes:
        rst_dict[p] = {}
        rst_dict[p]['train'] = {}
        rst_dict[p]['test'] = {}

        train_iou_dist_rst, train_th_dist_rst, train_rst_pred = calculate_rst_iou(train_gt, train_input, p, 'Train')
        np.save(os.path.join(DATA_DIR2, 'train_aux_gt_p_{}'.format(p)), train_rst_pred)
        
if __name__ == "__main__":
    generate_rst_gt(train_input, train_orig_gt)
    