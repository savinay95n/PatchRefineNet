# -----------------------------------------------------------------------------------------------------------------------
# generate_rst.py: This is the code to generate region-specific ground truths for DUTS dataset.
# Usage: python src/DUTS/seg-models/generate_rst.py --model pfanet

# Note: To independently run, always run this file from PatchRefineNetwork (root) directory.
# -----------------------------------------------------------------------------------------------------------------------
import os
from tqdm import tqdm
import logging
import argparse
import numpy as np
from metrics import threshold_predictions, calculate_ist_iou, calculate_rst_iou

WORKING_DIR = os.getcwd()
DATA_DIR = os.path.join(WORKING_DIR, 'data', 'DUTS', 'refine-models', 'pfanet')
LOG_DIR = os.path.join(WORKING_DIR, 'src', 'DUTS', 'seg-models', 'logs', 'pfanet')

# Command line arguments for which model to train
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Model Name: Choose one of [pfanet]', type=str, required=True)
args = parser.parse_args()

train_input = np.load(os.path.join(DATA_DIR, 'val_pred.npy'))
train_orig_gt = np.load(os.path.join(DATA_DIR, 'val_orig_gt.npy'))

def generate_rst_gt(train_input, train_gt):
    print('Generating Region Specific Thresholding Ground Truth for Auxiliary training ...')
    # patch size from 512 to 16 can be given.
    patch_sizes = [64]
    rst_dict = {}
    for p in tqdm(patch_sizes, position=0, leave=True):
        rst_dict[p] = {}
        rst_dict[p]['train'] = {}
        rst_dict[p]['test'] = {}
        if p == 512:
            train_iou_dist_ist, train_th_dist_ist, train_ist_pred = calculate_ist_iou(train_gt, train_input, 'Train')
            np.save(os.path.join(DATA_DIR, 'train_aux_gt_p_{}'.format(p)), train_ist_pred)
        else:
            train_iou_dist_rst, train_th_dist_rst, train_rst_pred = calculate_rst_iou(train_gt, train_input, p, 'Train')
            np.save(os.path.join(DATA_DIR, 'train_aux_gt_p_{}'.format(p)), train_rst_pred)
            

if __name__ == "__main__":
    generate_rst_gt(train_input, train_orig_gt)