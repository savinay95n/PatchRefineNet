# -----------------------------------------------------------------------------------------------------------------------
# generate_aux_training.py: This is the code to run inference for the base saliency detection network 
# Pfanet on DUTS test set.

# Usage: python src/DUTS/seg-models/calculate_base_iou.py --model pfanet

# Note: To independently run, always run this file from PatchRefineNetwork (root) directory.
# -----------------------------------------------------------------------------------------------------------------------

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from networks.pfanet import SODModel
from loss import EdgeSaliencyLoss
from dataloader import SODLoader
import logging
from tqdm import tqdm
import time
from torchsummary import summary
from metrics import int_uni_numpy
import cv2
from torchmetrics import JaccardIndex

WORKING_DIR = os.getcwd()

# Command line arguments for which model to train
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Model Name: Choose one of [pfanet]', type=str, required=True)
parser.add_argument('--checkpoint', help='Checkpoint Name', type=str)
args = parser.parse_args()


# Log directory
LOG_DIR = os.path.join(WORKING_DIR, 'src', 'DUTS', 'seg-models', 'logs', args.model)
# Checkpoint directory
CHECKPOINT_DIR = os.path.join(WORKING_DIR, 'src', 'DUTS', 'seg-models', 'checkpoints', args.model)
# log file name
LOG_FILE_NAME = args.model + '_log' + '_prediction' 
# Hardware Accelerator
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

jaccard = JaccardIndex(num_classes=2).to(DEVICE)

# Data root directories
csv_root = os.path.join(WORKING_DIR, 'data', 'DUTS', 'seg-models')
input_root = os.path.join(csv_root, 'DUTS-TE', 'DUTS-TE', 'DUTS-TE-Image')
mask_root = os.path.join(csv_root, 'DUTS-TE', 'DUTS-TE', 'DUTS-TE-Mask')

# Hyperparameters
BATCH_SIZE = 1
img_size = 256

print('Generating Base Model Confidence Maps for Auxiliary training ...')

model_name = 'best-model_epoch-204_mae-0.0505_loss-0.1370.pth'

model = SODModel()
chkpt = torch.load(os.path.join(CHECKPOINT_DIR, model_name), map_location=DEVICE)
model.load_state_dict(chkpt['model'])
model.to(DEVICE)
model.eval()
summary(model)

# Loading data
val_data = SODLoader(input_root, mask_root, csv_root, 'val.csv', augment_data=False, target_size=img_size)

val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

val_pred = []
val_gt = []
val_mae = []
val_iou = []
# validation data
with torch.no_grad():
    model.eval()
    val_data_loader_iter = iter(val_dataloader)
    for inp_imgs, gt_masks in tqdm(val_data_loader_iter, leave=True, position=0):
        inp_imgs = inp_imgs.to(DEVICE)
        gt_masks = gt_masks.to(DEVICE)
        pred_masks, ca_act_reg = model(inp_imgs)
        pred_masks = pred_masks.permute(0,2,3,1).cpu().detach().numpy()
        pred_masks = cv2.resize(pred_masks[0], (512, 512), interpolation=cv2.INTER_AREA)
        gt_masks = gt_masks.permute(0,2,3,1).cpu().detach().numpy()
        gt_masks = cv2.resize(gt_masks[0], (512, 512), interpolation=cv2.INTER_AREA)
        gt_masks = np.round(gt_masks)
        pred_masks = np.expand_dims(pred_masks, -1)
        gt_masks = np.expand_dims(gt_masks, -1)
        val_pred.append(pred_masks)
        val_gt.append(gt_masks)

val_pred = np.array(val_pred)
val_gt = np.array(val_gt)


print('Saving Aux Training Data ...')
np.save(os.path.join(WORKING_DIR, 'data', 'DUTS', 'refine-models', args.model, 'val_pred.npy'), val_pred)
np.save(os.path.join(WORKING_DIR, 'data', 'DUTS', 'refine-models', args.model, 'val_orig_gt.npy'), val_gt)
print('Saved ...')


    
