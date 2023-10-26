# -----------------------------------------------------------------------------------------------------------------------
# train.py: This is the code for running inference on PRN
# Usage: python src/kvasir/refine-models/predict.py --base_model resunetplusplus --aux_model p64 --checkpoint True

# Note: To independently run, always run this file from PatchRefineNetwork (root) directory.
# -----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import cv2
import os
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from network import ThreshNetP0, ThreshNetP4, ThreshNetP64
from torchsummary import summary
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import logging
import argparse
import time
from data_process import ImageFolder, ImageFolderPredictions
from metrics import int_uni, mae
from utils import int_uni_numpy, threshold_predictions
import random

# Working Directory (root)
WORKING_DIR = os.getcwd()

# Command line arguments for which model to train
parser = argparse.ArgumentParser()
parser.add_argument('--base_model', help='Model Name: Choose one of [unet, dkinknet]', type=str, required=True)
parser.add_argument('--checkpoint', help='Checkpoint Name to start training from', type=str, required=True)
parser.add_argument('--aux_model', 
            help='Choose Aux Model with number of local patches (granularity of refinement) [p0, p4, p16, p64, p256, p1024]',
            type=str, required=True)
args = parser.parse_args()

# Base path of dataset
DATA_DIR = os.path.join(WORKING_DIR, 'data', 'kvasir', 'refine-models', args.base_model)

# Log directory
LOG_DIR = os.path.join(WORKING_DIR, 'src', 'kvasir', 'refine-models', 'logs', args.aux_model)
# Checkpoint directory
CHECKPOINT_DIR = os.path.join(WORKING_DIR, 'src', 'kvasir', 'refine-models', 'checkpoints', args.aux_model)

# log file name
LOG_FILE_NAME = args.aux_model + '_predict_log'
# Hardware Accelerator
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
BATCH_SIZE = 1
model_name = args.aux_model + '_weights.th'

# Open Training Log File 
filename = os.path.join(LOG_DIR, LOG_FILE_NAME+'.log')
logging.basicConfig(filename=filename, level=logging.DEBUG)

logging.info('==========================================================================================')
logging.info('Predicting %s aux model on kvasir dataset trained on base model %s', args.aux_model, args.base_model)
logging.info('==========================================================================================')

# Invoke Aux model
if args.aux_model == 'p0':
    model = ThreshNetP0().to(DEVICE)
elif args.aux_model == 'p4':
    model = ThreshNetP4().to(DEVICE)
elif args.aux_model == 'p64':
    model = ThreshNetP64().to(DEVICE)

if args.checkpoint:
    logging.info('Loading Saved Checkpoint ...')
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, model_name)))
    print("=================================================================")
summary(model)

# Dataset
print('[INFO] Loading data ...')


test_input = np.load(os.path.join(DATA_DIR, 'test_input.npy'))
test_orig_gt = np.load(os.path.join(DATA_DIR, 'test_orig_gt.npy'))

if args.aux_model == 'p0':
    test_gt = np.load(os.path.join(DATA_DIR, 'test_aux_gt_p_512.npy'))
elif args.aux_model == 'p4':
    test_gt = np.load(os.path.join(DATA_DIR, 'test_aux_gt_p_256.npy'))
elif args.aux_model == 'p64':
    test_gt = np.load(os.path.join(DATA_DIR, 'test_aux_gt_p_64.npy'))
print('[INFO] Loaded data ...')

# Create train dataset
test_dataset = ImageFolderPredictions(test_input, test_gt, test_orig_gt)
test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE)

if args.aux_model == 'p0':
    pred_list = []
    with torch.no_grad():
        # loop over the training set
        for (x, y1, y2) in tqdm(test_dataloader, leave=True, position=0):
            x, y1, y2 = x.type(torch.FloatTensor), y1.type(torch.FloatTensor), y2.type(torch.FloatTensor)
            # Pytorch takes (N,C,H,W) format
            x = x.permute(0,3,1,2)
            y1 = y1.permute(0,3,1,2)
            y2 = y2.permute(0,3,1,2)
            (x, y1, y2) = (x.to(DEVICE), y1.to(DEVICE), y2.to(DEVICE))
            pred = model(x)

            pred = pred.permute(0,2,3,1).cpu().detach().numpy()[0]
            pred_list.append(pred)

    pred_list = np.array(pred_list)

    print('Calculating mIoU ...')
    pred_th = threshold_predictions(pred_list, 0.5)
    iou_global = int_uni_numpy(test_orig_gt, pred_th)

    logging.info('=================================================================')
    logging.info('Total mIoU after Refinement = %f', iou_global)

else:
    pred_global_list = []
    pred_local_list = []
    with torch.no_grad():
        # loop over the training set
        for (x, y1, y2) in tqdm(test_dataloader, leave=True, position=0):
            x, y1, y2 = x.type(torch.FloatTensor), y1.type(torch.FloatTensor), y2.type(torch.FloatTensor)
            # Pytorch takes (N,C,H,W) format
            x = x.permute(0,3,1,2)
            y1 = y1.permute(0,3,1,2)
            y2 = y2.permute(0,3,1,2)
            (x, y1, y2) = (x.to(DEVICE), y1.to(DEVICE), y2.to(DEVICE))
            pred_global, pred_local = model(x)
            pred_global = pred_global.permute(0,2,3,1).cpu().detach().numpy()[0]
            pred_local = pred_local.permute(0,2,3,1).cpu().detach().numpy()[0]
            pred_global_list.append(pred_global)
            pred_local_list.append(pred_local)

    pred_global_list = np.array(pred_global_list)
    pred_local_list = np.array(pred_local_list)
    
    print('Calculating mIoU ...')
    pred_global_th = threshold_predictions(pred_global_list, 0.5)
    pred_local_th = threshold_predictions(pred_local_list, 0.5)

    pred_global_th = pred_global_th.astype('uint8')
    pred_local_th = pred_local_th.astype('uint8')

    pred_local_th = pred_global_th * pred_local_th
    pred_local_th = [cv2.dilate(x[:,:,0], np.ones((3,3))) for x in pred_local_th]
    pred_local_th= np.expand_dims(pred_local_th, -1)

    iou_global = int_uni_numpy(test_orig_gt, pred_global_th)
    iou_local = int_uni_numpy(test_orig_gt, pred_local_th)
    pred = pred_global_th + pred_local_th
    pred[pred >= 1] = 1
    iou = int_uni_numpy(test_orig_gt, pred)

    logging.info('=================================================================')
    logging.info('Total mIoU after Refinement = %f', iou)


