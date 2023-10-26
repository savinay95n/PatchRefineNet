# -----------------------------------------------------------------------------------------------------------------------
# train.py: This is the code for training PRN
# Usage: python src/deepglobe/refine-models/train.py --base_model dlinknet --aux_model p64

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
from loss import RegionSpecificLoss, RegularizationLoss, bce_focal_loss
from torchsummary import summary
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import argparse
import time
from data_process import ImageFolder, ImageFolderPredictions
from metrics import int_uni, mae

# Working Directory (root)
WORKING_DIR = os.getcwd()

# Command line arguments for which model to train
parser = argparse.ArgumentParser()
parser.add_argument('--base_model', help='Model Name: Choose one of [dlinknet]', type=str, required=True)
parser.add_argument('--checkpoint', help='Checkpoint Name to start training from', type=str)
parser.add_argument('--aux_model', 
            help='Choose Aux Model with number of local patches (granularity of refinement) [p0, p4, p16, p64, p256, p1024]',
            type=str, required=True)
args = parser.parse_args()

# Base path of dataset
DATA_DIR = os.path.join(WORKING_DIR, 'data', 'deepglobe', 'refine-models', args.base_model)
# Log directory
LOG_DIR = os.path.join(WORKING_DIR, 'src', 'deepglobe', 'refine-models', 'logs', args.aux_model)
# Checkpoint directory
CHECKPOINT_DIR = os.path.join(WORKING_DIR, 'src', 'deepglobe', 'refine-models', 'checkpoints', args.aux_model)

# log file name
LOG_FILE_NAME = args.aux_model + '_train_log' + '_' + time.strftime("%Y%m%d-%H%M%S")
# Hardware Accelerator
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
LR = 8e-4
old_lr = LR
BATCH_SIZE = 1
loss_rst = RegionSpecificLoss(device=DEVICE)
loss_orig = RegularizationLoss(device=DEVICE)
model_name = args.aux_model + '_weights.th'

# Open Training Log File 
filename = os.path.join(LOG_DIR, LOG_FILE_NAME+'.log')
logging.basicConfig(filename=filename, level=logging.DEBUG)

logging.info('==========================================================================================')
logging.info('Training %s aux model on deepglobe dataset trained on base model %s', args.aux_model, args.base_model)
logging.info('==========================================================================================')
logging.info('Hyperparameters:')
logging.info('Learning Rate = %f', LR)
logging.info('Batch Size = %d', BATCH_SIZE)
logging.info('==========================================================================================')

# Invoke Aux model
if args.aux_model == 'p0':
    model = ThreshNetP0().to(DEVICE)
if args.aux_model == 'p4':
    model = ThreshNetP4().to(DEVICE)
if args.aux_model == 'p64':
    model = ThreshNetP64().to(DEVICE)

if args.checkpoint:
    logging.info('Loading Saved Checkpoint ...')
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, model_name)))
    print("=================================================================")
summary(model)
optimizer = torch.optim.Adam(params=model.parameters(), lr = LR)

# Dataset
print('[INFO] Loading data ...')

train_input = np.load(os.path.join(DATA_DIR, 'val_pred.npy'))
train_orig_gt = np.load(os.path.join(DATA_DIR, 'val_orig_gt.npy'))

if args.aux_model == 'p0':
    train_gt = np.load(os.path.join(DATA_DIR, 'train_aux_gt_p_512.npy'))
if args.aux_model == 'p4':
    train_gt = np.load(os.path.join(DATA_DIR, 'train_aux_gt_p_256.npy'))
if args.aux_model == 'p64':
    train_gt = np.load(os.path.join(DATA_DIR, 'train_aux_gt_p_64.npy'))
print('[INFO] Loaded data ...')
print('Training Data: ', train_input.shape, train_gt.shape, train_orig_gt.shape, np.unique(train_gt), np.unique(train_orig_gt))

# Create train dataset
train_dataset = ImageFolder(train_input, train_gt, train_orig_gt)
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True)

# Train
tic = time.time()
no_optim = 0
total_epoch = 300
train_epoch_best_loss = 1000000
trainSteps = len(train_dataset) // BATCH_SIZE

if args.aux_model == 'p0':
    for epoch in tqdm(range(1, total_epoch + 1), position = 0, leave=True):
        model.train()

        train_epoch_loss = 0
        train_epoch_iou_rst = 0
        train_epoch_iou_orig = 0

        # loop over the training set
        for (x, y1, y2) in tqdm(train_dataloader, leave=True, position=0):
            x, y1, y2 = x.type(torch.FloatTensor), y1.type(torch.FloatTensor), y2.type(torch.FloatTensor)
            # Pytorch takes (N,C,H,W) format
            x = x.permute(0,3,1,2)
            y1 = y1.permute(0,3,1,2)
            y2 = y2.permute(0,3,1,2)
            (x, y1, y2) = (x.to(DEVICE), y1.to(DEVICE), y2.to(DEVICE))
            optimizer.zero_grad()
            pred = model(x)
            rst_loss = loss_rst(y1, pred)
            reg_loss = loss_orig(y2, pred)
            loss = 0.7 * rst_loss + 0.3* reg_loss
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss
            train_epoch_iou_rst += int_uni(y1, pred, DEVICE)   
            train_epoch_iou_orig += int_uni(y2, pred, DEVICE)

        train_epoch_loss /= trainSteps
        train_epoch_iou_rst /= trainSteps
        train_epoch_iou_orig /= trainSteps

        logging.info('=================================================================')
        logging.info('epoch: %d , time: %d',epoch, int(time.time()-tic))
        logging.info('train_loss: %f', train_epoch_loss)
        logging.info('train_epoch_iou_rst: %f', train_epoch_iou_rst)
        logging.info('train_epoch_iou_orig: %f', train_epoch_iou_orig)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            logging.info('=================================================================')
            logging.info('Saving best model checkpoint ...')
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, model_name))
        if no_optim > 8:
            logging.info('=================================================================')
            logging.info('Early Stop at %d epoch', epoch)
            break
        if no_optim > 4:
            if old_lr < 5e-8:
                break
            model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, model_name)))
            new_lr = old_lr / 1.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            logging.info('=================================================================')
            logging.info('update learning rate: %f -> %f', old_lr, new_lr)
            old_lr = new_lr
    logging.info('=================================================================')
    logging.info('Finish!')

else:
    for epoch in tqdm(range(1, total_epoch + 1), position = 0, leave=True):
        model.train()

        train_epoch_loss = 0
        train_epoch_iou_rst_global = 0
        train_epoch_iou_rst_local = 0
        train_epoch_iou_rst = 0
        train_epoch_iou_orig_global = 0
        train_epoch_iou_orig_local = 0
        train_epoch_iou_orig = 0
 
        # loop over the training set
        for (x, y1, y2) in tqdm(train_dataloader, leave=True, position=0):
            x, y1, y2 = x.type(torch.FloatTensor), y1.type(torch.FloatTensor), y2.type(torch.FloatTensor)
            # Pytorch takes (N,C,H,W) format
            x = x.permute(0,3,1,2)
            y1 = y1.permute(0,3,1,2)
            y2 = y2.permute(0,3,1,2)
            (x, y1, y2) = (x.to(DEVICE), y1.to(DEVICE), y2.to(DEVICE))
            optimizer.zero_grad()
            pred_global, pred_local = model(x)
            rst_loss_global = loss_rst(y1, pred_global)
            rst_loss_local = loss_rst(y1, pred_local)
            rst_loss = 0.5 * rst_loss_global + 0.5 * rst_loss_local
            reg_loss_global = loss_orig(y2, pred_global)
            reg_loss_local = loss_orig(y2, pred_local)
            reg_loss = 0.5 * reg_loss_global + 0.5 * reg_loss_local
            loss = 0.7 * rst_loss + 0.3* reg_loss
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss
            train_epoch_iou_rst_global += int_uni(y1, pred_global, DEVICE)   
            train_epoch_iou_rst_local += int_uni(y1, pred_local, DEVICE) 
            train_epoch_iou_rst += (0.5 * int_uni(y1, pred_global, DEVICE) + 0.5 * int_uni(y1, pred_local, DEVICE))
            train_epoch_iou_orig_global += int_uni(y2, pred_global, DEVICE)
            train_epoch_iou_orig_local += int_uni(y2, pred_local, DEVICE)
            train_epoch_iou_orig += (0.5 * int_uni(y2, pred_global, DEVICE) + 0.5 * int_uni(y2, pred_local, DEVICE))

        train_epoch_loss /= trainSteps
        train_epoch_iou_rst /= trainSteps
        train_epoch_iou_orig /= trainSteps
        logging.info('=================================================================')
        logging.info('epoch: %d , time: %d',epoch, int(time.time()-tic))
        logging.info('train_loss: %f', train_epoch_loss)
        logging.info('train_epoch_iou_rst: %f', train_epoch_iou_rst)
        logging.info('train_epoch_iou_orig: %f', train_epoch_iou_orig)

        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            logging.info('=================================================================')
            logging.info('Saving best model checkpoint ...')
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, model_name))
        if no_optim > 8:
            logging.info('=================================================================')
            logging.info('Early Stop at %d epoch', epoch)
            break
        if no_optim > 4:
            if old_lr < 5e-8:
                break
            model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, model_name)))
            new_lr = old_lr / 1.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
            logging.info('=================================================================')
            logging.info('update learning rate: %f -> %f', old_lr, new_lr)
            old_lr = new_lr
    logging.info('=================================================================')
    logging.info('Finish!')