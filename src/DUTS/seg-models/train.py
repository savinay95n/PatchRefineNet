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
from metrics import int_uni, mae

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
LOG_FILE_NAME = args.model + '_log' + '_' + time.strftime("%Y%m%d-%H%M%S")
# Hardware Accelerator
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data root directories
csv_root = os.path.join(WORKING_DIR, 'data', 'DUTS', 'seg-models')
train_input_root = os.path.join(csv_root, 'DUTS-TR', 'DUTS-TR', 'DUTS-TR-Image')
train_mask_root = os.path.join(csv_root, 'DUTS-TR', 'DUTS-TR', 'DUTS-TR-Mask')
val_input_root = os.path.join(csv_root, 'DUTS-TE', 'DUTS-TE', 'DUTS-TE-Image')
val_mask_root = os.path.join(csv_root, 'DUTS-TE', 'DUTS-TE', 'DUTS-TE-Mask')

# Hyperparameters
LR = 0.0004
old_lr = LR
BATCH_SIZE = 4
img_size = 256
alpha_sal = 0.7
wbce_w0 = 1.0
wbce_w1 = 1.15
wd = 0.
log_interval = 250

model_name = args.model + '_weights.th'

# Open Training Log File 
filename = os.path.join(LOG_DIR, LOG_FILE_NAME+'.log')
logging.basicConfig(filename=filename, level=logging.DEBUG)

logging.info('========================================================')
logging.info('Training %s model on DUTS dataset', args.model)
logging.info('========================================================')
logging.info('Hyperparameters:')
logging.info('Learning Rate = %f', LR)
logging.info('Batch Size = %d', BATCH_SIZE)
logging.info('========================================================')

model = SODModel().to(DEVICE)
if args.checkpoint:
    logging.info('Loading Saved Checkpoint ...')
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, model_name)))
    print("=================================================================")

criterion = EdgeSaliencyLoss(device=DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=wd)

summary(model)

# Loading data
train_data = SODLoader(train_input_root, train_mask_root, csv_root, 'train.csv', augment_data=True, target_size=img_size)
val_data = SODLoader(val_input_root, val_mask_root, csv_root, 'val.csv', augment_data=False, target_size=img_size)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

tic = time.time()
no_optim = 0
total_epoch = 391
best_val_mae = float('inf') 

for epoch in tqdm(range(1, total_epoch + 1), position = 0, leave=True):
    # train
    model.train()
    train_data_loader_iter = iter(train_dataloader)
    train_epoch_loss = 0
    train_epoch_iou = 0
    train_epoch_mae = 0
    for inp_imgs, gt_masks in tqdm(train_dataloader, leave=True, position=0):
        inp_imgs = inp_imgs.to(DEVICE)
        gt_masks = gt_masks.to(DEVICE)
        optimizer.zero_grad()
        pred_masks, ca_act_reg = model(inp_imgs)
        loss = criterion(pred_masks, gt_masks) + ca_act_reg  # Activity regularizer from Channel-wise Att.
        loss.backward()
        optimizer.step()
        
        train_epoch_loss += loss
        train_epoch_iou += int_uni(gt_masks, pred_masks, DEVICE)   
        train_epoch_mae += mae(gt_masks, pred_masks)

    train_epoch_loss /= len(train_data_loader_iter)
    train_epoch_iou /= len(train_data_loader_iter)
    train_epoch_mae /= len(train_data_loader_iter)

    logging.info('=================================================================')
    logging.info('epoch: %d , time: %d',epoch, int(time.time()-tic))
    logging.info('train_loss: %f', train_epoch_loss)
    logging.info('train_iou: %f', train_epoch_iou)
    logging.info('train_mae: %f', train_epoch_mae)

    # validation
    with torch.no_grad():
        model.eval()
        val_data_loader_iter = iter(val_dataloader)
        val_epoch_loss = 0
        val_epoch_iou = 0
        val_epoch_mae = 0
        for inp_imgs, gt_masks in tqdm(val_data_loader_iter, leave=True, position=0):
            inp_imgs = inp_imgs.to(DEVICE)
            gt_masks = gt_masks.to(DEVICE)
            pred_masks, ca_act_reg = model(inp_imgs)
            loss = criterion(pred_masks, gt_masks) + ca_act_reg

            val_epoch_loss += loss
            val_epoch_iou += int_uni(gt_masks, pred_masks, DEVICE)   
            val_epoch_mae += mae(gt_masks, pred_masks)

        val_epoch_loss /= len(val_data_loader_iter)
        val_epoch_iou /= len(val_data_loader_iter)
        val_epoch_mae /= len(val_data_loader_iter)

    logging.info('=================================================================')
    logging.info('epoch: %d , time: %d',epoch, int(time.time()-tic))
    logging.info('val_loss: %f', val_epoch_loss)
    logging.info('val_iou: %f', val_epoch_iou)
    logging.info('val_mae: %f', val_epoch_mae)

    if val_epoch_mae >= best_val_mae:
        no_optim += 1
    else:
        no_optim = 0
        best_val_mae = val_epoch_mae
        logging.info('=================================================================')
        logging.info('Saving best model checkpoint ...')
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, model_name))
    if no_optim > 8:
        logging.info('=================================================================')
        logging.info('Early Stop at %d epoch', epoch)
        break
    if no_optim > 4:
        if old_lr < 5e-7:
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
