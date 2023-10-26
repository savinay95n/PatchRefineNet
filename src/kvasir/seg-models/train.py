# -----------------------------------------------------------------------------------------------------------------------
# generate_aux_training_data.py: This is the code to train base segmentation network on kvasir-seg dataset.

# Usage: python src/kvasir/seg-models/train.py --model resunetplusplus

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
from networks.resunetplusplus import ResUnetPlusPlus
from metrics import dice_bce_loss, int_uni
from data_process import ImageFolder
from torchsummary import summary
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import argparse
import time

#CONSTANTS
# Working Directory (root)
WORKING_DIR = os.getcwd()
# Base path of dataset
DATA_DIR = os.path.join(WORKING_DIR, 'data', 'kvasir', 'seg-models')

# Command line arguments for which model to train
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Model Name: Choose one of [unet, resunetplusplus]', type=str, required=True)
parser.add_argument('--checkpoint', help='Checkpoint Name', type=str)
args = parser.parse_args()

# Log directory
LOG_DIR = os.path.join(WORKING_DIR, 'src', 'kvasir', 'seg-models', 'logs', args.model)
# Checkpoint directory
CHECKPOINT_DIR = os.path.join(WORKING_DIR, 'src', 'kvasir', 'seg-models', 'checkpoints', args.model)
# Output directory
OUTPUT_DIR = os.path.join(WORKING_DIR, 'src', 'kvasir', 'seg-models', 'outputs', args.model)
# log file name
LOG_FILE_NAME = args.model + '_log' + '_' + time.strftime("%Y%m%d-%H%M%S")
# Hardware Accelerator
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# get ids of train images from csv file
train_dataframe = list(pd.read_csv(os.path.join(DATA_DIR, 'train.csv')).iloc[:,0])
trainlist = list(map(lambda x: x[:-4], train_dataframe))
# get ids of validation images from csv file
val_dataframe = list(pd.read_csv(os.path.join(DATA_DIR, 'val.csv')).iloc[:,0])
vallist = list(map(lambda x: x[:-4], val_dataframe))

# Hyperparameters
LR = 2e-4
old_lr = LR
BATCH_SIZE = 2
loss = dice_bce_loss()
model_name = args.model + '_weights.th'

# Create train dataset
train_dataset = ImageFolder(trainlist, os.path.join(DATA_DIR, 'train'), os.path.join(DATA_DIR, 'train_labels'))
train_data_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True)

# Create val dataset
val_dataset = ImageFolder(vallist, os.path.join(DATA_DIR, 'train'), os.path.join(DATA_DIR, 'train_labels'))
val_data_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True)

# Open Training Log File 
filename = os.path.join(LOG_DIR, LOG_FILE_NAME+'.log')
logging.basicConfig(filename=filename, level=logging.DEBUG)

logging.info('========================================================')
logging.info('Training %s model on deepglobe dataset', args.model)
logging.info('========================================================')
logging.info('Hyperparameters:')
logging.info('Learning Rate = %f', LR)
logging.info('Batch Size = %d', BATCH_SIZE)
logging.info('========================================================')
# Invoke UNet model
if args.model == 'unet':
    model = Unet().to(DEVICE)
elif args.model == 'resunetplusplus':
    model = ResUnetPlusPlus().to(DEVICE)
if args.checkpoint:
    logging.info('Loading Saved Checkpoint ...')
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, model_name)))
    print("=================================================================")
    
summary(model)
optimizer = torch.optim.Adam(params=model.parameters(), lr = LR)


tic = time.time()
no_optim = 0
total_epoch = 300
train_epoch_best_loss = 100.
val_epoch_best_loss = 100
for epoch in tqdm(range(1, total_epoch + 1), position = 0, leave=True):
    model.train()
    train_data_loader_iter = iter(train_data_loader)
    train_epoch_loss = 0
    train_epoch_iou = 0
    # loop over the training set
    for img, mask, _ in tqdm(train_data_loader_iter, position=0, leave=True):
        img, mask = V(img.to(DEVICE)), V(mask.to(DEVICE))
        pred = model(img)
        train_loss = loss(mask, pred)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        train_epoch_loss += train_loss
        train_epoch_iou += int_uni(mask, pred, DEVICE)
    train_epoch_loss /= len(train_data_loader_iter)
    train_epoch_iou /= len(train_data_loader_iter)
    logging.info('=================================================================')
    logging.info('epoch: %d , time: %d',epoch, int(time.time()-tic))
    logging.info('train_loss: %f', train_epoch_loss)
    logging.info('train_iou: %f', train_epoch_iou)
    # switch off autograd
    with torch.no_grad():
        model.eval()
        val_data_loader_iter = iter(val_data_loader)
        val_epoch_loss = 0
        val_epoch_iou = 0
        # loop over the training set
        for img, mask, _ in tqdm(val_data_loader_iter, position=0, leave=True):
            img, mask = V(img.to(DEVICE)), V(mask.to(DEVICE))
            pred = model(img)
            val_loss = loss(mask, pred)
            val_epoch_loss += val_loss
            val_epoch_iou += int_uni(mask, pred, DEVICE)
        val_epoch_loss /= len(val_data_loader_iter)
        val_epoch_iou /= len(val_data_loader_iter)
    logging.info('=================================================================')
    logging.info('epoch: %d , time: %d',epoch, int(time.time()-tic))
    logging.info('val_loss: %f', val_epoch_loss)
    logging.info('val_iou: %f', val_epoch_iou)

    if val_epoch_loss >= val_epoch_best_loss:
        no_optim += 1
    else:
        no_optim = 0
        val_epoch_best_loss = val_epoch_loss
        logging.info('=================================================================')
        logging.info('Saving best model checkpoint ...')
        torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, model_name))
    if no_optim > 12:
        logging.info('=================================================================')
        logging.info('Early Stop at %d epoch', epoch)
        break
    if no_optim > 6:
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
