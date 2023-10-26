# -----------------------------------------------------------------------------------------------------------------------
# generate_aux_training_data.py: This is the code to run inference for base segmentation network on kvasir-seg dataset.

# Usage: python src/kvasir/seg-models/generate_aux_training_data.py --model resunetplusplus

# Note: To independently run, always run this file from PatchRefineNetwork (root) directory.
# -----------------------------------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.autograd import Variable as V
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from time import time
from networks.resunetplusplus import ResUnetPlusPlus
import warnings
warnings.filterwarnings("ignore")
import logging
import argparse
from tqdm import tqdm
from metrics import int_uni_numpy
from data_process import ImageFolder, ImageFolderPredictions, sample_plots, int_uni, threshold_predictions, int_uni_numpy
import pandas as pd
import torch.nn.functional as F
from matplotlib import colors
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1 import ImageGrid


# Command line arguments for which model to train
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Model Name: Choose one of [unet, resunetplusplus]', type=str, required=True)
args = parser.parse_args()

#CONSTANTS
# Working Directory (root)
WORKING_DIR = os.getcwd()
# Base path of dataset
DATA_DIR = os.path.join(WORKING_DIR, 'data', 'kvasir', 'seg-models')
DATA_DIR2 = os.path.join(WORKING_DIR, 'data', 'kvasir', 'refine-models', args.model)

# Log directory
LOG_DIR = os.path.join(WORKING_DIR, 'src', 'kvasir', 'seg-models', 'logs', args.model)
# Checkpoint directory
CHECKPOINT_DIR = os.path.join(WORKING_DIR, 'src', 'kvasir', 'seg-models', 'checkpoints', args.model)
# Hardware Accelerator
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# get ids of validation images from csv file
val_dataframe = list(pd.read_csv(os.path.join(DATA_DIR, 'val.csv')).iloc[:,0])
vallist = list(map(lambda x: x[:-4], val_dataframe))
# get ids of test images from csv file
test_dataframe = list(pd.read_csv(os.path.join(DATA_DIR, 'test.csv')).iloc[:,0])
testlist = list(map(lambda x: x[:-4], test_dataframe))

model_name = args.model + '_weights.th'

BATCH_SIZE = 1
if args.model == 'resunetplusplus':
    model = ResUnetPlusPlus().to(DEVICE)
logging.info('Loading Saved Checkpoint ...')
model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, model_name)))
model.eval()

# Create train dataset
test_dataset = ImageFolder(testlist, os.path.join(DATA_DIR, 'train'), os.path.join(DATA_DIR, 'train_labels'))
test_data_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE)

# Create val dataset
val_dataset = ImageFolder(vallist, os.path.join(DATA_DIR, 'train'), os.path.join(DATA_DIR, 'train_labels'))
val_data_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE)

val_pred = []
val_gt = []
val_iou_list = []
val_iou_edge_list = []
count = -1
with torch.no_grad():
    val_data_loader_iter = iter(val_data_loader)
    for img, mask, orig_img in tqdm(val_data_loader_iter, position=0, leave=True):
        count += 1
        orig_img = orig_img.cpu().detach().numpy()[0]
        img, mask = V(img.to(DEVICE)), V(mask.to(DEVICE))
        pred = model(img)
        pred_th = torch.zeros(pred.shape).to(DEVICE)
        pred_th[pred >= 0.5] = 1
        pred_th[pred < 0.5] = 0
        
        pred = pred.permute(0,2,3,1).detach().cpu().numpy()[0]
        mask = mask.permute(0,2,3,1).detach().cpu().numpy()[0]
        pred_th = pred_th.permute(0,2,3,1).detach().cpu().numpy()[0]

        val_pred.append(pred)
        val_gt.append(mask)

val_pred = np.array(val_pred)
val_gt = np.array(val_gt)

np.save(os.path.join(DATA_DIR2, 'val_pred.npy'), val_pred)
np.save(os.path.join(DATA_DIR2, 'val_orig_gt.npy'), val_gt)
