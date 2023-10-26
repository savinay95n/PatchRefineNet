# -----------------------------------------------------------------------------------------------------------------------
# generate_aux_training_data.py: This is the code to run inference on DeepGlobe dataset and save the validation confidence maps.

# Usage: python src/deepglobe/seg-models/generate_aux_training_data.py --model dlinknet

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
from networks.dlinknet import DinkNet34
import warnings
warnings.filterwarnings("ignore")
import logging
import argparse
from tqdm import tqdm
from data_process import ImageFolder, ImageFolderPredictions, sample_plots, threshold_predictions
from metrics import int_uni_numpy, int_uni
import pandas as pd
import torch.nn.functional as F


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

# Checkpoint directory
CHECKPOINT_DIR = os.path.join(WORKING_DIR, 'src', 'deepglobe', 'seg-models', 'checkpoints', args.model)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# get ids of validation images from csv file
val_dataframe = list(pd.read_csv(os.path.join(DATA_DIR, 'val.csv')).iloc[:,0])
vallist = list(map(lambda x: x[:-8], val_dataframe))

model_name = args.model + '_weights.th'

BATCH_SIZE = 4
# Invoke UNet model
if args.model == 'dlinknet':
    model = DinkNet34().to(DEVICE)
print('Loading Saved Checkpoint ...')
model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, model_name)))
print("=================================================================")
model.eval()

val_pred = []
val_gt = []

val_dataset = ImageFolderPredictions(vallist, os.path.join(DATA_DIR, 'train'), os.path.join(DATA_DIR, 'train_labels'))
val_data_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
val_data_loader_iter = iter(val_data_loader)
for img, mask, orig_img in tqdm(val_data_loader_iter, position=0, leave=True):
    orig_img = orig_img.cpu().detach().numpy()
    img, mask = V(img.to(DEVICE)), V(mask.to(DEVICE))
    pred = model(img)
    pred = pred.permute(0,2,3,1)
    mask = mask.permute(0,2,3,1)
    img = img.permute(0,2,3,1)
    pred = pred.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy()
    val_pred.append(pred[0])
    val_gt.append(mask[0])

val_pred = np.array(val_pred)
val_gt = np.array(val_gt)

print('Saving aux data for training ...')
np.save(os.path.join(DATA_DIR2, 'val_pred'), val_pred)    
np.save(os.path.join(DATA_DIR2, 'val_orig_gt'), val_gt) 
print('Saved ...')
