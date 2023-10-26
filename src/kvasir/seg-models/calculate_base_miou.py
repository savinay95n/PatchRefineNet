# -----------------------------------------------------------------------------------------------------------------------
# calculate_base_iou.py: This is the code to calculate mIoU before refinement for kvasir-seg dataset.

# Usage: python src/kvasir/seg-models/calculate_base_iou.py --model resunetplusplus

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

# Log directory
LOG_DIR = os.path.join(WORKING_DIR, 'src', 'kvasir', 'seg-models', 'logs', args.model)

LOG_FILE_NAME = args.model + '_miou_log'

# Open Prediction Log File 
filename = os.path.join(LOG_DIR, LOG_FILE_NAME+'.log')
logging.basicConfig(filename=filename, level=logging.DEBUG)

logging.info('========================================================')
logging.info('Calculating Test mIoU of %s model on kvasir dataset', 'resunet++')
logging.info('========================================================')

test_input = np.load(os.path.join(DATA_DIR, 'test_input.npy'))
test_orig_gt = np.load(os.path.join(DATA_DIR, 'test_orig_gt.npy'))
iou_base = int_uni_numpy(test_orig_gt, test_input)
logging.info('Base Test mIoU before Refinement = %f', np.mean(iou_base))
