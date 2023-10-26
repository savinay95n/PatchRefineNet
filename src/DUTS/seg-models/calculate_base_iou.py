# -----------------------------------------------------------------------------------------------------------------------
# calculate_base_iou.py: This is the code to calculate mIoU before refinement for DUTS dataset.
# Pfanet on DUTS test set.

# Usage: python src/DUTS/seg-models/calculate_base_iou.py --model pfanet

# Note: To independently run, always run this file from PatchRefineNetwork (root) directory.
# -----------------------------------------------------------------------------------------------------------------------
import os
import argparse
import numpy as np
import logging
from tqdm import tqdm
import time
from metrics import int_uni_numpy, threshold_predictions
import sys
import gdown
import zipfile

WORKING_DIR = os.getcwd()
DATA_DIR = os.path.join(WORKING_DIR, 'data', 'DUTS', 'seg-models')

LOG_DIR = os.path.join(WORKING_DIR, 'src', 'DUTS', 'seg-models', 'logs', 'pfanet')

# Command line arguments for which model to train
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Model Name: Choose one of [pfanet]', type=str, required=True)
args = parser.parse_args()

LOG_FILE_NAME = args.model + '_miou_log'

# Open Prediction Log File 
filename = os.path.join(LOG_DIR, LOG_FILE_NAME+'.log')
logging.basicConfig(filename=filename, level=logging.DEBUG)

logging.info('========================================================')
logging.info('Calculating Test mIoU of %s model on DUTS dataset', 'pfanet')
logging.info('========================================================')

test_input = np.load(os.path.join(DATA_DIR, 'test_input.npy'))
test_orig_gt = np.load(os.path.join(DATA_DIR, 'test_orig_gt.npy'))

pred_base_th = threshold_predictions(test_input, 0.5)
iou_base = int_uni_numpy(test_orig_gt, pred_base_th)

logging.info('Base Test mIoU before Refinement = %f', iou_base)
