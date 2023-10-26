# -----------------------------------------------------------------------------------------------------------------------
# calculate_base_iou.py: This is the code to calculate mIoU before refinement for deepglobe dataset.

# Usage: python src/deepglobe/seg-models/calculate_base_iou.py --model dlinknet

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
DATA_DIR = os.path.join(WORKING_DIR, 'data', 'deepglobe', 'seg-models')

LOG_DIR = os.path.join(WORKING_DIR, 'src', 'deepglobe', 'seg-models', 'logs', 'dlinknet')

# Command line arguments for which model to train
parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Model Name: Choose one of [dlinknet]', type=str, required=True)
args = parser.parse_args()

LOG_FILE_NAME = args.model + '_miou_log'

# Open Prediction Log File 
filename = os.path.join(LOG_DIR, LOG_FILE_NAME+'.log')
logging.basicConfig(filename=filename, level=logging.DEBUG)

logging.info('========================================================')
logging.info('Calculating Test mIoU of %s model on deepglobe dataset', 'dlinknet')
logging.info('========================================================')

test_input = np.load(os.path.join(DATA_DIR, 'test_input.npy'))
test_orig_gt = np.load(os.path.join(DATA_DIR, 'test_orig_gt.npy'))

pred_base_th = threshold_predictions(test_input, 0.5)
iou_base = int_uni_numpy(test_orig_gt, pred_base_th)

logging.info('Base Test mIoU before Refinement = %f', iou_base)
