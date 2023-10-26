# -----------------------------------------------------------------------------------------------------------------------
# csv_generator.py: This file was used to generate train, val and test CSV files for the downloaded data.

# Usage: python data_utils/csv_generator.py

# Note: Always run this file from PatchRefineNetwork (root) directory.
# Note: CSV files have already been generated and is available inside data.zip when downloaded using download_data.py
# -----------------------------------------------------------------------------------------------------------------------

import os
from sklearn.model_selection import train_test_split
import pandas as pd

# Root Directory
WORKING_DIR = os.getcwd()
# Data Directory
DATA_DIR = os.path.join(WORKING_DIR, 'data')
# DeepGlobe data directory
DEEPGLOBE_DIR = os.path.join(DATA_DIR, 'deepglobe', 'seg-models')
# ISIC data directory
DUTS_DIR = os.path.join(DATA_DIR, 'DUTS', 'seg-models')
# KVASIR data directory
KVASIR_DIR = os.path.join(DATA_DIR, 'kvasir', 'seg-models')

# Deepglobe dataset
def create_csv_deepglobe():
    IMG_DIR = os.path.join(DEEPGLOBE_DIR, 'train')
    TARGET_DIR = os.path.join(DEEPGLOBE_DIR, 'train_labels')
    img_files = sorted(os.listdir(IMG_DIR))
    target_files = sorted(os.listdir(TARGET_DIR))
    train_img_files, test_img_files, train_target_files, test_target_files = train_test_split(img_files,
                                                                                            target_files, test_size = 0.2, random_state = 29)
    val_img_files, test_img_files, val_target_files, test_target_files = train_test_split(test_img_files,
                                                                                            test_target_files, test_size = 0.2, random_state = 29)

    train_dict = {}
    test_dict = {}
    val_dict = {}
    train_dict['img'] = train_img_files
    train_dict['target'] = train_target_files
    val_dict['img'] = val_img_files
    val_dict['target'] = val_target_files
    test_dict['img'] = test_img_files
    test_dict['target'] = test_target_files

    if 'desktop.ini' in train_dict['img']:
        train_dict['img'].remove('desktop.ini')
    if 'desktop.ini' in train_dict['target']:
        train_dict['target'].remove('desktop.ini')

    if 'desktop.ini' in val_dict['img']:
        val_dict['img'].remove('desktop.ini')
    if 'desktop.ini' in val_dict['target']:
        val_dict['target'].remove('desktop.ini')
        
    if 'desktop.ini' in test_dict['img']:
        test_dict['img'].remove('desktop.ini')
    if 'desktop.ini' in test_dict['target']:
        test_dict['target'].remove('desktop.ini')

    train_df = pd.DataFrame.from_dict(train_dict)
    val_df = pd.DataFrame.from_dict(val_dict)
    test_df = pd.DataFrame.from_dict(test_dict)

    train_df.to_csv(os.path.join(DEEPGLOBE_DIR, 'train.csv'), index = False)
    val_df.to_csv(os.path.join(DEEPGLOBE_DIR, 'val.csv'), index = False)
    test_df.to_csv(os.path.join(DEEPGLOBE_DIR, 'test.csv'), index = False)

    print('[INFO] Train, Validation and Test csv files have been created for DeepGlobe dataset ...')

# Kvasir dataset
def create_csv_kvasir():
    train_files = open(os.path.join(KVASIR_DIR, 'train.txt'))
    train_data = train_files.read()
    train_data_list = train_data.split("\n")
    train_data_list = [x + '.jpg' for x in train_data_list]

    test_files = open(os.path.join(KVASIR_DIR, 'test.txt'))
    test_data = test_files.read()
    test_data_list = test_data.split("\n")
    test_data_list = [x + '.jpg' for x in test_data_list]

    val_img_files, test_img_files, val_target_files, test_target_files = train_test_split(test_data_list,
                                                                                            test_data_list, test_size = 0.2, random_state = 29)
    train_img_files, train_target_files = train_data_list, train_data_list

    train_dict = {}
    test_dict = {}
    val_dict = {}
    train_dict['img'] = train_img_files
    train_dict['target'] = train_target_files
    val_dict['img'] = val_img_files
    val_dict['target'] = val_target_files
    test_dict['img'] = test_img_files
    test_dict['target'] = test_target_files

    if 'desktop.ini' in train_dict['img']:
        train_dict['img'].remove('desktop.ini')
    if 'desktop.ini' in train_dict['target']:
        train_dict['target'].remove('desktop.ini')

    if 'desktop.ini' in val_dict['img']:
        val_dict['img'].remove('desktop.ini')
    if 'desktop.ini' in val_dict['target']:
        val_dict['target'].remove('desktop.ini')
        
    if 'desktop.ini' in test_dict['img']:
        test_dict['img'].remove('desktop.ini')
    if 'desktop.ini' in test_dict['target']:
        test_dict['target'].remove('desktop.ini')

    train_df = pd.DataFrame.from_dict(train_dict)
    val_df = pd.DataFrame.from_dict(val_dict)
    test_df = pd.DataFrame.from_dict(test_dict)

    train_df.to_csv(os.path.join(KVASIR_DIR, 'train.csv'), index = False)
    val_df.to_csv(os.path.join(KVASIR_DIR, 'val.csv'), index = False)
    test_df.to_csv(os.path.join(KVASIR_DIR, 'test.csv'), index = False)
    # val and test comes from test set, with 80-20 split
    print('[INFO] Train, Validation and Test csv files have been created for Kvasir dataset ...')

# DUTS dataset
def create_csv_duts():
    TRAIN_IMG_DIR = os.path.join(DUTS_DIR, 'DUTS-TR', 'DUTS-TR-Image')
    TRAIN_TARGET_DIR = os.path.join(DUTS_DIR, 'DUTS-TR', 'DUTS-TR-Mask')
    TEST_IMG_DIR = os.path.join(DUTS_DIR, 'DUTS-TE', 'DUTS-TE-Image')
    TEST_TARGET_DIR = os.path.join(DUTS_DIR, 'DUTS-TE', 'DUTS-TE-Mask')
    train_img_files = sorted(os.listdir(TRAIN_IMG_DIR))
    train_target_files = sorted(os.listdir(TRAIN_TARGET_DIR))
    test_img_files = sorted(os.listdir(TEST_IMG_DIR))
    test_target_files = sorted(os.listdir(TEST_TARGET_DIR))

    val_img_files, test_img_files, val_target_files, test_target_files = train_test_split(test_img_files,
                                                                                            test_target_files, test_size = 0.2, random_state = 29)

    train_dict = {}
    test_dict = {}
    val_dict = {}
    train_dict['img'] = train_img_files
    train_dict['target'] = train_target_files
    val_dict['img'] = val_img_files
    val_dict['target'] = val_target_files
    test_dict['img'] = test_img_files
    test_dict['target'] = test_target_files

    if 'desktop.ini' in train_dict['img']:
        train_dict['img'].remove('desktop.ini')
    if 'desktop.ini' in train_dict['target']:
        train_dict['target'].remove('desktop.ini')

    if 'desktop.ini' in val_dict['img']:
        val_dict['img'].remove('desktop.ini')
    if 'desktop.ini' in val_dict['target']:
        val_dict['target'].remove('desktop.ini')
        
    if 'desktop.ini' in test_dict['img']:
        test_dict['img'].remove('desktop.ini')
    if 'desktop.ini' in test_dict['target']:
        test_dict['target'].remove('desktop.ini')       

                                                                                     
    train_df = pd.DataFrame.from_dict(train_dict)
    val_df = pd.DataFrame.from_dict(val_dict)
    test_df = pd.DataFrame.from_dict(test_dict)

    train_df.to_csv(os.path.join(DUTS_DIR, 'train.csv'), index = False)
    val_df.to_csv(os.path.join(DUTS_DIR, 'val.csv'), index = False)
    test_df.to_csv(os.path.join(DUTS_DIR, 'test.csv'), index = False)

    # val and test comes from test set, with 80-20 split
    print('[INFO] Train, Validation and Test csv files have been created for DUTS dataset ...')

if __name__ == "__main__":
    create_csv_deepglobe()
    create_csv_kvasir()
    create_csv_duts()