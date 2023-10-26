# ----------------------------------------------------------------------------------------------------------
# download_data.py: This file downloads and unzips data used for training base networks from Google Drive.

# The following data sets are downloaded:
#     1. DeepGlobe Road Extraction
#     2. Kvasir polyp segmentation dataset
#     3. DUTS salient object detecion dataset

# Usage: python data_utils/download_data.py

# Note: Always run this file from PatchRefineNetwork (root) directory. 
# ----------------------------------------------------------------------------------------------------------

import sys
import gdown
import argparse
import os
import zipfile

# Check if data directory exists:
if not os.path.exists(os.path.join(os.getcwd(), 'data')):
    os.makedirs(os.path.join(os.getcwd(), 'data'))
    print('[INFO] Create data directory ...')

# download
def download():
    url = 'https://drive.google.com/uc?id=1C8xFJF6dHCGuDZU3XIu8Gg0_jroycnCF'
    output = os.path.join(os.getcwd(), 'data.zip')
    # download zip file
    print('[INFO] Downloading ...')
    gdown.download(url, output, quiet=False)
    # Unzip
    print('[INFO] Unzipping ...')
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall(os.getcwd())
    # remove downloaded zip file after unzipping
    print('[INFO] Deleting downloaded Zip file ...')
    os.remove(output)

if __name__ == "__main__":
    download()


