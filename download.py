import os
import shutil
import gdown
from pathlib import Path
import zipfile

WORKING_DIR = os.getcwd()
DEEPGLOBE_SEG_DIR = os.path.join(WORKING_DIR, 'src', 'deepglobe', 'seg-models', 'checkpoints', 'dlinknet')
DEEPGLOBE_REF_DIR = os.path.join(WORKING_DIR, 'src', 'deepglobe', 'refine-models', 'checkpoints', 'p64')
DUTS_SEG_DIR = os.path.join(WORKING_DIR, 'src', 'DUTS', 'seg-models', 'checkpoints', 'pfanet')
DUTS_REF_DIR = os.path.join(WORKING_DIR, 'src', 'DUTS', 'refine-models', 'checkpoints', 'p64')
KVASIR_SEG_DIR = os.path.join(WORKING_DIR, 'src', 'kvasir', 'seg-models', 'checkpoints', 'resunetplusplus')
KVASIR_REF_DIR = os.path.join(WORKING_DIR, 'src', 'kvasir', 'refine-models', 'checkpoints', 'p64')

# create model directories
Path(DEEPGLOBE_SEG_DIR).mkdir(parents=True, exist_ok=True)
Path(DEEPGLOBE_REF_DIR).mkdir(parents=True, exist_ok=True)
Path(DUTS_SEG_DIR).mkdir(parents=True, exist_ok=True)
Path(DUTS_REF_DIR).mkdir(parents=True, exist_ok=True)
Path(KVASIR_SEG_DIR).mkdir(parents=True, exist_ok=True)
Path(KVASIR_REF_DIR).mkdir(parents=True, exist_ok=True)

# model urls
deepglobe_url = 'https://drive.google.com/uc?id=1taCeROWb_bYMvfcCDtC2ftv_QfsWqmou'
duts_url = 'https://drive.google.com/uc?id=1Viu_mTI3aCvOKDLw8RRnMYFWvLZeBXqO'
kvasir_url = 'https://drive.google.com/uc?id=1Gj8Y43w5to-CdukJ0NlPzhrcUgvTNEa1'

# DeepGlobe
output = os.path.join(os.getcwd(), 'deepglobe_models.zip')
# download zip file
print('[INFO] Downloading DeepGlobe Models...')
gdown.download(deepglobe_url, output, quiet=False)
# Unzip
print('[INFO] Unzipping ...')
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(os.getcwd())
# remove downloaded zip file after unzipping
print('[INFO] Deleting downloaded Zip file ...')
os.remove(output)

shutil.move(os.path.join(WORKING_DIR, 'deepglobe_models', 'dlinknet_weights.th'), DEEPGLOBE_SEG_DIR)
shutil.move(os.path.join(WORKING_DIR, 'deepglobe_models', 'p64_weights.th'), DEEPGLOBE_REF_DIR)
os.rmdir(os.path.join(WORKING_DIR, 'deepglobe_models'))

# DUTS
output = os.path.join(os.getcwd(), 'DUTS_models.zip')
# download zip file
print('[INFO] Downloading DUTS Models...')
gdown.download(duts_url, output, quiet=False)
# Unzip
print('[INFO] Unzipping ...')
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(os.getcwd())
# remove downloaded zip file after unzipping
print('[INFO] Deleting downloaded Zip file ...')
os.remove(output)

shutil.move(os.path.join(WORKING_DIR, 'DUTS_models', 'best-model_epoch-204_mae-0.0505_loss-0.1370.pth'), DUTS_SEG_DIR)
shutil.move(os.path.join(WORKING_DIR, 'DUTS_models', 'p64_weights.th'), DUTS_REF_DIR)
os.rmdir(os.path.join(WORKING_DIR, 'DUTS_models'))

# Kvasir 
output = os.path.join(os.getcwd(), 'kvasir_models.zip')
# download zip file
print('[INFO] Downloading Kvasir Models...')
gdown.download(kvasir_url, output, quiet=False)
# Unzip
print('[INFO] Unzipping ...')
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(os.getcwd())
# remove downloaded zip file after unzipping
print('[INFO] Deleting downloaded Zip file ...')
os.remove(output)

shutil.move(os.path.join(WORKING_DIR, 'kvasir_models', 'resunetplusplus_weights.th'), KVASIR_SEG_DIR)
shutil.move(os.path.join(WORKING_DIR, 'kvasir_models', 'p64_weights.th'), KVASIR_REF_DIR)
os.rmdir(os.path.join(WORKING_DIR, 'kvasir_models'))
