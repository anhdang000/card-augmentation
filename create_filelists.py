import os
from os.path import join
import numpy as np
from PIL import Image
import mmcv

# convert dataset annotation to semantic segmentation map
DATA_ROOT = '../card_dataset'
IMAGE_DIR = 'images'
LABEL_DIR = 'labels'
# define class and plaette for better visualization
classes = (
    'id_card_front', 
    'id_card_back', 
    'id_paper_front', 
    'id_paper_back', 
    'driver_license_front', 
    'driver_license_back'
    )
PALETTE = [
        [128, 0, 0], [0, 128, 0], [128, 128, 0], 
        [0, 0, 128], [128, 0, 128], [0, 128, 128]
        ]

IMAGE_FILES = os.listdir(join(DATA_ROOT, IMAGE_DIR))
FILE_IDS = [image_file.split('.')[0] for image_file in IMAGE_FILES if '_aug' not in image_file]
NUM_FILES = len(FILE_IDS)
print(NUM_FILES)
# split train/val set randomly
trainval_split = int(NUM_FILES * 0.8)
train_files = FILE_IDS[:trainval_split]
val_files = FILE_IDS[trainval_split:]

with open(join(DATA_ROOT, 'splits', 'train.txt'), 'w') as f:
    f.write('\n'.join(train_files))

with open(join(DATA_ROOT, 'splits', 'val.txt'), 'w') as f:
    f.write('\n'.join(val_files))