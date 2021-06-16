import sys
import os
from os.path import join
import cv2
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import imageio
import random


DATA_ROOT = '../card_dataset'
IMAGE_DIR = 'images'
LABEL_DIR = 'labels'

# IMAGE_FILES = os.listdir(join(DATA_ROOT, IMAGE_DIR))
# IMAGE_FILES = [image_file for image_file in IMAGE_FILES if '_aug' not in image_file]
# FILE_IDS = [image_file.split('.')[0] for image_file in IMAGE_FILES]
# LABEL_FILES = [file_id + '.png' for file_id in FILE_IDS]
LABEL_FILES = os.listdir(join(DATA_ROOT, LABEL_DIR))
LABEL_FILES = [label_file for label_file in  LABEL_FILES if '_aug' not in label_file]
FILE_IDS = [label_file.split('.')[0] for label_file in LABEL_FILES]
NUM_FILES = len(LABEL_FILES)


for i, (label_file, file_id) in tqdm(enumerate(zip(LABEL_FILES, FILE_IDS))):
    label = cv2.imread(join(DATA_ROOT, LABEL_DIR, label_file))

    # cv2.imwrite(join(DATA_ROOT, IMAGE_DIR, file_id + '_flip.jpg'), image[::-1,:,:])
    cv2.imwrite(join(DATA_ROOT, LABEL_DIR, file_id + '_flip.png'), label[::-1,:,:])