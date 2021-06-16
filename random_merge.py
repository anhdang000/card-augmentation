import sys
import os
from os.path import join
import json
import cv2
import numpy as np
from tqdm import tqdm
from scipy import ndimage
import imageio
import random


DATA_ROOT = '../card_dataset'
IMAGE_DIR = 'images'
LABEL_DIR = 'labels'

IMAGE_FILES = os.listdir(join(DATA_ROOT, IMAGE_DIR))
IMAGE_FILES.sort()
FILE_IDS = [image_file.split('.')[0] for image_file in IMAGE_FILES]
LABEL_FILES = [file_id + '.png' for file_id in FILE_IDS]

NUM_FILES = len(IMAGE_FILES)

# Aligned files
ALIGNED_IMAGE_FILES = [image_file for image_file in IMAGE_FILES if 'aligned' in image_file]
ALIGNED_LABEL_FILES = [label_file for label_file in LABEL_FILES if 'aligned' in label_file]
NUM_ALIGNED_FILES = len(ALIGNED_IMAGE_FILES)

OUTPUT_IMAGE_DIR = '../card_dataset_aug/images'
OUTPUT_LABEL_DIR = '../card_dataset_aug/labels'

PALETTE = [
        [128, 0, 0], [0, 128, 0], [128, 128, 0], 
        [0, 0, 128], [128, 0, 128], [0, 128, 128]
        ]
print(f'Number of files: {NUM_FILES}')
print(f'Number of aligned images: {NUM_ALIGNED_FILES}')

for i, (image_file, label_file, file_id) in tqdm(enumerate(zip(IMAGE_FILES, LABEL_FILES, FILE_IDS))):
    image = cv2.imread(join(DATA_ROOT, IMAGE_DIR, image_file))
    label = cv2.imread(join(DATA_ROOT, LABEL_DIR, label_file))

    assert image.shape == label.shape, f'Image and lael do not match: {file_id}'

    h, w = image.shape[:2]
    
    # Pick up some random images
    indices = random.sample(range(NUM_ALIGNED_FILES), random.randrange(2, 5))
    candidates = {
        'images': [ALIGNED_IMAGE_FILES[idx] for idx in indices],
        'labels': [ALIGNED_LABEL_FILES[idx] for idx in indices]
        }
    
    for i in range(len(indices)):
        minor_image_file = candidates['images'][i]
        minor_label_file = candidates['labels'][i]
        minor_image = cv2.imread(join(DATA_ROOT, IMAGE_DIR, minor_image_file))
        minor_label = cv2.imread(join(DATA_ROOT, LABEL_DIR, minor_label_file))

        # Rotate with random angle
        angle = random.randrange(360)
        minor_image = ndimage.interpolation.rotate(minor_image, angle)
        minor_label = ndimage.interpolation.rotate(minor_label, angle)
        cv2.imwrite('minor_image.jpg', minor_image)
        cv2.imwrite('minor_label.jpg', minor_label)

        obj_h, obj_w = minor_label.shape[:2]

        # Random location to merge
        min_y = random.randrange(-h//6, h - h//2)
        max_y = min_y + obj_h
        min_x = random.randrange(-w//6, w - w//2)
        max_x = min_x + obj_w
        # 
        min_y_clip = max(0, min_y)
        max_y_clip = min(h, max_y)
        min_x_clip = max(0, min_x)
        max_x_clip = min(w, max_x)

        # Merge masks
        try:
            
            label[min_y_clip: max_y_clip, min_x_clip: max_x_clip, :] = minor_label[
                (min_y_clip-min_y):obj_h + (max_y_clip-max_y), 
                (min_x_clip-min_x):obj_w + (max_x_clip-max_x), :
                ]
            
            image[min_y_clip: max_y_clip, min_x_clip: max_x_clip, :] = minor_image[
                (min_y_clip-min_y):obj_h + (max_y_clip-max_y), 
                (min_x_clip-min_x):obj_w + (max_x_clip-max_x), :
                ]
        except:
            pass

    cv2.imwrite(join(OUTPUT_IMAGE_DIR, file_id + '_aug_1.jpg'), image)
    cv2.imwrite(join(OUTPUT_LABEL_DIR, file_id + '_aug_1.png'), label)


