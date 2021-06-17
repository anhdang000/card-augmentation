import sys
import os
from os.path import join
import json
import cv2
import numpy as np
from tqdm import tqdm
from scipy import ndimage
from PIL import Image
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

for i, (image_file, label_file, file_id) in tqdm(enumerate(zip(IMAGE_FILES, LABEL_FILES, FILE_IDS)), total=NUM_FILES):
    image = Image.open(join(DATA_ROOT, IMAGE_DIR, image_file))
    label = Image.open(join(DATA_ROOT, LABEL_DIR, label_file))
    
    w, h = image.size

    # Pick up some random images
    indices = random.sample(range(NUM_ALIGNED_FILES), random.randrange(1, 3))
    candidates = {
        'images': [ALIGNED_IMAGE_FILES[idx] for idx in indices],
        'labels': [ALIGNED_LABEL_FILES[idx] for idx in indices]
        }
    
    # Create intial results
    result = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    result.paste(image, (0, 0))
    result_mask = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    result_mask.paste(label, (0, 0))

    for i in range(len(indices)):
        minor_image_file = candidates['images'][i]
        minor_label_file = candidates['labels'][i]
        minor_image = cv2.imread(join(DATA_ROOT, IMAGE_DIR, minor_image_file))[:, :, ::-1]
        minor_label = cv2.imread(join(DATA_ROOT, LABEL_DIR, minor_label_file))[:, :, ::-1]

        # Rotate with random angle
        angle = random.randrange(360)
        minor_image = ndimage.interpolation.rotate(minor_image, angle)
        minor_label = ndimage.interpolation.rotate(minor_label, angle)

        obj_h, obj_w = minor_label.shape[:2]

        minor_image = Image.fromarray(minor_image).convert('RGBA')
        minor_label = Image.fromarray(minor_label).convert('RGBA')
        
        # Convert to transparent masks
        minor_image_data = minor_image.getdata()
        minor_label_data = minor_label.getdata()
        new_img_data = []
        new_label_data = []
        for image_item, label_item in zip(minor_image_data, minor_label_data):
            if label_item[:3] == (0, 0, 0):
                new_img_data.append((0, 0, 0, 0))
                new_label_data.append((0, 0, 0, 0))
            else:
                new_img_data.append(tuple(list(image_item)))
                new_label_data.append(tuple(list(label_item)))
        minor_image.putdata(new_img_data)
        minor_label.putdata(new_label_data)

        # Random location to merge
        min_y = random.randrange(-h//6, h - h//3)
        max_y = min_y + obj_h
        min_x = random.randrange(-w//6, w - w//3)
        max_x = min_x + obj_w

        # Apply RGBA masks
        result.paste(minor_image, (min_x, min_y), mask=minor_image)
        result_mask.paste(minor_label, (min_x, min_y), mask=minor_label)

    result.load()
    result_rgb = Image.new("RGB", result.size, (255, 255, 255))
    result_rgb.paste(result, mask=result.split()[3])
    result_rgb.save(join(OUTPUT_IMAGE_DIR, file_id + '_merge.jpg'), "JPEG", quality=80)

    result_mask_p = result_mask.convert('P')
    result_mask_p.save(join(OUTPUT_LABEL_DIR, file_id + '_merge.png'), "PNG")

