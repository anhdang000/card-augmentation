import sys
import os
from os.path import join
import cv2
from PIL import Image, ImageEnhance
import numpy as np
from tqdm import tqdm
import random

# filename = 'hand.png'
# ironman = Image.open(filename, 'r')
# filename1 = 'test.jpg'
# bg = Image.open(filename1, 'r')
# w, h = bg.size
# text_img = Image.new('RGBA', (w,h), (0, 0, 0, 0))
# text_img.paste(bg, (0,0))
# text_img.paste(ironman, (0,0), mask=ironman)
# text_img.save("result.png", format="png")


DATA_ROOT = '../card_dataset'
IMAGE_DIR = 'images'
LABEL_DIR = 'labels'

IMAGE_FILES = os.listdir(join(DATA_ROOT, IMAGE_DIR))
IMAGE_FILES.sort()
FILE_IDS = [image_file.split('.')[0] for image_file in IMAGE_FILES]
LABEL_FILES = [file_id + '.png' for file_id in FILE_IDS]

NUM_FILES = len(IMAGE_FILES)

OUTPUT_IMAGE_DIR = '../card_dataset_aug/images'
OUTPUT_LABEL_DIR = '../card_dataset_aug/labels'

print(f'Number of files: {NUM_FILES}')

hand = Image.open('hand.png', 'r')
hand_mask = Image.open('hand_mask.png', 'r')
orig_hand_w, orig_hand_h = hand.size

for i, (image_file, label_file, file_id) in tqdm(
    enumerate(zip(IMAGE_FILES, LABEL_FILES, FILE_IDS)), 
    total=NUM_FILES):

    # Read image and label source
    image = Image.open(join(DATA_ROOT, IMAGE_DIR, image_file), 'r')
    label = Image.open(join(DATA_ROOT, LABEL_DIR, label_file), 'r')
    # print(image.mode, label.mode)
    assert image.size == label.size
    w, h = image.size

    # Find card's location
    img_array = cv2.imread(join(DATA_ROOT, LABEL_DIR, label_file), cv2.IMREAD_GRAYSCALE)
    indices = np.where(img_array > 0)

    min_y = indices[0].min()
    max_y = indices[0].max()
    min_x = indices[1].min()
    max_x = indices[1].max()

    obj_h = max_y - min_y
    obj_w = max_x - min_x

    # Resize hand template
    scale = int(obj_w/orig_hand_w)
    new_size = (orig_hand_w*scale, orig_hand_h*scale)
    try:
        hand_aug = hand.resize(new_size)
        hand_mask_aug = hand_mask.resize(new_size)
    except:
        print(f'Ignored {file_id}!')
        continue

    # Random change hand template brightness
    factor = random.uniform(0.2, 3)
    enhancer = ImageEnhance.Brightness(hand_aug)
    hand_aug = enhancer.enhance(factor)

    merged_image = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    merged_image.paste(image, (0, 0))
    merged_image.paste(hand_aug, (min_x-3*new_size[0]//6, max_y-obj_h//2), mask=hand_aug)

    merged_label = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    merged_label.paste(label, (0, 0))
    merged_label.paste(hand_mask_aug, (min_x-3*new_size[0]//6, max_y-obj_h//2), mask=hand_mask_aug)

    # merged_image.save('test.png', 'PNG')
    # merged_label.save('mask.png', 'PNG')
    # break
    
    # Convert to RGB
    merged_image.load()
    merged_image_rgb = Image.new("RGB", merged_image.size, (255, 255, 255))
    merged_image_rgb.paste(merged_image, mask=merged_image.split()[3])
    merged_image_rgb.save(join(OUTPUT_IMAGE_DIR, file_id + '_thumb.jpg'), 'JPEG', quality=80)

    merged_label_p = merged_label.convert('P')
    merged_label_p.save(join(OUTPUT_LABEL_DIR, file_id + '_thumb.png'), "PNG")
