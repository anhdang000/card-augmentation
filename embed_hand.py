import sys
import os
from PIL import Image
import numpy as np


filename = 'hand.png'
ironman = Image.open(filename, 'r')
filename1 = 'test.jpg'
bg = Image.open(filename1, 'r')
w, h = bg.size
text_img = Image.new('RGBA', (w,h), (0, 0, 0, 0))
text_img.paste(bg, (0,0))
text_img.paste(ironman, (0,0), mask=ironman)
text_img.save("result.png", format="png")