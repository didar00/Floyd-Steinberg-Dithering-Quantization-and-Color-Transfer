from plantcv import plantcv as pcv

from matplotlib import pyplot as plt
import cv2
import sys
import os
import numpy as np


ROOT_DIR = os.path.abspath("./")
#sys.path.append(ROOT_DIR)
print("Root directory:", ROOT_DIR)
IMAGE_DIR_PATH = os.path.join(ROOT_DIR, "colortransfer")

a = os.path.join(IMAGE_DIR_PATH, "scotland_house.jpg")
src_img = cv2.imread(a, cv2.COLOR_BGR2RGB)


# Set global debug behavior to None (default), "print" (to file), 
# or "plot" (Jupyter Notebooks or X11)
pcv.params.debug = "print"

# image converted from RGB to LAB, channels are then split. 
# Lightness ('l') channel is output
l_channel = pcv.rgb2gray_lab(rgb_img=src_img, channel='l')
a_channel = pcv.rgb2gray_lab(rgb_img=src_img, channel='a')
b_channel = pcv.rgb2gray_lab(rgb_img=src_img, channel='b')

fig = plt.figure(figsize=(6,6))
fig.add_subplot(1, 3, 1)
plt.imshow(l_channel)

fig.add_subplot(1, 3, 2)
plt.imshow(a_channel)
fig.add_subplot(1, 3, 3)
plt.imshow(b_channel)
plt.show()

