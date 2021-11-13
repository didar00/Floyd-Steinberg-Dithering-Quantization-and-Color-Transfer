"""

READ IMAGES AND CALL THE FUNCTION

"""

from matplotlib import pyplot as plt
import cv2
import sys
import os
from pa2_2 import colorTransfer
import numpy as np

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)
print("Root directory:", ROOT_DIR)
IMAGE_DIR_PATH = os.path.join(ROOT_DIR, "colortransfer")


def read_images(src, tgt):
    src_img = cv2.imread(os.path.join(IMAGE_DIR_PATH, src))
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)

    tgt_img = cv2.imread(os.path.join(IMAGE_DIR_PATH, tgt))
    tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)

    src_img = src_img.astype(np.float32)
    tgt_img = tgt_img.astype(np.float32)

    return src_img, tgt_img

def transfer(src, tgt):
    src_img, tgt_img = read_images(src, tgt)
    result = colorTransfer(src_img, tgt_img)
    result = result.astype(np.uint8)
    #result /=255
    """ plt.imshow(result)
    plt.title("result")
    plt.show() """
    plt.imshow(result)
    plt.savefig('output_image.jpg')



# call tranfer to perform color transfer
transfer("cat-eyes.jpg", "space.jpg")



