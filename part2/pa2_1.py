"""

READ IMAGES AND CALL THE FUNCTION

"""

from matplotlib import pyplot as plt
import cv2
import sys
import os
from pa2_2 import colorTransfer

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)
print("Root directory:", ROOT_DIR)
IMAGE_DIR_PATH = os.path.join(ROOT_DIR, "colortransfer")


def read_images(src, tgt):
    src_img = cv2.imread(os.path.join(IMAGE_DIR_PATH, src))
    tgt_img = cv2.imread(os.path.join(IMAGE_DIR_PATH, tgt))
    """ fig = plt.figure(figsize=(8,8))
    fig.add_subplot(1,2,1)
    plt.imshow(src_img)
    fig.add_subplot(1,2,2)
    plt.imshow(tgt_img) """
    #plt.show()
    return src_img, tgt_img

def transfer(src, tgt):
    src_img, tgt_img = read_images(src, tgt)
    result = colorTransfer(src_img, tgt_img)
    plt.imshow(result)
    plt.show()


transfer("autumn.jpg", "ocean_day.jpg")



