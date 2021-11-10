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
#sys.path.append(ROOT_DIR)
print("Root directory:", ROOT_DIR)
IMAGE_DIR_PATH = os.path.join(ROOT_DIR, "colortransfer")

""" plt.clf()
print(os.path.join(IMAGE_DIR_PATH, "woods.jpg"))
src_img = cv2.imread(os.path.join(IMAGE_DIR_PATH, "ocean_day.jpg"), cv2.COLOR_BGR2RGB)
plt.imshow(src_img)
plt.show()
 """
def read_images(src, tgt):
    src_img = cv2.imread(os.path.join(IMAGE_DIR_PATH, src))
    src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
    plt.imshow(src_img)
    plt.show()
    tgt_img = cv2.imread(os.path.join(IMAGE_DIR_PATH, tgt))
    tgt_img = cv2.cvtColor(tgt_img, cv2.COLOR_BGR2RGB)
    plt.imshow(tgt_img)
    plt.show()
    """ fig = plt.figure(figsize=(8,8))
    fig.add_subplot(1,2,1)
    plt.imshow(src_img)
    fig.add_subplot(1,2,2)
    plt.imshow(tgt_img) """
    #plt.show()
    src_img = src_img.astype(np.float32)
    #src_img/=255
    tgt_img = tgt_img.astype(np.float32)
    #tgt_img/=255
    return src_img, tgt_img

def transfer(src, tgt):
    src_img, tgt_img = read_images(src, tgt)
    result = colorTransfer(src_img, tgt_img)
    result = result.astype(np.uint8)
    #result /=255
    plt.imshow(result)
    plt.title("result")
    plt.show()
    plt.clf()


#transfer("ocean_day.jpg", "ocean_sunset.jpg")
#transfer("autumn.jpg", "storm.jpg")
transfer("ocean_day.jpg", "fallingwater.jpg")



