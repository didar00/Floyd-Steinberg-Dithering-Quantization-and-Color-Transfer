from PIL import Image
import cv2
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from pa1_2 import FloydSteinberg

"""

GET THE ABSOLUTE PATH OF THE PROJECT

"""

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)
print("Root directory:", ROOT_DIR)



"""

READ AN IMAGE

"""



#IMAGE_DIR_PATH = os.path.join(ROOT_DIR, "dithering", "2.png")
IMAGE_DIR_PATH = os.path.join(ROOT_DIR, "dithering", "1.png")
image = cv2.imread(IMAGE_DIR_PATH)
print("Image size ", image.shape)


"""

QUANTIZE THE IMAGE

"""

from matplotlib import pyplot as plt

def quantize(image, q):
    img = np.copy(image)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                img[i][j][k] = round(image[i][j][k]/q)*q

    return img






"""

APPLY DITHERING

"""


for q in (2,4,6,8):
    print('q =', q)
    fig = plt.figure(figsize=(12,12))
    quantized = quantize(image, q)
    dithered_img_1= FloydSteinberg(quantized, 2)
    dithered_img_2= FloydSteinberg(quantized, 4)
    dithered_img_3= FloydSteinberg(quantized, 6)
    dithered_img_4= FloydSteinberg(quantized, 8)

    fig.add_subplot(2, 4, 2)
    plt.imshow(image)
    plt.title("Image")
    fig.add_subplot(2, 4, 3)
    plt.imshow(quantized)
    title = "Quantized image (q=" + str(q) +  ")"
    plt.title(title)
    fig.add_subplot(2, 4, 5)
    plt.imshow(dithered_img_1)
    plt.title("Dithered image (q=2)")
    fig.add_subplot(2, 4, 6)
    plt.imshow(dithered_img_2)
    plt.title("Dithered image (q=4)")
    fig.add_subplot(2, 4, 7)
    plt.imshow(dithered_img_3)
    plt.title("Dithered image (q=6)")
    fig.add_subplot(2, 4, 8)
    plt.title("Dithered image (q=8)")
    plt.imshow(dithered_img_4)
    


    plt.show()
    plt.clf()



