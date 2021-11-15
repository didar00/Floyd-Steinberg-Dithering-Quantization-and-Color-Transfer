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


fig = plt.figure(figsize=(12,12))
fig.add_subplot(1, 5, 1)
plt.imshow(image)
plt.title("Image")

print("Quantization")

i = 2
for q in (2,4,6,8):
    print('q =', q)
    quantized_img= quantize(image, q)
    title = "Quantization (q=" + str(q) +  ")"
    fig.add_subplot(1, 5, i)
    plt.title(title)
    plt.imshow(quantized_img)
    i += 1

plt.show()
plt.clf()


"""

APPLY FLOYD-STEINBERG DITHERING

"""


fig = plt.figure(figsize=(12,12))
fig.add_subplot(2, 4, 1)
plt.imshow(image)
plt.title("Image")

print()
print("Dithering")

i = 5
for q in (2,4,6,8):
    print('q =', q)
    dithered_img= FloydSteinberg(image, q)
    title = "Dithering (q=" + str(q) +  ")"
    fig.add_subplot(2, 4, i)
    plt.title(title)
    plt.imshow(dithered_img)
    i += 1
    

plt.show()
plt.clf()



