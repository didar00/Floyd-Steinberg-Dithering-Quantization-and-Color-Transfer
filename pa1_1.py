from PIL import Image
import cv2
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = os.path.abspath("./")
sys.path.append(ROOT_DIR)
print(ROOT_DIR)

"""

READ AN IMAGE

"""

fig = plt.figure(figsize=(8, 8))

IMAGE_DIR_PATH = os.path.join(ROOT_DIR, "images", "dithering", "1.png")
image = cv2.imread(IMAGE_DIR_PATH)
# print(image)
fig.add_subplot(2, 2, 1)
plt.imshow(image)


"""

QUANTIZE THE IMAGE

"""

from matplotlib import pyplot as plt
ratio=128  # Set quantization ratio
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        for k in range(image.shape[2]):
            image[i][j][k]=int(image[i][j][k]/ratio)*ratio
print("Quantized image:")
fig.add_subplot(2, 2, 2)
plt.imshow(image)

"""

APPLY DITHERING

"""


def get_new_val(old_val, nc):
    """
    Get the "closest" colour to old_val in the range [0,1] per channel divided
    into nc values.

    """

    return np.round(old_val * (nc - 1)) / (nc - 1)

# For RGB images, the following might give better colour-matching.
#p = np.linspace(0, 1, nc)
#p = np.array(list(product(p,p,p)))
#def get_new_val(old_val):
#    idx = np.argmin(np.sum((old_val[None,:] - p)**2, axis=1))
#    return p[idx]

def fs_dither(img, nc):
    """
    Floyd-Steinberg dither the image img into a palette with nc colours per
    channel.

    """

    arr = np.array(img, dtype=float) / 255

    for ir in range(img.shape[0]):
        for ic in range(img.shape[1]):
            # NB need to copy here for RGB arrays otherwise err will be (0,0,0)!
            old_val = arr[ir, ic].copy()
            new_val = get_new_val(old_val, nc)
            arr[ir, ic] = new_val
            err = old_val - new_val
            # In this simple example, we will just ignore the border pixels.
            if ic < img.shape[1] - 1:
                arr[ir, ic+1] += err * 7/16
            if ir < img.shape[0] - 1:
                if ic > 0:
                    arr[ir+1, ic-1] += err * 3/16
                arr[ir+1, ic] += err * 5/16
                if ic < img.shape[1] - 1:
                    arr[ir+1, ic+1] += err / 16

    carr = np.array(arr/np.max(arr, axis=(0,1)) * 255, dtype=np.uint8)
    return Image.fromarray(carr)


def palette_reduce(img, nc):
    """Simple palette reduction without dithering."""
    arr = np.array(img, dtype=float) / 255
    arr = get_new_val(arr, nc)

    carr = np.array(arr/np.max(arr) * 255, dtype=np.uint8)
    return Image.fromarray(carr)


for nc in (2,3,4,5,6):
    print('nc =', nc)
    dim = fs_dither(image, nc)
    dim.save('dimg-{}.jpg'.format(nc))
    rim = palette_reduce(image, nc)
    rim.save('rimg-{}.jpg'.format(nc))
plt.show()