
from matplotlib import pyplot as plt
import cv2
import sys
import os
import numpy as np

""" img = cv2.imread("colortransfer/ocean_day.jpg")		# this is read in BGR format
rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)		# this converts it into RGB

plt.imshow(rgb_img)
plt.show()
 """


ROOT_DIR = os.path.abspath("./")
#sys.path.append(ROOT_DIR)
print("Root directory:", ROOT_DIR)
IMAGE_DIR_PATH = os.path.join(ROOT_DIR, "colortransfer")

a = os.path.join(IMAGE_DIR_PATH, "ocean_day.jpg")
src_img = cv2.imread(a, cv2.COLOR_BGR2RGB)
""" plt.imshow(src_img)
plt.show()
plt.clf() """
""" img_rgb = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
print(img_rgb.shape)
print(img_rgb)

img_rgb = img_rgb.astype(np.float32)
print(img_rgb)

img_rgb = img_rgb.astype(np.uint8)
print(img_rgb) """

f_img = np.array([[1.3, 1.1], [1.7, 1.2]])
f_img = f_img.astype(np.uint8)
print(f_img)
plt.imshow(f_img)
plt.show()
plt.clf()



b = np.array([[ 9.73171465,  9.40304144, 15.00434143],
  [18.31422442, 19.5245560, 32.84124588],
  [10.74063142, 10.9931502, 12.18037119],[59.51198265,77.39051798,71.17570747],
  [61.28734795,78.74617371,75.2951643],[66.11103577,82.67368846,89.69272768]])

b = b.astype(np.float32)
print(b)
plt.imshow(b)
plt.show()

""" plt.imshow(img_rgb)
plt.show()
print(src_img) """

def _min_max_scale(arr, new_range=(0, 255)):
	# get array's current min and max
	mn = arr.min()
	mx = arr.max()

	# check if scaling needs to be done to be in new_range
	if (mn < new_range[0] or mx > new_range[1]) and mn != mx:
		# perform min-max scaling
		scaled = (new_range[1] - new_range[0]) * (arr - mn) / (mx - mn) + new_range[0]
	else:
		# return array if already in range
		scaled = arr

	return scaled

def _scale_array(arr, clip=True):
	if clip:
		scaled = np.clip(arr, 0, 255)
	else:
		scale_range = (max([arr.min(), 0]), min([arr.max(), 255]))
		scaled = _min_max_scale(arr, new_range=scale_range)

	return scaled

""" img = np.zeros((20,20,3))
img = img.astype(np.float32)
print(img)
plt.imshow(img)
plt.show()

img = _min_max_scale(img, new_range=(1,255))
print(img)
plt.imshow(img)
plt.show()

img = np.where(img < 1, 1, img)
print(img)
plt.imshow(img)
plt.show()
 """
