from PIL import Image
import numpy as np

def FloydSteinberg(image, q):
    """
    Floyd-Steinberg function to dither the image into a palette with q colours per channel.

    """

    # convert the image data type to float
    arr = np.array(image, dtype=float) / 255

    for ir in range(image.shape[0]):
        for ic in range(image.shape[1]):
            # NB need to copy here for RGB arrays otherwise err will be (0,0,0)!
            old_val = arr[ir, ic].copy()
            #print(old_val)
            new_val = find_quantized_value(old_val, q)
            # assign the new value
            arr[ir, ic] = new_val
            # calculate the error between old and new values
            err = old_val - new_val
            # ignore the border pixels
            if ic < image.shape[1] - 1:
                arr[ir, ic+1] += err * 7/16
            if ir < image.shape[0] - 1:
                if ic > 0:
                    arr[ir+1, ic-1] += err * 3/16
                arr[ir+1, ic] += err * 5/16
                if ic < image.shape[1] - 1:
                    arr[ir+1, ic+1] += err / 16
    # rescale the image
    carr = np.array(arr/np.max(arr, axis=(0,1)) * 255, dtype=np.uint8)
    # return in Image form to save it later
    return Image.fromarray(carr)

def find_quantized_value(old_val, q):
    """
    Get the "closest" colour to old_val in the range [0,1] per channel divided
    into q values.

    """
    return np.round(old_val * (q-1)) / (q-1)