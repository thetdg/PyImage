import cv2
import skimage as sk
import numpy as np
import matplotlib.pyplot as plt


def enhance(img_filename):
    img_orig = cv2.imread(img_filename, 0)
    min_val = np.amin(img_orig)
    max_val = np.amax(img_orig)

    img_new = np.zeros_like(img_orig, dtype='float')
    img_new[:, :] = (img_orig[:, :] - min_val)
    img_new = img_new * 255 / (max_val - min_val)
    img_new = np.array(img_new, dtype='uint8')
    return img_orig, img_new


a, b = enhance('./images/low_2.jpg')
plt.figure('Original')
sk.io.imshow(a)
plt.figure('Enhanced')
sk.io.imshow(b)
plt.show()


