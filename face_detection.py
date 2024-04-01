from skimage import data
from skimage.feature import Cascade
from skimage.io import imread
from skimage.transform import rescale

import matplotlib.pyplot as plt
from matplotlib import patches

# Load the trained file from the module root.
trained_file = data.lbp_frontal_face_cascade_filename()
detector = Cascade(trained_file)


def detect(img_path, max_dim=512, cascade_min=50, cascade_max=250):
    img_orig = imread(img_path)
    height, width, depth = img_orig.shape
    scaling_factor = float(max_dim) / max(height, width)
    img = rescale(img_orig, scaling_factor, channel_axis=2)

    detected = detector.detect_multi_scale(img=img,
                                           scale_factor=1.2,
                                           step_ratio=1,
                                           min_size=(cascade_min, cascade_min),
                                           max_size=(cascade_max, cascade_max))

    plt.imshow(img)
    img_desc = plt.gca()
    plt.set_cmap('gray')

    for patch in detected:

        img_desc.add_patch(
            patches.Rectangle(
                (patch['c'], patch['r']),
                patch['width'],
                patch['height'],
                fill=False,
                color='r',
                linewidth=2
            )
        )

    plt.show()
