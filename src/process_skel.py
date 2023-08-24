"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller and Sergi Pujades
See LICENCE.txt for licensing and contact information.
"""

import copy
import numpy as np
import numpy as np
import matplotlib.pyplot as plt


def thresh_percent(im, th_perc):
    im = copy.copy(im)
    th_count = (1 - th_perc) * im.shape[0] * im.shape[1]

    count = np.bincount(im.flatten())
    # ignore pixels to zero
    acc = [count[0]]

    for i, c in enumerate(count[1:]):
        acc.append(acc[i]+c)

    bin_mask = (np.array(acc) > th_count)
    th = np.argmax(bin_mask)
    # print(th_count, th)

    im[im<th] = 0
    im[im>=th] = 255

    return im


def preprocess_dicom_skel(im_dk, im_ds_mask, pad, gender, debug=False):
    assert (im_dk.shape[0] == im_dk.shape[1])
    # Set background to 0
    im = copy.copy(im_dk)
    bg_val = np.bincount(im[:, pad[0]:-pad[1]].flatten()).argmax()

    skin_mask = im_ds_mask
    assert skin_mask.max() <= 1
    bg_mask = 1-skin_mask

    mask = np.logical_and(im == bg_val, bg_mask==1)
    im[mask] = 0

    # Threshold
    if gender =='female':
        t = 0.16
    else:
        t = 0.20
    im[:, pad[0]:-pad[1]] = thresh_percent(im[:, pad[0]:-pad[1]], th_perc=t)
    im = im / 255.

    if debug:
        plt.imshow(im, cmap=plt.cm.gray)
        plt.title("After tresh")
        plt.show()

    return im