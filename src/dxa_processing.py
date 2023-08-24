"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller and Sergi Pujades
See LICENCE.txt for licensing and contact information.
"""

import numpy as np
from src.process_skel import preprocess_dicom_skel
from src.process_skin import preprocess_dicom_skin


def pad_to_square(im):
    """
    Pas image on the sides to make it square
    :param im: Image to pad
    :return: padded image, (nb of columns added on left, right)
    """
    assert(im.shape[0] > im.shape[1])
    # nb px to add on the width to get a square
    dw = im.shape[0] - im.shape[1]
    if dw % 2 == 0:
        w_pad_l = dw // 2
        w_pad_r = dw // 2
    else:
        w_pad_l = dw // 2
        w_pad_r = dw // 2 + 1
    im = np.pad(im, ((0, 0), (w_pad_l, w_pad_r)))
    return im, (w_pad_l, w_pad_r)


def compute_skin_mask(data):

    # pad scans and save padding
    im_ds, padding_ds = pad_to_square(data['skin'])
    im_dk, padding_dk = pad_to_square(data['skel'])
    assert padding_dk == padding_ds

    im_ds_pp = preprocess_dicom_skin(im_ds, padding_ds, debug=False)*255

    return im_ds_pp


def compute_skel_mask(data, im_ds_pp):

    # pad scans and save padding
    im_ds, padding_ds = pad_to_square(data['skin'])
    im_dk, padding_dk = pad_to_square(data['skel'])
    assert padding_dk == padding_ds
    
    gender = data['gender']
    im_pp_skel = preprocess_dicom_skel(im_dk, im_ds_pp/255, padding_ds, gender, debug=False)*255
    
    return im_pp_skel
