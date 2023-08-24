"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller and Sergi Pujades
See LICENCE.txt for licensing and contact information.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_hips_x_range(im, dw=0.25, debug=False):
    h = im.shape[0]
    # We look for an image row of the torso
    edges = cv2.Canny(im, 100, 200)
    # by default we take the line at 37% of the image height
    hcut_index = int(0.37 * h)
    slice = im[hcut_index, :]
    x0 = np.argmin(slice)
    x1 = slice.shape[0] - np.argmin(np.flipud(slice))
    x_min = x0 + int((x1-x0)*dw)
    x_max = x1 - int((x1-x0)*dw)

    if debug:
        # plt.plot(slice)
        im[:, x0]   = 100
        im[:, x1-1] = 100
        im[:, x_min] = 100
        im[:, x_max] = 100
        im[im.shape[0]//2,:] = 100
        im[hcut_index, :] = 100
        print(f"hcut: {hcut_index}, xmin: {x_min}, xmax: {x_max}")
        plt.imshow(im, cmap=plt.cm.gray)
        plt.show()
    return (x_min, x_max)


def chest_contours(cnts, im, pad):
    chess_cnts = []
    (x_min, x_max) = get_hips_x_range(im[:, pad[0]:-pad[1]])
    x_min += pad[0]
    x_max += pad[0]
    y_max = im.shape[0]*0.55

    for cnt in cnts:
        in_w = np.all(cnt[:,0,0] > x_min) and np.all(cnt[:,0,0] < x_max)
        in_h = np.all(cnt[:,0,1] < y_max)
        if in_w and in_h:
            chess_cnts.append(cnt)

    return chess_cnts


def zero_background(im_ds, pad, debug=False):
    """ Set the background of the dxa image to zero"""
    
    assert(pad[0]!=0 and pad[1]!=0)
    # Get the value that appears the most in the img
    # and check if it is low or high
    im = np.zeros_like(im_ds, dtype=float)
    bg_val = np.bincount(im_ds[:, pad[0]:-pad[1]].flatten()).argmax()

    if bg_val > 255/2:
        #white bg
        th = bg_val-2

        #no pad region only
        im[:, pad[0]:-pad[1]][im_ds[:, pad[0]:-pad[1]] < th] = 255
        im[:, pad[0]:-pad[1]][im_ds[:, pad[0]:-pad[1]] >= th] = 0
    else:
        # black bg
        th = bg_val
        
        im[:, pad[0]:-pad[1]][im_ds[:, pad[0]:-pad[1]] > th] = 255
        im[:, pad[0]:-pad[1]][im_ds[:, pad[0]:-pad[1]] <= th] = 0


    if debug:
        print(bg_val)
        plt.imshow(im_ds[:, pad[0]:-1-pad[1]])
        plt.title('unpadded')
        plt.show()
        plt.imshow(im)
        plt.title('zero bg')
        plt.show()
        
    return im


def preprocess_dicom_skin(im_ds, padding, debug=False):
    "Create a mask from a DXA body image"
    
    # Some dxa have black backgrounds, other have white bg
    assert(im_ds.shape[0]==im_ds.shape[1])
    im = zero_background(im_ds, padding)

    if debug:
        plt.imshow(im, cmap=plt.cm.gray)
        plt.title("After tresh")
        plt.show()

    # Filter out the big contour. The biggest contour can be either the image frame or the image frame + silhouette
    # so we can not identify the body silhouette to just fill it
    im = im.astype(np.uint8)
    contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
    contours = chest_contours(contours, im, padding)

    if debug:
        for j, cnt in enumerate(contours):
            im_t = np.dstack([im]*3)

            im_t = cv2.drawContours(im_t, contours, j, (0, 255, 0), 3)
            plt.imshow(im_t, cmap=plt.cm.gray)
            plt.title("Contour to fill ")
            plt.show()

    # -1 argument fills the inside of the contour
    im = cv2.drawContours(im, contours, -1, (255,255,255), -1)

    #im in [0,1], black bg
    im = im/255

    if debug:
        plt.imshow(im, cmap=plt.cm.gray)
        plt.title('final image')
        plt.show()

    return im