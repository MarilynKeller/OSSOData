import random
import numpy as np
from skimage.transform import resize

def color_gradient(N, scale=1, shuffle=False, darken=False, pastel=False):
    """Return a list of N color values forming a gradient"""
    import colorsys
    if darken:
        V = 0.75
    else:
        V = 1

    if pastel:
        S = 0.5
    else:
        S = 1

    HSV_tuples = [((N-x) * 1.0 / (scale*N), S, V) for x in range(N)] # blue - grean - yellow - red gradient
    RGB_list = list(map(lambda x: list(colorsys.hsv_to_rgb(*x)), HSV_tuples))

    if shuffle:
        random.Random(0).shuffle(RGB_list) #Seeding the random this way always produces the same shuffle
    return RGB_list


def segmentation_figure(seg_map_3D, base_image=None, normalize=False, color_list=None, th=0, shuffle_colors = True):
    nb_class = seg_map_3D.shape[0]
    if base_image is None:
        base_image = np.zeros((seg_map_3D.shape[-1], seg_map_3D.shape[-1]))
    else:
        # upsize masks to superimpose with base image
        seg_map_3D = resize(seg_map_3D, (seg_map_3D.shape[0], base_image.shape[0], base_image.shape[1]), order=2)

    if color_list is None:
        color_list = color_gradient(seg_map_3D.shape[0], shuffle=shuffle_colors)
    assert seg_map_3D.shape[0] == len(color_list)
    seg_image = np.dstack([base_image]*3)
    for i in range(nb_class):
        class_mask = (seg_map_3D[i,:,:]>th)
        for c in range(3):
            # seg_image[class_mask,c] += color_list[i][c] * seg_map_3D[i,class_mask]/np.max(seg_map_3D[i,class_mask])
            seg_image[class_mask,c] += color_list[i][c] * seg_map_3D[i,class_mask]
    if normalize:
        seg_image = seg_image/np.max(seg_image)
    return seg_image


def normalize_mask(mask):
    """
    :param mask: numpy array or tensor of any dimension
    :return: same as input win min value 0 and max value 1
    """
    mask -= mask.min()
    mask /= mask.max()
    return mask


def normalize_mask_per_channel(pred):
    """
    :param pred: CxWxH tensor
    :return:  CxWxH tensor , each channel has min value 0 and max value 1
    """
    """"""
    for i in range(pred.shape[0]):
        pred[i,...] = normalize_mask(pred[i,...])
        # pred[:,i,...] -= pred[:,i,...].clone().min()
        # pred[:,i,...] /= pred[:,i,...].clone().max()
    return pred