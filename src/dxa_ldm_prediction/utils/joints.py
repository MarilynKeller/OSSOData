import numpy as np
from skimage.transform import resize
from scipy import ndimage
import matplotlib.pyplot as plt
from utils.segmentation import normalize_mask
import cv2

kin_tree=[
    #right leg
    [0,1],
    [1,4],
    [4,7],
    [7,10],

    #left leg
    [0,2],
    [2,5],
    [5,8],
    [8,11],

    #torso
    [0,3],
    [3,6],
    [6,9],
    [9,12],
    [12,15],

    #Left arm
    [9,14],
    [14,17],
    [17,19],
    [19,21],
    [21,23],
    #Right arm
    [9, 13],
    [13,16],
    [16,18],
    [18,20],
    [20,22],

    #additional landmarks
    [15, 24],
    [23, 25],
    [22, 26],
    [11, 27],
    [10, 28],
]

def joint_locations(joint_masks, target_im_size):
    """
    Return joint position as an array  from joints masks
    :param joint_masks: CxWxH np array
    :param target_im_size: scalar giving the size of the image in which you want the 2D location
    :return: Cx2 2D coordinates of the joint in the image
    """
    joint_masks = resize(joint_masks, (joint_masks.shape[0], target_im_size, target_im_size))

    joint_pos_list = []
    for j in range(joint_masks.shape[0]):
        from scipy import ndimage
        joint_pos_list.append(ndimage.measurements.center_of_mass(joint_masks[j, ...]))
    return np.array(joint_pos_list)

def joint_locations_from_pred(joint_masks, target_im_size, thresh = 0.7):
    """
    Return joint position as an array  from joints masks
    :param : DxCxWxH tensor where D is the nb of hourglasses
    :param target_im_size: scalar giving the size of the image in which you want the 2D location
    :return: Cx2 2D coordinates of the joint in the image
    """
    joint_masks = joint_masks.cpu().numpy()
    joint_masks = joint_masks.mean(axis=0)
    joint_masks = resize(joint_masks, (joint_masks.shape[0], target_im_size, target_im_size))

    joint_pos_list = []
    for j in range(joint_masks.shape[0]):
        mask = normalize_mask(joint_masks[j, ...])
        mask[mask < thresh] = 0
        joint_pos_list.append(ndimage.measurements.center_of_mass(joint_masks[j, ...]))
    return np.array(joint_pos_list)


def joints_on_im(im, joints_list, js, joints_mask_size, darken=False, color_gradient_scale=1):
    """
    From a list of 2D possition, print them on the image im
    :param im: WixWi or WixWix3 np array
    :param joints_list:
    :param js: Joint size /2 in px to print on the image
    :param joints_mask_size: 1D size of the image from which the joint coordinates where taken from
    :return: im with 3 color channels, with the joint locations printed in different colors
    """
    im = im / im.max()
    if len(im.shape) == 2:
        im = np.dstack([im] * 3)

    s = im.shape[0] / joints_mask_size
    from utils.draw import color_gradient
    c = color_gradient(len(joints_list), shuffle=False, scale=color_gradient_scale, darken=darken)

    for i, jp in enumerate(joints_list):
        x = int(jp[0] * s)
        # in case im is rectangle, take into account the wodth padding
        m = im.shape[0] / 2 - im.shape[1] / 2
        y = int(jp[1] * s - m)
        # im[x - js:x + js+1, y - js:y + js + 1, :] = c[i]
        import cv2
        im = cv2.circle(im, (y, x), js, tuple(c[i]), cv2.FILLED, 8, 0)

    return im

def show_all_channels(joint_masks):

    im_list = []
    for j in range(joint_masks.shape[0]):
        mask = normalize_mask(joint_masks[j, ...])
        im_list.append(mask)

    im_list.append(joint_masks.mean(axis=0))
    from utils.plot_image_grid import image_grid
    image_grid(im_list)
    return

def draw_kin_tree(im, joints_loc, kin_thickness):
    from utils.draw import color_gradient
    im = im*255
    def l2t(p):
        return (int(p[1]), int(p[0]))

    colors = color_gradient(len(joints_loc), shuffle=True)
    for seg in kin_tree:
        p1 = l2t(joints_loc[seg[0]])
        p2 = l2t(joints_loc[seg[1]])
        c = colors[seg[0]]
        c = tuple ([int(x*255) for x in c])
        im = cv2.line(im, tuple(p1), tuple(p2), c, kin_thickness)
    return im/255.
