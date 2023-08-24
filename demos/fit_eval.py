"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller and Sergi Pujades
See LICENCE.txt for licensing and contact information.
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.spatial.distance import directed_hausdorff
    
from src.loaders.return_loader import ReturnData
from src.loaders.dxa_loader import DxaLoader
from src.dxa_processing import compute_skel_mask, compute_skin_mask, pad_to_square
from src.pyrender_renderer import render_dxa_mesh

debug = False

def stack_colored(s1, s2, colors=None):
    """
    :param s1:
    :param s2: mask (GT) orange
    :param colors:
    :return:
    """

    if colors is not None:
        assert len(colors) == 3
        s1_col, s1_col, intersect_col = colors
    else:
        s1_col = [0.5,0.9,0.3] #light green
        s2_col = [0.9,0.5,0] #orange
        intersect_col = [1,1,1]

    if len(s1.shape) == 3:
        s1 = np.sum(s1, axis=2)
    if len(s2.shape) == 3:
        s2 = np.sum(s2, axis=2)

    im = np.zeros(s1.shape + (3,))
    for ci in range(3):

        im[s1>0, ci] = s1_col[ci]
        im[s2>0, ci] = s2_col[ci]

        i_mask = np.logical_and(s1>0, s2>0)
        im[i_mask, ci] = intersect_col[ci]
    return im


def iou(s, s_gt, over_gt):
    """

    :param s: silhouette
    :param s_gt: silhouette GT
    :param over_gt: It true, divide by np.count_nonzero(s_gt) instread of the union
    :return:
    """
    
    # normalize the image
    s = s/s.max()
    s_gt = s_gt/s_gt.max()
    
    assert np.all(np.logical_or(s == 0, s == 1))
    assert np.all(np.logical_or(s_gt == 0, s_gt == 1))

    intersection = np.count_nonzero(np.logical_and(s, s_gt))
    union = np.count_nonzero(np.logical_or(s, s_gt))

    if over_gt:
        return intersection / np.count_nonzero(s_gt)
    else:
        return intersection / union


def hausdorf_dist(s, s_gt):
    
    # normalize the image
    s = s/s.max()
    s_gt = s_gt/s_gt.max()

    assert np.all(np.logical_or(s == 0, s == 1))
    assert np.all(np.logical_or(s_gt == 0, s_gt == 1))

    s = s.copy()
    s_gt = s_gt.copy()
    l_s_gt = np.array(np.where(s_gt)).T
    l_s = np.array(np.where(s)).T
    d_gt, index_gt, index_comp = directed_hausdorff(l_s_gt, l_s)

    d_comp, _, _ = directed_hausdorff(l_s, l_s_gt)

    if debug == True:
        import matplotlib.pyplot as plt
        print(d_gt, index_gt, index_comp)
        im = np.dstack([s, s_gt, np.zeros_like(s)])
        # Display the pixels responsible for the Hausdorff distance
        gt_index_d = l_s_gt[index_gt]
        index_d = l_s[index_comp]
        print(gt_index_d)
        print(index_d)
        im[gt_index_d[0], gt_index_d[1], ...] = [0, 0, 1]
        im[index_d[0], index_d[1], ...] = [0, 0, 1]
        plt.imshow(im)
        plt.show()

    return d_gt, d_comp


def connected_comp(gray):
    """Remove noise on a binary image by keeping only the biggest connected components"""

    gray = gray.astype(np.uint8)
    res = cv2.connectedComponentsWithStats(gray)
    (numLabels, labels, stats, centroids) = res

    # Init mask
    mask = np.zeros(gray.shape, dtype="uint8")

    # Skip component 0 - background
    for i in range(1, numLabels):

        # Extract the connected component statistics and centroid for the current label
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        area = stats[i, cv2.CC_STAT_AREA]
        #print(f'Area {area}')

        # Ensure the width, height, and area are all neither too small nor too big
        keepWidth = w > 5 # and w < 50
        keepHeight = h > 5 # and h < 65
        keepArea = area > 50 #and area < 1500

        # ensure the connected component we are examining passes all three tests
        if all((keepWidth, keepHeight, keepArea)):
            # Construct a mask for the current connected component and then take the bitwise OR with the mask
            #print("[INFO] keeping connected component '{}'".format(i))
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)

    #cv2.imshow("Image", gray.astype("uint8") * 255)
    #cv2.imshow("Characters", mask)
    #cv2.waitKey(0)

    return mask / 255


def eval_fit(dxa_data, return_data, part='body'):
    """Evaluate how well the mesh from the return_data overlap with the silhouette of the dxa image in dxa_data."""
    
    # Load images and meshes
    if part == 'body':
        dxa_img = dxa_data['skin'].astype(np.uint8)
        mesh_path = return_data['body_mesh_path']
    elif part == 'skel':
        dxa_img = dxa_data['skel'].astype(np.uint8)
        mesh_path = return_data['skeleton_mesh_path']
    else:
        raise ValueError(f'Unknown part {part}, should be body or skel')
    
    assert os.path.exists(mesh_path), f'Mesh path {mesh_path} does not exist'
    
    # Compute masks from the DXA
    dxa_skin_silhouette = compute_skin_mask(ukb_data)
    
    if part == 'body':
        dxa_silhouette = dxa_skin_silhouette
    elif part=='skel':
        dxa_silhouette = compute_skel_mask(ukb_data, dxa_skin_silhouette)
        dxa_silhouette = connected_comp(dxa_silhouette) # Remove noise from the segmented skeleton mask

    # Generate a mask from the mesh
    mesh_render = render_dxa_mesh(mesh_path, dxa_data, transparent=False, silhouette=False)
    mesh_silhouette = render_dxa_mesh(mesh_path, dxa_data, transparent=False, silhouette=True)
    
    dxa_image_padded, _ = pad_to_square(dxa_img)
    
    # Compute overlap between the GT dxa image and the mesh's rendered silhouette
    ovverlap = stack_colored(mesh_silhouette, dxa_silhouette)
    
    # Compute the overlap metrics
    hausedorf_gt, hausedorf_comp = hausdorf_dist(mesh_silhouette, dxa_silhouette)
    print('\tHausedorf distance (dxa to mesh) :', f'{hausedorf_gt:0.3f} px')
    
    if part == 'skel':
        iou_err = iou(mesh_silhouette, dxa_silhouette, over_gt=True)
        print('\tIntersection over DXA silhouette :', f'{iou_err:0.3f}')
    else:
        iou_err = iou(mesh_silhouette, dxa_silhouette, over_gt=False)
        print('\tIntersection over union :', f'{iou_err:0.3f}')

    
    # Plot both silhouettes and overlap
    fig, axs = plt.subplots(2, 3)
    plt.axis('off')
    ax = axs[0][0]
    ax.imshow(mesh_render)
    ax.set_title('Mesh rendering')
    ax = axs[0][1]
    ax.imshow(mesh_silhouette, cmap='gray')
    ax.set_title('Mesh silhouette')
    
    ax = axs[1][0]
    ax.imshow(dxa_image_padded, cmap='gray')
    ax.set_title('DXA image')
    ax = axs[1][1]
    ax.imshow(dxa_silhouette, cmap='gray')
    ax.set_title('DXA silhouette')
    
    ax = axs[0][2]
    ax.imshow(ovverlap)
    ax.set_title('Overlap')
    
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.show() 

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Load and display Biobank data for a subject')  
    parser.add_argument('--subject_id', type=str, default='1000098', help='ID of the subject to display')  
    parser.add_argument('--gender', type=str, default='female', help='Gender of the subject.')
    args = parser.parse_args()
    
    ukb_dataloader = DxaLoader()
    return_data_loader = ReturnData()
    
    return_data = return_data_loader.load_subject_paths(args.subject_id)
    ukb_data = ukb_dataloader.load_dxa_raw_data(ukb_dataloader.get_patient_folder(args.subject_id), gender=args.gender)
    
    print('Evaluating body mesh fit to DXA')
    eval_fit(ukb_data, return_data, part='body')
    
    print('Evaluating skeleton mesh fit to DXA')
    eval_fit(ukb_data, return_data, part='skel')
    