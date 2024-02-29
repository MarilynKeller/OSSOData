""" Prediction of landmarks from DXA images
python 
"""

import argparse
import logging
import os
import config as cg
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

from dxa_ldm_prediction.model.posenet import PoseNet
from dxa_ldm_prediction.utils.joints import draw_kin_tree, joint_locations_from_pred, joints_on_im
from dxa_ldm_prediction.utils.segmentation import normalize_mask_per_channel, segmentation_figure
from src.dxa_processing import compute_skel_mask, compute_skin_mask
from src.loaders.dxa_loader import DxaLoader

def predict_img(net,
                img,
                device,
                true_mask=None):
    net.eval()

    img = img.unsqueeze(0)
    img = img.to(device=device)

    with torch.no_grad():
        pred = net(img)

    img = img.squeeze()
    pred_j = joint_locations_from_pred(pred[0, ...], img.shape[0], thresh=cg.joint_pred_th)

    pred_error = None
    if true_mask is not None:
        true_j = jt.joint_locations(true_mask, img.shape[0])
        pred_error = np.linalg.norm(true_j - pred_j, axis=1)


    return pred, pred_error

@classmethod
def preprocess_image(img, im_size):
    #Add a dimension for the channel number, here there is one channel
    img = cv2.resize(img, (im_size, im_size), interpolation=cv2.INTER_LINEAR)
    _, img = cv2.threshold(img, 0.5, 1, cv2.THRESH_BINARY)
    return img


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='', type=str, help="Path to the prediction model checkpoint.")
    parser.add_argument('--subject_id', type=str, default='1000098', help='ID of the subject to process')
    parser.add_argument('--gender', type=str, default='female', help='Gender of the subject. The processing differs slightly')
   
    parser.add_argument('--display', '-d', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


if __name__ == "__main__":
    args = get_args()
    in_files = args.input

    if not args.model:
        args.model = cg.joint_pred_skel_checkpoint

    n_channels = 1
    device = 'cuda'

    net = PoseNet(nstack=cg.nstack,
                  inp_dim=cg.jl_inp_dim,
                  in_channels=cg.in_channels,
                  oup_dim=cg.nb_ldm,
                  bn=False,
                  increase=0).to(device)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    #Test on random image
    errors = []

    ukb_dataloader = DxaLoader()
    ukb_data = ukb_dataloader.load_dxa_raw_data(ukb_dataloader.get_patient_folder(args.subject_id), args.gender)
    
    im_ds = ukb_data['skin']
    im_dk = ukb_data['skel']    

    #Extract mask from dxa
    im_ds_pp = compute_skin_mask(ukb_data)  
    im_dk_pp = compute_skel_mask(ukb_data, im_ds_pp)
    
    # Use the skeleton image  
    img_np = im_dk_pp

    img_np = preprocess_image(img_np, cg.jl_inp_dim)
    img = torch.from_numpy(img_np).type(torch.FloatTensor)
    true_mask = None

    #unsqueeze to add the batch dimension
    img = img.unsqueeze(0)
    # show_data(data)
    # img = preprocess(img)

    pred, pred_error = predict_img(net=net,
                        img=img,
                        device=device)

    # print distances
    if pred_error is not None:
        print(["{:.2f}".format(x) for x in pred_error])
        print(f'Joint dist error \n   mean: {np.mean(pred_error)}  \n    std: {np.std(pred_error)}')

    if args.display:
        plt.figure(figsize=(20,20))
        pred = normalize_mask_per_channel(pred)
        pred_np = pred[0,7,...].cpu().numpy().transpose(1, 2, 0)
        img_np = img[0,...].cpu().numpy()

        # img_np = segmentation_figure(pred_np, base_image=load_dcm(), normalize=True, color_list=None)
        # plt.imshow(img_np)
        # plt.show()

        # im = pp.superpose(load_dcm(), pred_np[...,cg.joints_to_show])
        # plt.imshow(im)
        # # plt.imshow(pred_np[...,cg.joints_to_show])
        # plt.show()

        # for k in range(8):
        #     show_all_channels(pred[0, :,k, ...])
        #     plt.show()

        joints_list = joint_locations_from_pred(pred[0, ...], target_im_size=im_ds.shape[0],
                                                    thresh=cg.joint_pred_th)

        # Compute prediction mask
        pred_mask = pred[0,...].cpu().numpy().mean(axis=0)
        # show_all_channels(pred_mask)
        pred_mask = normalize_mask_per_channel(pred_mask)
        # show_all_channels(pred_mask)
        th = cg.joint_pred_th
        pred_mask[pred_mask<th] = 0

        joint_map_2d = segmentation_figure(pred_mask, img_np, normalize=True)
        joints_loc_image = joints_on_im(joint_map_2d, joints_list, joints_mask_size=im_ds.shape[0], js=2)

        im_skin = joints_on_im(im_ds, joints_list, joints_mask_size=im_ds.shape[0], js=8)
        im_skel = joints_on_im(im_dk, joints_list, joints_mask_size=im_dk.shape[0], js=8)

        title = "Predicted joints on input silhouette / Predicted joint on corresponding DXA scans"
        joints_loc_image = cv2.resize(joints_loc_image, (im_ds.shape[0], im_ds.shape[0]))
        joints_loc_image = draw_kin_tree(joints_loc_image, joints_list, kin_thickness=5)
        im = np.concatenate([joints_loc_image, im_skin, im_skel], axis=1)
        
        plt.imshow(im)
        plt.suptitle(title)
        plt.show()

           