"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See LICENCE.txt for licensing and contact information.
"""

import argparse
from matplotlib import pyplot as plt
from src.dxa_processing import compute_skel_mask, compute_skin_mask
from src.loaders.dxa_loader import DxaLoader


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Load and display Biobank data for a subject') 
    parser.add_argument('--subject_id', type=str, default='1000098', help='ID of the subject to display')
    parser.add_argument('--gender', type=str, default='female', help='Gender of the subject. The processing differs slightly')
    args = parser.parse_args()
    
    ukb_dataloader = DxaLoader()

    ukb_data = ukb_dataloader.load_dxa_raw_data(ukb_dataloader.get_patient_folder(args.subject_id), args.gender)
    
    im_ds = ukb_data['skin']
    im_dk = ukb_data['skel']    

    #Extract masks from dxa
    im_ds_pp = compute_skin_mask(ukb_data)  
    im_dk_pp = compute_skel_mask(ukb_data, im_ds_pp)

    # Show in a 2x2 subplot
    fig, ax = plt.subplots(2, 2, figsize=(10,10))
    ax[0, 0].imshow(im_ds, cmap='gray')
    ax[0, 0].set_title('DXA skin')
    ax[0, 1].imshow(im_ds_pp, cmap='gray')
    ax[0, 1].set_title('Extracted skin mask')
    ax[1, 0].imshow(im_dk, cmap='gray')
    ax[1, 0].set_title('DXA skeleton')
    ax[1, 1].imshow(im_dk_pp, cmap='gray')
    ax[1, 1].set_title('Extracted skeleton mask')
    
    for axi in ax.flatten():
        axi.set_xticks([])
        axi.set_yticks([])
        
    plt.show()
