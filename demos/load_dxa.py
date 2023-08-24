"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See LICENCE.txt for licensing and contact information.
"""

import argparse
from matplotlib import pyplot as plt
from src.loaders.dxa_loader import DxaLoader

if __name__=="__main__":
    
    # Example of loading a subject data from ukbiobank DXAs   
    parser = argparse.ArgumentParser(description='Load and display Biobank data for a subject')
    
    parser.add_argument('--subject_id', type=str, default='1000098', help='ID of the subject to display')
    
    args = parser.parse_args()
    
    dl = DxaLoader()
    
    subject_folder = dl.get_patient_folder(args.subject_id)
    print(f'Patient folder: { dl.get_patient_folder(args.subject_id)}')
    print(f'Patient scans: { dl.get_patient_scans_files(subject_folder)}')
    
    data = dl.load_dxa_raw_data(subject_folder)
    
    # Print data dictionary
    for k, v in data.items():
        print(k, v)

    fig = plt.subplot(1,2,1)
    plt.imshow(data['skin'], cmap='gray')
    
    fig = plt.subplot(1,2,2)
    plt.imshow(data['skel'], cmap='gray')
    plt.show()