"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller
See LICENCE.txt for licensing and contact information.
"""

import argparse
from src.loaders.return_loader import ReturnData

if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Display Biobank data', epilog='For each subject, display the body and skeleton meshes, as well as the STAR parameters')

    parser.add_argument('--subject_id', type=str, default='1000098', help='ID of the subject to display')
    
    args = parser.parse_args()
    
    return_data_loader = ReturnData()
    return_data_loader.display_subject_data(args.subject_id)
    