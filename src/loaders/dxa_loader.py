"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller and Sergi Pujades
See LICENCE.txt for licensing and contact information.
"""

import os
import glob
from pydicom import dcmread
import config as cg


class DxaLoader :
    """ Handle loading a DXA from UK Biobank database"""
    
    def __init__(self, ukbiobank_data_root=cg.ukbiobank_data_root) -> None:
        self.ukb_root = ukbiobank_data_root
        pass
    
    def get_patient_folder(self, patient_id):
        candidates = glob.glob(os.path.join(self.ukb_root, f'*/{patient_id}_*'))
        assert len(candidates)>0 , f'No patient folder found for patient {patient_id}'
        if len(candidates) > 1:
            print(f'Found {len(candidates)} candidate folders for patient {patient_id}: {candidates}. Keeping the first one.')
        candidate_folder = candidates[0]
        return candidate_folder
    
    def get_patient_scans_files(self, patient_folder):
        """Return path of the skin and skel dxa dicom"""
        patient_folder_path = os.path.join(self.ukb_root, patient_folder)
        patient_files = os.listdir(patient_folder_path)
        skin_pattern = '11.12.1.dcm'
        skel_pattern = '12.12.1.dcm'
        dxa_skin_files = [f for f in patient_files if skin_pattern in f]
        dxa_skel_files = [f for f in patient_files if skel_pattern in f]
        for dxa_files, pattern in  [[dxa_skin_files, skin_pattern], [dxa_skel_files, skel_pattern]]:
            if len(dxa_files)!=1:
                print(f'Patient folder {patient_folder} should contain one file ending in {pattern} in folder. Contains {len(dxa_files)} files.')
                return None
        skin_path = os.path.join(patient_folder, dxa_skin_files[0])
        skel_path = os.path.join(patient_folder, dxa_skel_files[0])
        return skin_path, skel_path
    
    def load_dxa_scan(self, patient_folder):
        """
        :param i: index
        :param paths_file: file containing one relative path by line giving the path to a patient folder
        :return: dicom file
        """
        assert os.path.exists(patient_folder), f' {patient_folder} does not exist'

        skin_path, skel_path =  self.get_patient_scans_files(patient_folder)

        ds = dcmread(os.path.join(self.ukb_root, skin_path))
        dk = dcmread(os.path.join(self.ukb_root, skel_path))
        return ds, dk

    def get_patient_name(self, patient_folder):
        return patient_folder[-17:-10]

    def load_dxa_raw_data(self, patient_folder, gender=None):
        """

        :param i: index
        :param paths_file: file containing one relative path by line giving the path to a patient folder
        :return: data dictionnary containing the following keys:
            'patient_id' : patient id
            'skin' : DXA body image
            'skel' : DXA skeleton image
            'field_of_view' : field of view of the DXA image, in meters
            'height' : height of the patient
            'weight' : weight of the patient
        """
        
        assert os.path.exists(patient_folder), f' {patient_folder} does not exist'

        ds, dk = self.load_dxa_scan(patient_folder)
        assert ds is not None
        assert dk is not None

        im_ds = ds.pixel_array
        im_dk = dk.pixel_array
        patient_height = ds.PatientSize
        patient_weight = ds.PatientWeight
        fov_mm = ds.ExposedArea
        fov = [fov_mm[0]/1000, fov_mm[1]/1000]

        data = {}
        data['patient_id'] = self.get_patient_name(patient_folder)
        data['skin'] = im_ds
        data['skel'] = im_dk
        data['field_of_view'] = fov
        data['height'] = patient_height
        data['weight'] = patient_weight
        data['gender'] = gender

        return data


