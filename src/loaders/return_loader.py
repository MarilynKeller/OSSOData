"""
Copyright©2023 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Author: Marilyn Keller and Sergi Pujades
See LICENCE.txt for licensing and contact information.
"""

import os
import pickle
from psbody.mesh import Mesh, MeshViewers
import config as cg

data_path = cg.return_data_path

class ReturnData:
    """ Handle loading meshes from our UK Biobank return"""
    
    def __init__(self):
        body_meshes_path = os.path.join(data_path, 'body')
        self.list_subjects = os.listdir(body_meshes_path)

    
    def get_subject_paths(self, subject_id):
        # Return the paths to the subject data
        body_mesh_path = os.path.join(data_path, 'body', subject_id + '_body.ply')
        skeleton_mesh_path = os.path.join(data_path, 'skeleton', subject_id + '_skeleton.ply')
        star_params_path = os.path.join(data_path, 'STAR_params', subject_id + '_star_params.pkl')
        return body_mesh_path, skeleton_mesh_path, star_params_path
    
    def load_subject_paths(self, subject_id):
        # Return the subject data
        body_mesh_path, skeleton_mesh_path, star_params_path = self.get_subject_paths(subject_id)
        print(body_mesh_path, skeleton_mesh_path, star_params_path)
        for ff in [body_mesh_path, skeleton_mesh_path, star_params_path]:
            if not os.path.exists(ff):
                raise ValueError(f'File {ff} does not exist')
        body_mesh = Mesh(filename=body_mesh_path)
        skeleton_mesh = Mesh(filename=skeleton_mesh_path)
        star_params = pickle.load(open(star_params_path, 'rb'))
        
        data_dict = {'body_mesh': body_mesh, 'skeleton_mesh': skeleton_mesh, 'star_params': star_params, 
                     'body_mesh_path': body_mesh_path, 'skeleton_mesh_path': skeleton_mesh_path, 'star_params_path': star_params_path}
        return data_dict
        

    def display_subject_data(self, subject_id):
        # Display the meshes of a subject
        
        data_dict = self.load_subject_paths(subject_id)
        body_mesh = data_dict['body_mesh']
        skeleton_mesh = data_dict['skeleton_mesh']
        star_params = data_dict['star_params']
        
        # Print the content of the star_params dictionary
        print('Star parameters:')
        for key, value in star_params.items():
            print('\t', key, value)
        
        mv = MeshViewers((1, 2), keepalive=False)
        mv[0][0].set_titlebar(f'Subject {subject_id}')
        
        mv[0][0].set_static_meshes([body_mesh])
        mv[0][1].set_static_meshes([skeleton_mesh])
        
        print('Click on the figure to terminate.')
        mv[0][0].get_mouseclick()
        
        print('Done.')

