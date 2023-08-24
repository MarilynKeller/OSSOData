"""
Code from https://github.com/mmatl/pyrender/blob/master/examples/example.py, modified by Marilyn Keller.
"""


import pyglet
pyglet.options['shadow_window'] = False
import numpy as np
import trimesh

from pyrender import OrthographicCamera,\
                    PointLight,\
                    Mesh, Node, Scene,\
                    OffscreenRenderer, RenderFlags
                     
                     
def render_dxa_mesh(mesh_path, dxa_data, transparent=True, silhouette=False):

    im_height= dxa_data['skin'].shape[0]
    fov = dxa_data['field_of_view']

    if silhouette:
        black_color = np.zeros((3))
        mesh_render = render_mesh(mesh_path, fov=fov, im_height=im_height, use_light=False, color=black_color, transparent=transparent)
        mesh_render = 255 - mesh_render
        
        # Binarise the image
        mesh_render[mesh_render <= 128] = 0
        mesh_render[mesh_render > 128] = 255
        mesh_render = mesh_render[:,:,0] # Keep only one channel

    else:
        mesh_render = render_mesh(mesh_path, fov=fov, im_height=im_height, color=None, transparent=transparent, use_light=True)

    return mesh_render


def render_mesh(mesh_path, fov=[1,1], im_height=500, color=None, transparent=True, use_light=True):

    #==============================================================================
    # Mesh creation
    #==============================================================================

    #------------------------------------------------------------------------------
    # Creating meshes from trimeshes
    #------------------------------------------------------------------------------

    # Fuze trimesh
    fuze_trimesh = trimesh.load(mesh_path)
    fuze_trimesh.visual.vertex_colors = [ 255 , 255, 255]
    if color is not None:
        fuze_trimesh.visual.vertex_colors = list((color*255).astype(np.uint8)) + [255]
    fuze_mesh = Mesh.from_trimesh(fuze_trimesh)

    #==============================================================================
    # Light creation
    #==============================================================================
    
    if use_light:
        key_light = PointLight(color=np.array([1., 0.90, 0.85]), intensity=30.0)
        fill_light = PointLight(color=np.array([0.8, 0.9, 1]), intensity=5.0)
        rim_light = PointLight(color=np.array([1., 1., 1.]), intensity=50.0)

    #==============================================================================
    # Camera creation
    #==============================================================================

    cam = OrthographicCamera(xmag=0.5*fov[1], ymag=0.5*fov[1])
    cam_pose = np.array([
        [1.0,  0, 0, 0],
        [0.0, 1.0,           0.0,           0.0],
        [0.0,  0,       1, 3],
        [0.0,  0.0,           0.0,          1.0]
    ])

    #==============================================================================
    # Scene creation
    #==============================================================================

    scene = Scene(ambient_light=np.array([0.01, 0.01, 0.015, 1.0]),  bg_color=1 * np.array([1.0, 1.0, 1.0, 0.0]))

    #==============================================================================
    # Adding objects to the scene
    #==============================================================================

    #------------------------------------------------------------------------------
    # By manually creating nodes
    #------------------------------------------------------------------------------
    fuze_node = Node(mesh=fuze_mesh, translation=np.array([0, 0, 0]))
    scene.add_node(fuze_node)

    if use_light:
        key_light_node = Node(light=key_light, translation=2*np.array([-0.7,0.8,1]))
        scene.add_node(key_light_node)

        fill_light_node = Node(light=fill_light, translation=2*np.array([1, -0.5, 0.5]))
        scene.add_node(fill_light_node)

        rim_light_node = Node(light=rim_light, translation=2*np.array([0.3, 1,-1.5]))
        scene.add_node(rim_light_node) 


    #==============================================================================
    # Using the viewer with a pre-specified camera
    #==============================================================================
    cam_node = scene.add(cam, pose=cam_pose)
    # v = Viewer(scene)

    #==============================================================================
    # Rendering offscreen from that camera
    #==============================================================================

    r = OffscreenRenderer(viewport_width=im_height, viewport_height=im_height)
    if transparent:
        color, depth = r.render(scene, flags=RenderFlags.RGBA)
    else:
        color, depth = r.render(scene)

    return color



if __name__ == "__main__":
    mesh_path = '/is/cluster/fast/mkeller2/Data/Skeleton/biobank_return/biobank_return/body/1000098_body.ply'
    color = render_mesh(mesh_path)

    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(color)
    plt.show()
