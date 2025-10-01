import math

import numpy as np
import open3d as o3d
import torch


def get_camera_space_ray_directions(img_height,
                                    img_width,
                                    K,
                                    use_pixel_centers=False,
                                    return_uvs=False,
                                    return_flat=False):
    # Get ray directions for all pixels in camera coordinates
    # OpenCV convention (x right, y down, z front)
    pixel_center = 0.5 if use_pixel_centers else 0
    uu, vv = np.meshgrid(
        np.arange(img_width, dtype=np.float32) + pixel_center,
        np.arange(img_height, dtype=np.float32) + pixel_center, indexing='xy')  # uu right, vv down

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    directions = np.stack([(uu - cx) / fx,
                           (vv - cy) / fy,
                           np.ones_like(uu)], axis=-1) # (H, W, 3)
    if return_flat:
        directions = directions.reshape(-1, 3)  # (H*W, 3)

    if return_uvs:
        uvs = np.stack([uu, vv], axis=-1)
        return directions, uvs
    return directions

def get_world_space_ray_directions(directions_cam, c2w):
    # The shape of directions and c2w can be:
    # 1. directions: (N,3), c2w: (3,4)
    # 2. directions: (N,3), c2w: (N,3,4)
    # 3. directions: (H,W,3), c2w: (3,4)
    # return rays_d as the same shape as directions_cam
    R_c2w = c2w[..., :3, :3]  # (3,3) or (N,3,3)
    if directions_cam.ndim == 2:  # (N,3):
        assert directions_cam.shape[1] == 3
        if c2w.ndim == 2:
            rays_d = np.einsum('ij,nj->ni', R_c2w, directions_cam)  # (N,3)
        elif c2w.ndim == 3:  # (N,3,4)
            rays_d = np.einsum('nij,nj->ni', R_c2w, directions_cam)  # (N,3)
    else:  # (H,W,3)
        assert directions_cam.shape[2] == 3
        assert c2w.ndim == 2
        rays_d = np.einsum('ij,hwj->hwi', R_c2w, directions_cam)
    return rays_d


def generate_sphere_normal_map(res=512):
    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    xx, yy = np.meshgrid(x, y)

    zz = - np.sqrt(1 - xx ** 2 - yy ** 2)
    fg_mask = (1 - xx ** 2 - yy ** 2) > 0

    normal_map = np.stack([xx, yy, zz], -1)
    return normal_map, fg_mask


def generate_rotation_camera_poses(num_poses, radius=5.0, lookat=(0., 0., 0.), elevation=40.):
    c2ws = []
    for i in range(num_poses):
        theta = 2 * math.pi * i / num_poses
        x = math.cos(theta) * math.cos(math.radians(elevation)) * radius
        y = math.sin(theta) * math.cos(math.radians(elevation))  * radius
        z = math.sin(math.radians(elevation))  * radius

        camera_center = np.array([x, y, z])

        z_axis =  np.array(lookat) - camera_center
        z_axis = z_axis / np.linalg.norm(z_axis)
        x_axis = np.cross(z_axis, np.array([0, 0, 1]))
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        R = np.stack([x_axis, y_axis, z_axis], axis=1)

        C2W = np.column_stack((R, camera_center))
        c2ws.append(C2W)

    # convert to tensor
    c2ws = torch.from_numpy(np.stack(c2ws, axis=0)).float()
    return c2ws


def generate_ring_lights(num_lights, elevation=60.):
    lights = []
    for i in range(num_lights):
        theta = 2 * math.pi * i / num_lights
        x = math.cos(theta) * math.cos(math.radians(elevation))
        y = math.sin(theta) * math.cos(math.radians(elevation))
        z = math.sin(math.radians(elevation))
        lights.append([x, y, -z])
    lights = torch.from_numpy(np.array(lights, dtype=np.float32))
    return lights


def visualize_camera_poses(C2W_list, img_list=None, K=None):
    things_to_draw = []

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1, resolution=10)
    sphere = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    sphere.paint_uniform_color((1, 0, 0))
    things_to_draw.append(sphere)

    if K is not None:
        K[0, 0] /= 4
        K[1, 1] /= 4
        K[0, 2] /= 4
        K[1, 2] /= 4
    for idx, C2W in enumerate(C2W_list):
        c = C2W[:3, 3]
        x_axis = C2W[:3, 0] * 0.5
        y_axis = C2W[:3, 1] * 0.5
        z_axis = C2W[:3, 2] * 0.5

        points = [c, c+x_axis, c+y_axis, c+z_axis]  # (4, 3)
        lines = [[0, 1],
                 [0, 2],
                 [0, 3]]
        colors = [[1, 0, 0],
                  [0, 1, 0],
                  [0, 0, 1]]  # (r,g,b)

        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(points)
        lineset.lines = o3d.utility.Vector2iVector(lines)
        lineset.colors = o3d.utility.Vector3dVector(colors)
        things_to_draw.append(lineset)
        if img_list is not None:
            img = img_list[idx]
            # resize img to 1/4
            img = img[::4, ::4]
            H, W = img.shape[:2]
            xx, yy = np.meshgrid(range(W), range(H))
            yy = np.flip(yy, axis=0)
            # ic(xx, yy)

            u = np.zeros((H, W, 3))
            u[..., 0] = xx
            u[..., 1] = yy
            u[..., 2] = 1
            u = u.reshape(-1, 3).T # 3 x m
            pixel_coord_in_camera = (np.linalg.inv(K[:3, :3]) @ u).T  # m x 3

            pixel_coord_in_camera_homo = np.concatenate((pixel_coord_in_camera * 1,
                                                         np.ones((pixel_coord_in_camera.shape[0], 1))), axis=-1)
            pixel_coord_in_world = (C2W[:3] @ pixel_coord_in_camera_homo.T).T


            img_pcd = o3d.geometry.PointCloud()
            img_pcd.points = o3d.utility.Vector3dVector(pixel_coord_in_world)
            img_pcd.colors = o3d.utility.Vector3dVector(img.reshape(-1, 3))
            things_to_draw.append(img_pcd)

    o3d.visualization.draw_geometries(things_to_draw)