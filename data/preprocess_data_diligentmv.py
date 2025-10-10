import json
import os
import shutil

import cv2
import numpy as np
import pyexr
from data_utils import (convert_numpy, fill_holes_in_mask, load_K_Rt_from_P,
                        scene_normalization, visualize_scene_normalization)
from scipy.io import loadmat
from tqdm import tqdm

root_dir = "data/DiLiGenT-MV_origin/DiLiGenT-MV/mvpmsData"
target_dir = "data/DiLiGenT-MV"

for obj_name in ["bear", "buddha", "pot2", "cow", "reading"]:
    target_image_dir = os.path.join(target_dir, obj_name, "image")
    os.makedirs(target_image_dir, exist_ok=True)
    target_mask_dir = os.path.join(target_dir, obj_name, "mask")
    os.makedirs(target_mask_dir, exist_ok=True)
    target_normal_dir = os.path.join(target_dir, obj_name, "normal_camera_space_GT")
    os.makedirs(target_normal_dir, exist_ok=True)

    # Copy and rename images
    print(f"Copying images for {obj_name}...")
    for view_idx in tqdm(range(20)):
        for light_idx in range(96):
            src_image_path = os.path.join(root_dir, f"{obj_name}PNG", f"view_{view_idx+1:02d}", f"{light_idx+1:03d}.png")
            target_image_path = os.path.join(target_image_dir, f"V{view_idx:02d}L{light_idx:02d}.png")  # 0 indexed
            shutil.copy(src_image_path, target_image_path)

        # Copy masks
        src_mask_path = os.path.join(root_dir, f"{obj_name}PNG", f"view_{view_idx+1:02d}", "mask.png")
        target_mask_path = os.path.join(target_mask_dir, f"V{view_idx:02d}.png")  # 0 indexed
        shutil.copy(src_mask_path, target_mask_path)

        # if the object is bear, fill the holes in its masks
        if obj_name == "bear":
            mask = fill_holes_in_mask(target_mask_path)
            cv2.imwrite(target_mask_path, mask)

        # Copy GT camera-space normal maps
        normal_fpath = os.path.join(root_dir, f"{obj_name}PNG", f"view_{view_idx+1:02d}", "Normal_gt.mat")
        normal = loadmat(normal_fpath)["Normal_gt"]
        target_normal_fpath = os.path.join(target_normal_dir, f"V{view_idx:02d}.exr")
        pyexr.write(target_normal_fpath, normal)

    # Copy camera and lighting.py parameters
    src_camera_path = os.path.join(root_dir, f"{obj_name}PNG", "Calib_Results.mat")
    src_mesh_gt_path = os.path.join(root_dir, f"{obj_name}PNG", "mesh_Gt.ply")

    # light directions and intensities are identical for all views, so we only need to copy them once
    src_light_dir_path = os.path.join(root_dir, f"{obj_name}PNG", f"view_01", "light_directions.txt")
    src_light_int_path = os.path.join(root_dir, f"{obj_name}PNG", f"view_01", "light_intensities.txt")

    target_camera_path = os.path.join(target_dir, obj_name, "Calib_Results.mat")
    target_mesh_gt_path = os.path.join(target_dir, obj_name, "mesh_Gt.ply")
    target_light_dir_path = os.path.join(target_dir, obj_name, "light_directions.txt")
    target_light_int_path = os.path.join(target_dir, obj_name, "light_intensities.txt")

    shutil.copy(src_camera_path, target_camera_path)
    shutil.copy(src_mesh_gt_path, target_mesh_gt_path)
    shutil.copy(src_light_dir_path, target_light_dir_path)
    shutil.copy(src_light_int_path, target_light_int_path)

    # compute the O2W scale and translation using camera parameters and masks
    # load camera_parameters
    cams = loadmat(target_camera_path)
    K = cams["KK"][:3, :3]
    P_list = []
    mask_list = []
    W2C_list = []
    for view_idx in tqdm(range(1, 21)):
        R_W2C = cams[f"Rc_{view_idx}"]
        t_W2C = cams[f"Tc_{view_idx}"]

        W2C = np.eye(4)
        W2C[:3, :3] = R_W2C
        W2C[:3, 3] = t_W2C.flatten()
        P = K @ W2C[:3, :4]  # (3, 4)
        P_list.append(P)
        W2C_list.append(W2C)

        mask_fpath = os.path.join(target_mask_dir, f"V{view_idx-1:02d}.png")
        mask = cv2.imread(mask_fpath, cv2.IMREAD_GRAYSCALE).astype(bool)
        mask_list.append(mask)

    s_O2W, d_O2W = scene_normalization(np.array(P_list), mask_list, fg_area_ratio=5)
    draw_dir = os.path.join(target_dir, obj_name, "scene_normalization")
    os.makedirs(draw_dir, exist_ok=True)

    cam_dict = {}
    cam_dict["O2W_scale"] = s_O2W
    cam_dict["O2W_translation"] = d_O2W
    for view_idx in tqdm(range(20)):
        P = P_list[view_idx]
        K, C2W = load_K_Rt_from_P(P)

        fpath = os.path.join(draw_dir, f"V{view_idx:02d}.png")
        visualize_scene_normalization(P, mask_list[view_idx], s_O2W, d_O2W, fpath, downscale_factor=1)

        for light_idx in range(96):
            cam_fname = f"V{view_idx:02d}L{light_idx:02d}"
            cam_dict.update({
                f"K_{cam_fname}": K,
                f"C2W_{cam_fname}": C2W
            })

    with open(os.path.join(target_dir, obj_name, 'camera_params.json'), 'w', encoding="utf-8") as f:
        json.dump(convert_numpy(cam_dict), f, indent=4, sort_keys=True)