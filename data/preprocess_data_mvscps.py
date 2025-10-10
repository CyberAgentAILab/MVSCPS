import glob

import os, json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import cv2
from tqdm import tqdm
from data_utils import *

if __name__ == "__main__":
    obj_names = [
        "20250205_ceramic_buddha",
        "20250304_diffuse_dog",
        "20250304_diffuse_flower_girl",
        "20250304_metallic_fox",
        "20250303_bronze_loong",
        "20250515_ceramic_buddha"
    ]

    for obj_name in obj_names:
        print(f"Processing object: {obj_name}")
        obj_dir = os.path.join("data", "mvscps", obj_name)
        mask_dir = os.path.join(obj_dir, "mask")

        cam_list = glob.glob(os.path.join(obj_dir, "CAM", "*_P.txt"))
        cam_list.sort(key=lambda x: int(x.split("/")[-1].split("V")[-1].split("L")[0]))

        view_light_idx = [os.path.basename(fpath).split("_")[0] for fpath in cam_list]
        # save view_light_idx into a txt file
        with open(os.path.join(obj_dir, "view_light_idx_all.txt"), "w") as f:
            for item in view_light_idx:
                f.write("%s\n" % item)

        # load camera matrices
        with ThreadPoolExecutor() as executor:
            P = np.array(list(executor.map(np.loadtxt, cam_list)))  # shape: (N, 3, 4)

        cam_fnames = [os.path.basename(cam_path).split("_")[0] for cam_path in cam_list]
        mask_fname_list = [os.path.join(mask_dir, cam_fname + ".png") for cam_fname in cam_fnames]

        # load masks
        with ThreadPoolExecutor() as executor:
            def load_mask(mask_path):
                mask = cv2.imread(mask_path, -1)
                mask = mask.astype(bool)
                return mask
            mask_list = list(executor.map(load_mask, mask_fname_list))

        # rename masks
        for i, mask_path in enumerate(mask_fname_list):
            view_idx = int(os.path.basename(mask_path).split("L")[0][1:])
            new_mask_path = os.path.join(mask_dir, f"V{view_idx:02d}.png")
            if mask_path != new_mask_path:
                os.rename(mask_path, new_mask_path)

        s_O2W, d_O2W = scene_normalization(P, mask_list, fg_area_ratio=5)
        cam_dict = {}
        cam_dict["O2W_scale"] = s_O2W
        cam_dict["O2W_translation"] = d_O2W

        draw_dir = os.path.join(obj_dir, "scene_normalization")
        os.makedirs(draw_dir, exist_ok=True)

        for i, cam_fname in tqdm(enumerate(cam_fnames)):
            K, C2W = load_K_Rt_from_P(P[i])
            cam_dict.update({
                f"K_{cam_fname}": K,
                f"C2W_{cam_fname}": C2W
            })

            # draw the scene normalization results
            fpath = os.path.join(draw_dir, f"{cam_fname}.png")
            visualize_scene_normalization(P[i], mask_list[i], s_O2W, d_O2W, fpath, downscale_factor=8)

        # save the camera parameters
        with open(os.path.join(obj_dir, 'camera_params.json'), 'w', encoding="utf-8") as f:
            json.dump(convert_numpy(cam_dict), f, indent=4, sort_keys=True)







