import glob
import os

import numpy as np
from torch.utils.data import Dataset

from core.registry import REG
from dataloader.utils import (generate_sphere_normal_map,
                              get_camera_space_ray_directions,
                              get_world_space_ray_directions)


@REG.register("dataset", name='brdf_render')
class DiligentTestDataset(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        cfg_brdf = cfg.predict_brdf

        self.mask_load_fn = REG.get("fn", cfg.mask_load_fn)
        self.camera_load_fn = REG.get("fn", "load_camera")
        self.camera_select_fn = REG.get("fn", "select_camera")

        self.cams = self.camera_load_fn(cfg.cameras_fpath)

        # ---- resolve size ----
        sample_mask_path = os.path.join(cfg.data_dir, cfg.mask_dirname, cfg.sample_mask_fname)
        self.img_h0, self.img_w0 = self.mask_load_fn(sample_mask_path).shape[:2]
        self.img_h = int(self.img_h0 / cfg_brdf.img_downscale)
        self.img_w = int(self.img_w0 / cfg_brdf.img_downscale)

        self.view_light_index_test = cfg_brdf.view_light_index

        C2W, K = self.camera_select_fn(self.view_light_index_test, self.cams, img_downscale=cfg_brdf.img_downscale)
        rays_dir_camera_space = get_camera_space_ray_directions(self.img_h, self.img_w, K)  # (H, W, 3)

        view_idx = int(self.view_light_index_test.split("V")[1].split("L")[0])
        mask_fpath = glob.glob(os.path.join(cfg.data_dir, cfg.mask_dirname, f'V{view_idx:02d}*.{cfg.mask_ext}'))[0]
        # mask_fpath = os.path.join(cfg.data_dir, cfg.mask_dirname, f'{self.view_light_index_test}.{cfg.mask_ext}')
        fg_mask = self.mask_load_fn(mask_fpath, img_downscale=cfg_brdf.img_downscale)

        # crop foreground regions
        # only render the BRDF spheres within the bounding box of the foreground mask
        mask_axis0, mask_axis1 = np.where(fg_mask)
        min_mask_axis0 = mask_axis0.min()
        max_mask_axis0 = mask_axis0.max()
        min_mask_axis1 = mask_axis1.min()
        max_mask_axis1 = mask_axis1.max()

        self.fg_mask = fg_mask[min_mask_axis0:max_mask_axis0, min_mask_axis1:max_mask_axis1]
        img_h, img_w = self.fg_mask.shape
        uu, vv = np.meshgrid(
            np.arange(img_w, dtype=np.float32),
            np.arange(img_h, dtype=np.float32),
            indexing='xy')
        self.rays_uvs = np.stack([uu, vv], -1).reshape(-1, 2)
        self.rays_dir_camera_space = rays_dir_camera_space[min_mask_axis0:max_mask_axis0, min_mask_axis1:max_mask_axis1].reshape(-1, 3)

        rays_d = get_world_space_ray_directions(self.rays_dir_camera_space, C2W)  # (N, 3)
        rays_o = C2W[:3, 3]
        self.O2W_scale = self.cams["O2W_scale"]
        self.O2W_translation = np.array(self.cams["O2W_translation"])
        rays_o = (rays_o - self.O2W_translation) / self.O2W_scale

        rays_o = np.repeat(rays_o[None], repeats=rays_d.shape[0], axis=0)  # (N, 3)

        self.rays = np.concatenate([rays_o, rays_d], -1)  # (N, 6)
        self.rays = self.rays[self.fg_mask.reshape(-1)]
        self.rays_uvs = self.rays_uvs[self.fg_mask.reshape(-1)]

        # prepare the sphere normal map based on which we render the BRDF sphere per pixel
        self.brdf_sphere_normal_map, self.normal_fg_mask = generate_sphere_normal_map(cfg_brdf.brdf_sphere_res)

        self.light_dir = np.array(cfg_brdf.light_direction).astype(np.float32)
        self.light_dir /= np.linalg.norm(self.light_dir)

    def __len__(self):
        # we only have one test image for BRDF visualization
        return 1

    def __getitem__(self, idx):
        return {
            "rays": self.rays.astype(np.float32),
            "normal_sphere": self.brdf_sphere_normal_map.astype(np.float32),
            "normal_sphere_mask": self.normal_fg_mask.astype(bool),
            "light_dir": self.light_dir.astype(np.float32),
        }