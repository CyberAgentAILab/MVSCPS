import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from torch.utils.data import Dataset

from core.registry import REG
from dataloader.utils import io_worker_count, parse_light_idx, parse_view_idx


class DiligentEvalDataset(Dataset):
    """Shared base for the validation and test datasets.

    The two only differ in: the config sub-section they read (`cfg.val` vs
    `cfg.test`), the output-dict key prefix (`val_` vs ``), and a couple of
    test-only extras (a GT mesh path, mesh-resolution image sizes, and using
    `img_mesh_downscale` for the mesh cameras). Subclasses set `stage` and
    `key_prefix`; everything else is derived from the stage config.
    """

    stage = None        # 'val' or 'test'
    key_prefix = ''      # 'val_' for the validation dataset, '' for test

    def __init__(self, cfg):
        self.config = cfg
        stage_cfg = getattr(cfg, self.stage)
        self.img_downscale = stage_cfg.img_downscale
        self.validation_crop = self.config.get('validation_crop', True)

        self.img_load_fn = REG.get("fn", cfg.img_load_fn)
        self.mask_load_fn = REG.get("fn", cfg.mask_load_fn)
        self.camera_load_fn = REG.get("fn", "load_camera")
        self.camera_select_fn = REG.get("fn", "select_camera")

        self.cams = self.camera_load_fn(cfg.cameras_fpath)

        # test provides a GT mesh for Chamfer/F-score; val has no such key (-> None).
        self.gt_mesh_fpath = None if stage_cfg.get('gt_mesh_fpath', "None") == "None" else stage_cfg.gt_mesh_fpath

        with open(stage_cfg.view_light_index_file, 'r') as f:
            self.view_light_indices = f.read().splitlines()
        self.light_idx = np.array([parse_light_idx(idx) for idx in self.view_light_indices])

        if self.config.light_dir_file != "None":
            self.light_dirs_GT = np.loadtxt(self.config.light_dir_file)  # (N_L, 3)
            self.light_intensity_GT = np.loadtxt(self.config.light_int_file)  # (N_L, 3)
        else:
            self.light_dirs_GT = None
            self.light_intensity_GT = None

        # ---- resolve size ----
        sample_img_path = os.path.join(cfg.data_dir, cfg.img_dirname, cfg.sample_img_fname)
        self.img_h0, self.img_w0 = self.img_load_fn(sample_img_path).shape[:2]
        self.img_h = int(self.img_h0 / self.img_downscale)
        self.img_w = int(self.img_w0 / self.img_downscale)
        # test renders the mesh at its own (coarser) resolution.
        if 'img_mesh_downscale' in stage_cfg:
            self.img_h_mesh = int(self.img_h0 / stage_cfg.img_mesh_downscale)
            self.img_w_mesh = int(self.img_w0 / stage_cfg.img_mesh_downscale)

        self.has_mask = True

        self.has_gt_normal = True if cfg.normal_dirname != "None" else False
        self.normal_dirname = cfg.normal_dirname

        # load images and masks
        with ThreadPoolExecutor(max_workers=io_worker_count(64)) as executor:
            def load_img_and_mask(view_light_index):
                view_idx = parse_view_idx(view_light_index)
                img_file = os.path.join(cfg.data_dir, cfg.img_dirname, f'{view_light_index}.{cfg.img_ext}')
                mask_path = os.path.join(cfg.data_dir, cfg.mask_dirname, f'V{view_idx:02d}.{cfg.mask_ext}')
                return img_file, mask_path
            self.images, self.fg_masks = zip(*list(executor.map(load_img_and_mask, self.view_light_indices)))

        # load normal maps, evaluation purpose only
        if self.has_gt_normal:
            self.normal_load_fn = REG.get("fn", cfg.normal_load_fn)
            with ThreadPoolExecutor(max_workers=io_worker_count(64)) as executor:
                def load_normal(view_light_index):
                    view_idx = parse_view_idx(view_light_index)
                    normal_path = os.path.join(self.config.data_dir, self.normal_dirname, f'V{view_idx:02d}.exr')
                    return normal_path
                self.normal_maps = list(executor.map(load_normal, self.view_light_indices))

        with ThreadPoolExecutor(max_workers=io_worker_count(64)) as executor:
            C2W_list, K_list = zip(*list(executor.map(
                partial(self.camera_select_fn, cams=self.cams, img_downscale=self.img_downscale), self.view_light_indices)))

        self.C2W_img = np.stack(C2W_list, 0)  # (N, 3, 4)
        self.K_img = np.stack(K_list, 0)  # (N, 3, 3)
        self.camera_centers_img = self.C2W_img[:, :3, 3]  # (N, 3)

        self.O2W_scale = self.cams["O2W_scale"]
        self.O2W_translation = np.array(self.cams["O2W_translation"])
        self.camera_centers_obj_space_img = (self.camera_centers_img - self.O2W_translation) / self.O2W_scale

        self.num_images = self.C2W_img.shape[0]

        # read in the training view-light indices
        with open(self.config.train.view_light_index_file, 'r') as f:
            self.view_light_indices_train = f.read().splitlines()

        self.unique_train_light_indices = np.unique([parse_light_idx(idx) for idx in self.view_light_indices_train])
        self.unique_train_view_indices = np.unique(
            [parse_view_idx(idx) for idx in self.view_light_indices_train])

        # select the view-light indices with unique view indices
        self.unique_train_view_light_indices = []
        for i in self.unique_train_view_indices:
            for j in self.view_light_indices_train:
                if parse_view_idx(j) == i:
                    self.unique_train_view_light_indices.append(j)
                    break

        # test renders the mesh at img_mesh_downscale; val reuses img_downscale.
        mesh_downscale = stage_cfg.get('img_mesh_downscale', self.img_downscale)
        with ThreadPoolExecutor(max_workers=io_worker_count(64)) as executor:
            C2W_mesh, K_mesh = zip(*list(executor.map(partial(self.camera_select_fn, cams=self.cams,
                    img_downscale=mesh_downscale), self.unique_train_view_light_indices)))

        self.C2W_mesh = np.stack(C2W_mesh, 0)
        self.K_mesh = np.stack(K_mesh, 0)

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        view_light_idx = self.view_light_indices[index]
        view_idx = parse_view_idx(view_light_idx)
        light_idx = parse_light_idx(view_light_idx)

        view_type = "train" if view_idx in self.unique_train_view_indices else "test"
        light_type = "train" if light_idx in self.unique_train_light_indices else "test"

        c2w = self.C2W_img[index]
        R_w2c = c2w[:3, :3].T  # (3, 3)
        R_c2w = c2w[:3, :3]

        mask = self.mask_load_fn(self.fg_masks[index], self.img_downscale)
        if self.validation_crop:
            # Render only the foreground bounding box (faster than the full frame).
            # axis0 = rows (image y), axis1 = cols (image x).
            mask_axis0, mask_axis1 = np.where(mask)
            min_mask_axis0 = mask_axis0.min()
            max_mask_axis0 = mask_axis0.max()
            min_mask_axis1 = mask_axis1.min()
            max_mask_axis1 = mask_axis1.max()

            val_img_h = max_mask_axis0 - min_mask_axis0
            val_img_w = max_mask_axis1 - min_mask_axis1
        else:
            val_img_h = self.img_h
            val_img_w = self.img_w

        xx, yy = np.meshgrid(
            np.arange(val_img_w, dtype=np.float32),
            np.arange(val_img_h, dtype=np.float32),
            indexing='xy'
        )

        K = self.K_img[index]
        cx = K[0, 2]
        cy = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        if self.validation_crop:
            # shift the principal point into the cropped frame (K was for the full image)
            cx = cx - min_mask_axis1
            cy = cy - min_mask_axis0
            fg_mask = mask[min_mask_axis0:max_mask_axis0, min_mask_axis1:max_mask_axis1]

        directions = np.stack([(xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)], -1)
        directions = directions[fg_mask]  # (B, 3)

        rays_o = self.camera_centers_obj_space_img[index]
        rays_o = np.repeat(rays_o[None], repeats=directions.shape[0], axis=0)  # (B, 3)
        rays_d = np.einsum('ij,bj->bi', c2w[:3, :3], directions)  # (B, 3)
        rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True)

        light_index = self.light_idx[index]
        rays_light_index = np.repeat(light_index[None], repeats=directions.shape[0], axis=0)[..., None]  # (B, 1)
        rays_R_c2w = np.repeat(R_c2w.reshape(1, 9), repeats=directions.shape[0], axis=0)  # (B, 9)

        # (B, 16) ray bundle expected by NeuSModel.forward:
        # [0:3] origin, [3:6] direction, [6] light index, [7:16] R_c2w (3x3 flattened)
        rays = np.concatenate([rays_o, rays_d, rays_light_index, rays_R_c2w], -1)

        rgb = self.img_load_fn(self.images[index], self.img_downscale)
        if self.validation_crop:
            rgb = rgb[min_mask_axis0:max_mask_axis0, min_mask_axis1:max_mask_axis1]

        if self.has_gt_normal:
            normal = self.normal_load_fn(self.normal_maps[index])
            if self.validation_crop:
                normal = normal[min_mask_axis0:max_mask_axis0, min_mask_axis1:max_mask_axis1]

        p = self.key_prefix
        batch_data = {
            'rays': rays.astype(np.float32),
            "c2w": c2w.astype(np.float32),
            'R_c2w': c2w[:3, :3].astype(np.float32),
            'R_w2c': R_w2c.astype(np.float32),

            'image_rgb': rgb.astype(np.float32),
            'image_fg_mask': fg_mask.astype(np.float32),

            f'{p}view_type': view_type,
            f'{p}light_type': light_type,
            f'{p}view_light_idx': view_light_idx,
        }
        if self.has_gt_normal:
            batch_data.update({'image_normal_camera_space': normal.astype(np.float32),})

        return batch_data
