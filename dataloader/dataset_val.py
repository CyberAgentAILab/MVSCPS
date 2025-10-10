import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from torch.utils.data import Dataset

from core.registry import REG


@REG.register("dataset", name='dataset_val')
class DiligentValDataset(Dataset):
    def __init__(self, cfg):
        self.config = cfg
        self.validation_crop = self.config.get('validation_crop', True)

        self.img_load_fn = REG.get("fn", cfg.img_load_fn)
        self.mask_load_fn = REG.get("fn", cfg.mask_load_fn)
        self.camera_load_fn = REG.get("fn", "load_camera")
        self.camera_select_fn = REG.get("fn", "select_camera")

        self.cams = self.camera_load_fn(cfg.cameras_fpath)

        with open(self.config.val.view_light_index_file, 'r') as f:
            self.view_light_indices = f.read().splitlines()
        self.light_idx = np.array([int(idx.split("L")[-1]) for idx in self.view_light_indices])

        if self.config.light_dir_file != "None":
            self.light_dirs_GT = np.loadtxt(self.config.light_dir_file)  # (N_L, 3)
            self.light_intensity_GT = np.loadtxt(self.config.light_int_file)  # (N_L, 3)
        else:
            self.light_dirs_GT = None
            self.light_intensity_GT = None

        # ---- resolve size ----
        sample_img_path = os.path.join(cfg.data_dir, cfg.img_dirname, cfg.sample_img_fname)
        self.img_h0, self.img_w0 = self.img_load_fn(sample_img_path).shape[:2]
        self.img_h = int(self.img_h0 / cfg.val.img_downscale)
        self.img_w = int(self.img_w0 / cfg.val.img_downscale)

        self.has_mask = True

        self.has_gt_normal = True if cfg.normal_dirname != "None" else False
        self.normal_dirname = cfg.normal_dirname

        # load images and masks
        with ThreadPoolExecutor(max_workers=min(64, os.cpu_count())) as executor:
            def load_img_and_mask(view_light_index):
                view_idx = int(view_light_index.split("V")[1].split("L")[0])
                img_file = os.path.join(cfg.data_dir, cfg.img_dirname, f'{view_light_index}.{cfg.img_ext}')
                mask_path = os.path.join(cfg.data_dir, cfg.mask_dirname, f'V{view_idx:02d}.{cfg.mask_ext}')
                return img_file, mask_path
            self.images_val, self.fg_masks_val = zip(*list(executor.map(load_img_and_mask, self.view_light_indices)))

        # load normal maps, evaluation purpose only
        if self.has_gt_normal:
            self.normal_load_fn = REG.get("fn", cfg.normal_load_fn)
            with ThreadPoolExecutor(max_workers=min(64, os.cpu_count())) as executor:
                def load_normal(view_light_index):
                    view_idx = int(view_light_index.split("V")[1].split("L")[0])
                    normal_path = os.path.join(self.config.data_dir, self.normal_dirname, f'V{view_idx:02d}.exr')
                    return normal_path
                self.normal_maps_val = list(executor.map(load_normal, self.view_light_indices))

        with ThreadPoolExecutor(max_workers=min(64, os.cpu_count())) as executor:
            C2W_val_list, K_val_list = zip(*list(executor.map(
                partial(self.camera_select_fn, cams=self.cams, img_downscale=cfg.val.img_downscale), self.view_light_indices)))

        self.C2W_val_img = np.stack(C2W_val_list, 0)  # (N, 3, 4)
        self.K_val_img = np.stack(K_val_list, 0)  # (N, 3, 3)
        self.camera_centers_val_img = self.C2W_val_img[:, :3, 3]  # (N, 3)

        self.O2W_scale = self.cams["O2W_scale"]
        self.O2W_translation = np.array(self.cams["O2W_translation"])
        self.camera_centers_obj_space_val_img = (self.camera_centers_val_img - self.O2W_translation) / self.O2W_scale

        self.num_imgs_val = self.C2W_val_img.shape[0]

        # read in the training view-light indices
        with open(self.config.train.view_light_index_file, 'r') as f:
            self.view_light_indices_train = f.read().splitlines()

        self.unique_train_light_indices = np.unique([int(idx.split("L")[-1]) for idx in self.view_light_indices_train])
        self.unique_train_view_indices = np.unique(
            [int(idx.split("V")[1].split("L")[0]) for idx in self.view_light_indices_train])

        # select the view-light indices with unique view indices
        self.unique_train_view_light_indices = []
        for i in self.unique_train_view_indices:
            for j  in self.view_light_indices_train:
                if int(j.split("V")[1].split("L")[0]) == i:
                    self.unique_train_view_light_indices.append(j)
                    break

        with ThreadPoolExecutor(max_workers=min(64, os.cpu_count())) as executor:
            C2W_val_mesh, K_val_mesh = zip(*list(executor.map(partial(self.camera_select_fn, cams=self.cams,
                    img_downscale=cfg.val.img_downscale), self.unique_train_view_light_indices)))

        self.C2W_val_mesh = np.stack(C2W_val_mesh, 0)
        self.K_val_mesh = np.stack(K_val_mesh, 0)

    def __len__(self):
        return self.num_imgs_val

    def __getitem__(self, index):
        val_view_light_idx = self.view_light_indices[index]
        val_view_idx = int(val_view_light_idx.split("V")[1].split("L")[0])
        val_light_idx = int(val_view_light_idx.split("L")[1])

        view_type = "train" if val_view_idx in self.unique_train_view_indices else "test"
        light_type = "train" if val_light_idx in self.unique_train_light_indices else "test"

        c2w = self.C2W_val_img[index]
        R_w2c = c2w[:3, :3].T  # (3, 3)
        R_c2w = c2w[:3, :3]

        # mask = self.load_mask(self.fg_masks_val[index])
        mask = self.mask_load_fn(self.fg_masks_val[index], self.config.val.img_downscale)
        if self.validation_crop:
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

        K = self.K_val_img[index]
        cx = K[0, 2]
        cy = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        if self.validation_crop:
            cx = cx - min_mask_axis1
            cy = cy - min_mask_axis0
            fg_mask = mask[min_mask_axis0:max_mask_axis0, min_mask_axis1:max_mask_axis1]

        directions = np.stack([(xx - cx) / fx, (yy - cy) / fy, np.ones_like(xx)], -1)
        directions = directions[fg_mask]  # (B, 3)

        rays_o = self.camera_centers_obj_space_val_img[index]
        rays_o = np.repeat(rays_o[None], repeats=directions.shape[0], axis=0)  # (B, 3)
        rays_d = np.einsum('ij,bj->bi', c2w[:3, :3], directions)  # (B, 3)
        rays_d /= np.linalg.norm(rays_d, axis=-1, keepdims=True)

        light_index = self.light_idx[index]
        rays_light_index = np.repeat(light_index[None], repeats=directions.shape[0], axis=0)[..., None]  # (B, 1)
        rays_R_c2w = np.repeat(R_c2w.reshape(1, 9), repeats=directions.shape[0], axis=0)  # (B, 9)

        rays = np.concatenate([rays_o, rays_d, rays_light_index, rays_R_c2w], -1)

        rgb = self.img_load_fn(self.images_val[index], self.config.val.img_downscale)
        if self.validation_crop:
            rgb = rgb[min_mask_axis0:max_mask_axis0, min_mask_axis1:max_mask_axis1]

        if self.has_gt_normal:
            normal = self.normal_load_fn(self.normal_maps_val[index])
            if self.validation_crop:
                normal = normal[min_mask_axis0:max_mask_axis0, min_mask_axis1:max_mask_axis1]

        batch_data = {
            'rays': rays.astype(np.float32),
            "c2w": c2w.astype(np.float32),
            'R_c2w': c2w[:3, :3].astype(np.float32),
            'R_w2c': R_w2c.astype(np.float32),

            'image_rgb': rgb.astype(np.float32),
            'image_fg_mask': fg_mask.astype(np.float32),

            'val_view_type': view_type,
            'val_light_type': light_type,
            'val_view_light_idx': val_view_light_idx,
        }
        if self.has_gt_normal:
            batch_data.update({'image_normal_camera_space': normal.astype(np.float32),})

        return batch_data