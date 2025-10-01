import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from icecream import ic
from torch.utils.data import IterableDataset

from core.registry import REG


@REG.register("dataset", name='dataset_train')
class DiligentTrainDataset(IterableDataset):
    def __init__(self, cfg):
        self.config = cfg

        self.img_load_fn = REG.get("fn", cfg.img_load_fn)
        self.mask_load_fn = REG.get("fn", cfg.mask_load_fn)
        self.camera_load_fn = REG.get("fn", "load_camera")
        self.camera_select_fn = REG.get("fn", "select_camera")

        self.cams = self.camera_load_fn(cfg.cameras_fpath)

        self.num_rays_per_batch = cfg.train.init_num_rays_per_batch
        self.max_num_rays_per_batch = cfg.train.max_num_rays_per_batch

        # resolve size
        sample_img_path = os.path.join(cfg.data_dir, cfg.img_dirname, cfg.sample_img_fname)
        self.img_h0, self.img_w0 = self.img_load_fn(sample_img_path).shape[:2]
        self.img_h = int(self.img_h0 / cfg.train.img_downscale)
        self.img_w = int(self.img_w0 / cfg.train.img_downscale)

        # load training image indices
        with open(cfg.train.view_light_index_file, 'r') as f:
            self.view_light_indices = f.read().splitlines()
        self.light_idx = np.array([int(idx.split("L")[-1]) for idx in self.view_light_indices])

        self.has_mask = True

        # load images and masks
        with ThreadPoolExecutor(max_workers=min(64, os.cpu_count())) as executor:
            def load_img_and_mask(view_light_index):
                view_idx = int(view_light_index.split("V")[1].split("L")[0])
                img_fpath = os.path.join(cfg.data_dir, cfg.img_dirname, f'{view_light_index}.{cfg.img_ext}')
                mask_fpath = os.path.join(cfg.data_dir, cfg.mask_dirname, f'V{view_idx:02d}.{cfg.mask_ext}')
                print("Loading", img_fpath, mask_fpath)
                return self.img_load_fn(img_fpath, cfg.train.img_downscale), self.mask_load_fn(mask_fpath,cfg.train.img_downscale)

            print("Loading images and masks...")
            tic = time.time()
            images_train_list, fg_masks_train_list = zip(*list(executor.map(load_img_and_mask, self.view_light_indices)))
            toc = time.time()
            print("Loading images and masks done.")
            print(f"Time for loading images and masks: {toc - tic:.2f} s")
        self.images_train = np.stack(images_train_list, 0)  # (N, H, W, 3)
        self.fg_masks_train = np.stack(fg_masks_train_list, 0)  # (N, H, W)

        # load camera parameters
        with ThreadPoolExecutor(max_workers=min(64, os.cpu_count())) as executor:
            C2W_train_list, K_train_list = zip(*list(executor.map(
                partial(self.camera_select_fn, cams=self.cams, img_downscale=cfg.train.img_downscale),
                self.view_light_indices)))

        self.C2W_train = np.stack(C2W_train_list, 0)  # (N, 3, 4)
        self.K_train = np.stack(K_train_list, 0)  # (N, 3, 3)
        self.fx = self.K_train[:, 0, 0]  # (N,)
        self.fy = self.K_train[:, 1, 1]
        self.cx = self.K_train[:, 0, 2]
        self.cy = self.K_train[:, 1, 2]
        self.camera_centers_train_world_space = self.C2W_train[:, :3, 3]  # (N, 3)

        self.O2W_scale = self.cams["O2W_scale"]
        self.O2W_translation = self.cams["O2W_translation"]
        self.camera_centers_train_obj_space = (self.camera_centers_train_world_space - self.O2W_translation) / self.O2W_scale

        # load GT normal maps, evaluation purpose only
        if cfg.normal_dirname != "None":
            self.has_gt_normal = True
            self.normal_load_fn = REG.get("fn", cfg.normal_load_fn)

            # Since diligent-mv consists of view-aligned OLAT images.
            # Loading a GT normal map per image is redundant.
            # We instead load a GT normal map per unique view.
            # And prepare a mapping from all view-light indices (i.e., all image indices) to unique view indices.
            # So that we can index into the normal maps correctly for evaluation.
            self.all_image_to_unique_view_mapping = np.array(
                [int(idx.split("V")[1].split("L")[0]) for idx in self.view_light_indices])
            self.unique_train_view_indices = np.unique(self.all_image_to_unique_view_mapping)
            self.unique_train_view_light_indices = []

            for i in self.unique_train_view_indices:
                for j in self.view_light_indices:
                    if int(j.split("V")[1].split("L")[0]) == i:
                        self.unique_train_view_light_indices.append(j)
                        break

            # create a mapping from unique view idx to its index in the unique view list
            self.unique_view_idx_to_unique_view_list_idx = {v: i for i, v in enumerate(self.unique_train_view_indices)}
            self.all_image_to_unique_view_list_idx = np.array(
                [self.unique_view_idx_to_unique_view_list_idx[v] for v in self.all_image_to_unique_view_mapping])

            with ThreadPoolExecutor(max_workers=min(64, os.cpu_count())) as executor:
                def load_normal(view_light_index):
                    view_idx = int(view_light_index.split("V")[1].split("L")[0])
                    normal_fpath = os.path.join(cfg.data_dir, cfg.normal_dirname, f'V{view_idx:02d}.{cfg.normal_ext}')
                    C2W, _ = self.camera_select_fn(view_light_index, self.cams)
                    R_C2W = C2W[:3, :3]
                    normal_cam = self.normal_load_fn(normal_fpath)  # (H, W, 3)
                    normal_cam[..., [1, 2]] *= -1  # change from OpenGL to OpenCV convention
                    normal_world = normal_cam @ R_C2W.T
                    return normal_cam, normal_world

                print("Loading normal maps...")
                normal_maps_val_camera_space, normal_maps_val_world_space = zip(
                    *list(executor.map(load_normal, self.unique_train_view_light_indices)))
                self.normal_maps_val_world_space = np.stack(normal_maps_val_world_space, 0)  # (N_unique_views, H, W, 3)
                self.normal_maps_val_camera_space = np.stack(normal_maps_val_camera_space, 0)  # (N_unique_views, H, W, 3)
                print("Loading normal maps done.")
        else:
            self.has_gt_normal = False

        # load GT light directions and intensities for evaluation.
        if self.config.light_dir_file != "None":
            self.has_gt_light = True
            self.light_load_fn = REG.get("fn", "diligentmv_light")
            self.light_dirs_GT = self.light_load_fn(self.config.light_dir_file)  # (N_L, 3)
            self.light_intensity_GT = self.light_load_fn(self.config.light_int_file)  # (N_L, 3)
        else:
            self.has_gt_light = False
            self.light_dirs_GT = None
            self.light_intensity_GT = None

        self.num_imgs_train = self.C2W_train.shape[0]

        print("Training data loaded!")
        ic(self.images_train.shape,
           self.fg_masks_train.shape,
           self.normal_maps_val_camera_space.shape if self.has_gt_normal else None,
           self.C2W_train.shape,
           self.K_train.shape,
           self.camera_centers_train_obj_space.shape)

    def __iter__(self):
        while True:
            batch_rays_image_index = np.random.randint(0, self.num_imgs_train, size=self.num_rays_per_batch).astype(
                np.int64)

            batch_c2w = self.C2W_train[batch_rays_image_index]  # (B, 3, 4)
            batch_R_c2w = batch_c2w[..., :3, :3]  # (B, 3, 3)
            batch_rays_o = self.camera_centers_train_obj_space[batch_rays_image_index]  # (B, 3)

            # random sample pixel coordinates
            x = np.random.randint(0, self.img_w, size=self.num_rays_per_batch).astype(np.int64)
            y = np.random.randint(0, self.img_h, size=self.num_rays_per_batch).astype(np.int64)

            batch_rgb = self.images_train[batch_rays_image_index, y, x].reshape(-1, 3)
            batch_fg_mask = self.fg_masks_train[batch_rays_image_index, y, x].reshape(-1)
            if self.has_gt_normal:
                batch_normal = self.normal_maps_val_world_space[
                    self.all_image_to_unique_view_list_idx[batch_rays_image_index], y, x].reshape(-1,
                                                                                                 3) if self.has_gt_normal else None

            batch_rgb = batch_rgb * batch_fg_mask[..., None] + (1 - batch_fg_mask[..., None])

            cx = self.cx[batch_rays_image_index]
            cy = self.cy[batch_rays_image_index]
            fx = self.fx[batch_rays_image_index]
            fy = self.fy[batch_rays_image_index]
            batch_directions = np.stack([(x - cx) / fx, (y - cy) / fy, np.ones_like(x)], -1)  # (B, 3)

            batch_rays_d = np.einsum('bij,bj->bi', batch_c2w[..., :3, :3], batch_directions)  # (B, 3)
            batch_rays_d /= np.linalg.norm(batch_rays_d, axis=-1, keepdims=True)
            batch_rays_light_index = self.light_idx[batch_rays_image_index]  # (B,)

            rays = np.concatenate([batch_rays_o,
                                   batch_rays_d,
                                   batch_rays_light_index[..., None],
                                   batch_R_c2w.reshape(-1, 9)
                                   ], -1)  # (B, 3+3+1+9=16)

            batch_data = {
                'rays': rays.astype(np.float32),
                'rays_rgb': batch_rgb.astype(np.float32),
                'rays_fg_mask': batch_fg_mask.astype(np.float32),
            }
            if self.has_gt_normal:
                batch_data.update({'rays_normal_dir_world_space': batch_normal.astype(np.float32)})
            yield batch_data