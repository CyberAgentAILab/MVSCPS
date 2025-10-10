import numpy as np
from torch.utils.data import Dataset

from core.registry import REG
from dataloader.utils import (generate_ring_lights,
                              generate_rotation_camera_poses,
                              get_camera_space_ray_directions,
                              get_world_space_ray_directions)


@REG.register("dataset", name='relighting')
class DiligentPredictDataset(Dataset):
    def __init__(self, cfg):
        cfg_relighting = cfg.predict_relighting
        self.cfg = cfg
        interval = cfg_relighting.camera_frames // 3

        light_directions = generate_ring_lights(cfg_relighting.light_frames, elevation=cfg_relighting.light_elevation)
        c2ws = generate_rotation_camera_poses(cfg_relighting.camera_frames,
                                              radius=cfg_relighting.camera_distance,
                                              lookat=cfg_relighting.camera_lookat,
                                              elevation=cfg_relighting.camera_elevation)

        # sequence of novel views and novel light directions
        first_row_repeated = np.repeat(light_directions[0:1], repeats=interval, axis=0)
        self.light_directions_predict = np.concatenate([light_directions,
                                           first_row_repeated,
                                           light_directions,
                                           first_row_repeated,
                                           light_directions,
                                           first_row_repeated,
                                           ], axis=0)
        self.c2w_predict = np.concatenate([
            np.repeat(c2ws[0:1], cfg_relighting.light_frames, axis=0),
            c2ws[:interval],
            np.repeat(c2ws[interval:interval + 1], cfg_relighting.light_frames, axis=0),
            c2ws[interval:interval * 2],
            np.repeat(c2ws[interval * 2:interval * 2 + 1], cfg_relighting.light_frames, axis=0),
            c2ws[interval * 2:],
        ], axis=0)
        self.camera_centers_obj_space = self.c2w_predict[:, :3, 3]  # (N, 3)

        self.img_w, self.img_h = cfg_relighting.img_width, cfg_relighting.img_height
        cx = self.img_w // 2
        cy = self.img_h // 2
        fx = fy = cfg_relighting.camera_focal_length_to_img_width_ratio * self.img_w
        K = np.array([[fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]])

        self.directions = get_camera_space_ray_directions(self.img_h, self.img_w, K, return_flat=True)  # (H*W, 3)

    def __len__(self):
        # the total number of images to be rendered
        return self.c2w_predict.shape[0]

    def __getitem__(self, idx):
        c2w = self.c2w_predict[idx]  # (3, 4)
        rays_d = get_world_space_ray_directions(self.directions, c2w)  # (H*W, 3)
        rays_o = self.camera_centers_obj_space[idx]
        rays_o = np.repeat(rays_o[None], repeats=rays_d.shape[0], axis=0)  # (H*W, 3)
        rays = np.concatenate([rays_o, rays_d], -1)  # (H*W, 6)

        light_dir = self.light_directions_predict[idx]  # (3, )
        light_dir_world_space = (c2w[:3, :3] @ light_dir[:, None])[:, 0]  # (3,)
        light_dir_world_space = light_dir_world_space / np.linalg.norm(light_dir_world_space)

        return {
            'rays': rays.astype(np.float32),
            'single_light_dir_world_space': light_dir_world_space.astype(np.float32),
        }