import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn


class LightingParameters(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Read light indices used during training
        with open(config.view_light_index_fname_train, 'r') as f:
            view_light_indices = f.read().splitlines()
        self.train_light_indices = np.unique([int(idx.split("L")[-1]) for idx in view_light_indices])
        self.train_num_lights = len(self.train_light_indices)

        # Initialize light directions
        if config.light_dir_file != "None":
            # Load calibrated light directions (e.g., for DiLiGenT-MV dataset)
            light_dirs = np.loadtxt(config.light_dir_file)  # (N_L, 3)
            # Convert from OpenGL to OpenCV coordinate system
            light_dirs[..., [1, 2]] = -light_dirs[..., [1, 2]]
            if not config.use_gt_light:
                # Use initial light directions from config for training lights
                light_dirs[self.train_light_indices] = np.array(config.init_light_dir)
        else:
            # If no light direction file is provided, initialize with the same direction for all lights
            light_dirs = np.tile(np.array(config.init_light_dir), (self.train_num_lights, 1))

        # Initialize light intensities
        if config.light_int_file != "None":
            light_intensity = np.loadtxt(config.light_int_file)  # (N_L, 3)
            if not config.use_gt_light:
                light_intensity[self.train_light_indices] = np.array(config.init_intensity)
        else:
            light_intensity = np.ones((self.train_num_lights, 3)) * np.array(config.init_intensity)

        self.light_dir = nn.Parameter(torch.tensor(light_dirs, dtype=torch.float32), requires_grad=True)  # (N_L, 3)
        self.intensity = nn.Parameter(torch.tensor(light_intensity, dtype=torch.float32), requires_grad=True) # (N_L, 3)

    def forward(self, rays_light_indices):
        """
        Args:
            rays_light_indices: (N_rays,) LongTensor
            rays_view_indices: (N_rays,) LongTensor
        Returns:
            rays_light_intensity: (N_rays, 3) FloatTensor
            rays_light_dir: (N_rays, 3) FloatTensor
        """

        rays_light_intensity = self.intensity[rays_light_indices]
        rays_light_dir_camera_space = Fn.normalize(self.light_dir, p=2, dim=-1)[rays_light_indices]

        return rays_light_intensity, rays_light_dir_camera_space


