import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from torch.utils.data import Dataset

from core.registry import REG


@REG.register("dataset", name='mesh_export')
class MeshExportDataset(Dataset):
    def __init__(self, cfg):
        self.config = cfg
        cfg_mesh = cfg.predict_mesh

        self.mask_load_fn = REG.get("fn", cfg.mask_load_fn)
        self.camera_load_fn = REG.get("fn", "load_camera")
        self.camera_select_fn = REG.get("fn", "select_camera")

        self.cams = self.camera_load_fn(cfg.cameras_fpath)

        # ---- resolve size ----
        sample_mask_path = os.path.join(cfg.data_dir, cfg.mask_dirname, cfg.sample_mask_fname)
        self.img_h0, self.img_w0 = self.mask_load_fn(sample_mask_path).shape[:2]
        self.img_h = int(self.img_h0 / cfg_mesh.img_downscale)
        self.img_w = int(self.img_w0 / cfg_mesh.img_downscale)

        with open(cfg_mesh.view_light_index_file, 'r') as f:
            self.view_light_indices = f.read().splitlines()

        self.unique_view_indices = np.unique(
            [int(idx.split("V")[1].split("L")[0]) for idx in self.view_light_indices])

        # select the view-light indices with unique view indices
        self.unique_view_light_indices = []
        for i in self.unique_view_indices:
            for j  in self.view_light_indices:
                if int(j.split("V")[1].split("L")[0]) == i:
                    self.unique_view_light_indices.append(j)
                    break

        with ThreadPoolExecutor(max_workers=min(64, os.cpu_count())) as executor:
            C2W_mesh, K_mesh = zip(*list(executor.map(partial(self.camera_select_fn, cams=self.cams,
                    img_downscale=cfg_mesh.img_downscale), self.unique_view_light_indices)))

        self.C2W_mesh = np.stack(C2W_mesh, 0)
        self.K_mesh = np.stack(K_mesh, 0)
        self.O2W_scale = self.cams["O2W_scale"]
        self.O2W_translation = np.array(self.cams["O2W_translation"])

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return {
            "C2W": self.C2W_mesh,
            "K": self.K_mesh,
            "img_h": self.img_h,
            "img_w": self.img_w,
        }