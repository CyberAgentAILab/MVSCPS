import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from icecream import ic
from nerfacc import ContractionType
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from core.registry import REG
from models.base import BaseModel
from models.network_utils import get_encoding, get_mlp
from models.utils import chunk_batch, cleanup, get_activation, scale_anything
from systems.utils import update_module_step
from utils.misc import get_rank


class VarianceNetwork(nn.Module):
    def __init__(self, config):
        super(VarianceNetwork, self).__init__()
        self.config = config
        self.init_val = self.config.init_val
        self.register_parameter('variance', nn.Parameter(torch.tensor(self.config.init_val)))
        self.modulate = self.config.get('modulate', False)
        if self.modulate:
            self.mod_start_steps = self.config.mod_start_steps
            self.reach_max_steps = self.config.reach_max_steps
            self.max_inv_s = self.config.max_inv_s

    @property
    def inv_s(self):
        val = torch.exp(self.variance * 10.0)
        if self.modulate and self.do_mod:
            val = val.clamp_max(self.mod_val)
        return val

    def forward(self, x):
        return torch.ones([len(x), 1], device=self.variance.device) * self.inv_s

    def update_step(self, epoch, global_step):
        if self.modulate:
            self.do_mod = global_step > self.mod_start_steps
            if not self.do_mod:
                self.prev_inv_s = self.inv_s.item()
            else:
                self.mod_val = min(
                    (global_step / self.reach_max_steps) * (self.max_inv_s - self.prev_inv_s) + self.prev_inv_s,
                    self.max_inv_s)


def contract_to_unisphere(x, radius, contraction_type):
    if contraction_type == ContractionType.AABB:
        x = scale_anything(x, (-radius, radius), (0, 1))
    elif contraction_type == ContractionType.UN_BOUNDED_SPHERE:
        x = scale_anything(x, (-radius, radius), (0, 1))
        x = x * 2 - 1  # aabb is at [-1, 1]
        mag = x.norm(dim=-1, keepdim=True)
        mask = mag.squeeze(-1) > 1
        x[mask] = (2 - 1 / mag[mask]) * (x[mask] / mag[mask])
        x = x / 4 + 0.5  # [-inf, inf] is at [0, 1]
    else:
        raise NotImplementedError
    return x


def contract_to_unit_bbox(uv, img_wh):
    # u: x axis, v: y axis
    uv_norm = 2 * uv / torch.tensor(img_wh, device=uv.device) - 1
    return uv_norm


class MarchingCubeHelper(nn.Module):
    def __init__(self, use_torch=True):
        super().__init__()
        self.use_torch = use_torch
        self.points_range = (0, 1)
        if self.use_torch:
            import torchmcubes
            self.mc_func = torchmcubes.marching_cubes
        else:
            import mcubes
            self.mc_func = mcubes.marching_cubes
        self.verts = None

    def grid_vertices(self, resolution=256):
        x, y, z = torch.linspace(*self.points_range, resolution), torch.linspace(*self.points_range, resolution), torch.linspace(*self.points_range, resolution)
        x, y, z = torch.meshgrid(x, y, z, indexing='ij')
        verts = torch.cat([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)], dim=-1).reshape(-1, 3)

        return verts

    def forward(self, level, threshold=0., resolution=256):
        if type(level) is not torch.Tensor:
            level = torch.cat(level, dim=0)
        ic(level.shape)
        level = level.float().view(resolution, resolution, resolution)
        if self.use_torch:
            verts, faces = self.mc_func(level.to(get_rank()), threshold)
            verts, faces = verts.cpu(), faces.cpu().long()
        else:
            verts, faces = self.mc_func(-level.numpy(), threshold) # transform to numpy
            verts, faces = torch.from_numpy(verts.astype(np.float32)), torch.from_numpy(faces.astype(np.int64)) # transform back to pytorch
        verts = verts / (resolution - 1.)
        return {
            'v_pos': verts,
            't_pos_idx': faces
        }


class BaseImplicitGeometry(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        if self.config.isosurface is not None:
            assert self.config.isosurface.method in ['mc', 'mc-torch']
            if self.config.isosurface.method == 'mc-torch':
                raise NotImplementedError("Please do not use mc-torch. It currently has some scaling issues I haven't fixed yet.")
            self.helper = MarchingCubeHelper(use_torch=self.config.isosurface.method=='mc-torch')
        self.radius = self.config.get('radius', 1.0)
        img_rootdir = self.config.get('img_rootdir', None)
        if img_rootdir is not None:
            img_dir = self.config.get('img_dirname', None)
            img_path = os.path.join(img_rootdir, img_dir, "000000.png")
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            self.img_wh = img.shape[:2][::-1]
            ic(self.img_wh)
        self.contraction_type = None # assigned in system

    def forward_level(self, points, with_grad):
        raise NotImplementedError

    def isosurface_(self, vmin, vmax, resolution):
        # ic(vmin, vmax)
        def batch_func(x, ph=None):
            # ic(x.shape)
            x = torch.stack([
                scale_anything(x[...,0], (0, 1), (vmin[0], vmax[0])),
                scale_anything(x[...,1], (0, 1), (vmin[1], vmax[1])),
                scale_anything(x[...,2], (0, 1), (vmin[2], vmax[2])),
            ], dim=-1).to(self.rank)
            rv = self.forward_level(x).cpu()
            cleanup()
            return rv
        ic(resolution)
        level = chunk_batch(batch_func, self.config.isosurface.chunk, True, self.helper.grid_vertices(resolution=resolution))
        mesh = self.helper(level, threshold=self.config.isosurface.threshold, resolution=resolution)
        mesh['v_pos'] = torch.stack([
            scale_anything(mesh['v_pos'][...,0], (0, 1), (vmin[0], vmax[0])),
            scale_anything(mesh['v_pos'][...,1], (0, 1), (vmin[1], vmax[1])),
            scale_anything(mesh['v_pos'][...,2], (0, 1), (vmin[2], vmax[2]))
        ], dim=-1)
        return mesh

    def slice_plane(self, axis="x", level=0, resolution=512):
        if axis == "x":
            x = level * torch.ones(resolution**2)
            y, z = torch.meshgrid(torch.linspace(-1, 1, resolution), torch.linspace(-1, 1, resolution))
            y, z = y.flatten(), z.flatten()

        elif axis == "y":
            y = level * torch.ones(resolution**2)
            x, z = torch.meshgrid(torch.linspace(-1, 1, resolution), torch.linspace(-1, 1, resolution))
            x, z = x.flatten(), z.flatten()

        elif axis == "z":
            z = level * torch.ones(resolution**2)
            x, y = torch.meshgrid(torch.linspace(-1, 1, resolution), torch.linspace(-1, 1, resolution))
            x, y = x.flatten(), y.flatten()

        points = torch.stack([x, y, z], dim=-1)
        # move to device
        points = points.to(self.rank)
        level = self.forward_level(points).cpu()
        return level.reshape(resolution, resolution)

    @torch.no_grad()
    def isosurface(self):
        if self.config.isosurface is None:
            raise NotImplementedError
        mesh_coarse = self.isosurface_((-self.radius, -self.radius, -self.radius), (self.radius, self.radius, self.radius), 128)

        vmin, vmax = mesh_coarse['v_pos'].amin(dim=0), mesh_coarse['v_pos'].amax(dim=0)
        vmin_ = (vmin - (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)
        vmax_ = (vmax + (vmax - vmin) * 0.1).clamp(-self.radius, self.radius)

        mesh_fine = self.isosurface_(vmin_, vmax_, self.config.isosurface.resolution)
        return mesh_fine

@REG.register('model', name='neural_sdf')
class VolumeSDF(BaseImplicitGeometry):
    def setup(self):
        self.n_output_dims = self.config.feature_dim
        self.encoding = get_encoding(3, self.config.xyz_encoding_config)
        self.network = get_mlp(self.encoding.n_output_dims, self.n_output_dims, self.config.mlp_network_config)
        self.grad_type = self.config.grad_type
        self.finite_difference_eps = self.config.get('finite_difference_eps', 1e-3)
        if self.grad_type == 'finite_difference':
            rank_zero_info(f"Using finite difference to compute gradients with eps={self.finite_difference_eps}")

    def forward(self, points, with_grad=True, with_feature=True, with_laplace=False):
        with torch.inference_mode(torch.is_inference_mode_enabled() and not (with_grad and self.grad_type == 'analytic')):
            with torch.set_grad_enabled(self.training or (with_grad and self.grad_type == 'analytic')):
                if with_grad and self.grad_type == 'analytic':
                    if not self.training:
                        points = points.clone() # points may be in inference mode, get a copy to enable grad
                    points.requires_grad_(True)

                points_ = points # points in the original scale
                points = contract_to_unisphere(points, self.radius, self.contraction_type) # points normalized to (0, 1)

                point_encoding = self.encoding(points.view(-1, 3))
                out = self.network(point_encoding).view(*points.shape[:-1], self.n_output_dims).float()
                sdf = out[..., 0]
                feature = out[..., 1:]

                if 'sdf_activation' in self.config:
                    sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
                if 'feature_activation' in self.config:
                    feature = get_activation(self.config.feature_activation)(feature)
                if with_grad:
                    if self.grad_type == 'analytic':
                        grad = torch.autograd.grad(
                            sdf, points_, grad_outputs=torch.ones_like(sdf),
                            create_graph=True, retain_graph=True, only_inputs=True)[0]
                    elif self.grad_type == 'finite_difference':
                        eps = self._finite_difference_eps
                        offsets = torch.as_tensor(
                            [
                                [eps, 0.0, 0.0],
                                [-eps, 0.0, 0.0],
                                [0.0, eps, 0.0],
                                [0.0, -eps, 0.0],
                                [0.0, 0.0, eps],
                                [0.0, 0.0, -eps],
                            ]
                        ).to(points_)
                        points_d_ = (points_[...,None,:] + offsets).clamp(-self.radius, self.radius)
                        points_d = scale_anything(points_d_, (-self.radius, self.radius), (0, 1))
                        points_d_sdf = self.network(self.encoding(points_d.view(-1, 3)))[...,0].view(*points.shape[:-1], 6).float()
                        grad = 0.5 * (points_d_sdf[..., 0::2] - points_d_sdf[..., 1::2]) / eps  

                        if with_laplace:
                            laplace = (points_d_sdf[..., 0::2] + points_d_sdf[..., 1::2] - 2 * sdf[..., None]).sum(-1) / (eps ** 2)

        rv = [sdf]
        if with_grad:
            rv.append(grad)
        if with_feature:
            rv.append(feature)
        if with_laplace:
            assert self.config.grad_type == 'finite_difference', "Laplace computation is only supported with grad_type='finite_difference'"
            rv.append(laplace)
        rv = [v if self.training else v.detach() for v in rv]
        return rv[0] if len(rv) == 1 else rv

    def forward_level(self, points, with_grad=False):
        points = contract_to_unisphere(points, self.radius, self.contraction_type) # points normalized to (0, 1)
        if with_grad:
            points = points.clone()  # points may be in inference mode, get a copy to enable grad
            points.requires_grad_(True)
        sdf = self.network(self.encoding(points.view(-1, 3)))[...,0]
        if with_grad:
            sdf = sdf.view(*points.shape[:-1])
            grad = torch.autograd.grad(
                sdf, points, grad_outputs=torch.ones_like(sdf),
                create_graph=True, retain_graph=True, only_inputs=True
            )[0]
        if 'sdf_activation' in self.config:
            sdf = get_activation(self.config.sdf_activation)(sdf + float(self.config.sdf_bias))
        if with_grad:
            return sdf, grad
        else:
            return sdf

    def update_step(self, epoch, global_step):
        update_module_step(self.encoding, epoch, global_step)    
        update_module_step(self.network, epoch, global_step)  
        if self.grad_type == 'finite_difference':
            if isinstance(self.finite_difference_eps, float):
                self._finite_difference_eps = self.finite_difference_eps
            elif self.finite_difference_eps == 'progressive':
                hg_conf = self.config.xyz_encoding_config
                assert hg_conf.otype == "ProgressiveBandHashGrid", "finite_difference_eps='progressive' only works with ProgressiveBandHashGrid"
                current_level = min(
                    hg_conf.start_level + max(global_step - hg_conf.start_step, 0) // hg_conf.update_steps,
                    hg_conf.n_levels
                )
                grid_res = hg_conf.base_resolution * hg_conf.per_level_scale**(current_level - 1)
                grid_size = 2 * self.config.radius / grid_res
                if grid_size != self._finite_difference_eps:
                    rank_zero_info(f"Update finite_difference_eps to {grid_size}")
                self._finite_difference_eps = grid_size
            else:
                raise ValueError(f"Unknown finite_difference_eps={self.finite_difference_eps}")