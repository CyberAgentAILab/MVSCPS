import torch
import torch.nn as nn
import torch.nn.functional as Fn
from nerfacc import (ContractionType, OccupancyGrid, accumulate_along_rays,
                     ray_marching, render_weight_from_alpha)
from tqdm import tqdm

from core.registry import REG
from models.geometry import VarianceNetwork
from models.lighting import LightingParameters
from models.utils import chunk_batch
from systems.utils import update_module_step
from utils.misc import get_rank


@REG.register('model', name='mvscps')
class NeuSModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = get_rank()
        self.setup()
        if self.config.get('weights', None):
            self.load_state_dict(torch.load(self.config.weights))

    def setup(self):
        self.geometry = REG.build('model', self.config.geometry)
        self.brdf = REG.build('model', self.config.brdf)
        self.lighting = LightingParameters(self.config.light)
        self.shadow_mapping = REG.build('model', self.config.shadow)
        self.use_shadow_mlp = self.config.get('use_shadow_mlp', True)
        self.geometry.contraction_type = ContractionType.AABB

        self.variance = VarianceNetwork(self.config.variance)
        self.register_buffer('scene_aabb', torch.as_tensor([-self.config.radius, -self.config.radius, -self.config.radius, self.config.radius, self.config.radius, self.config.radius], dtype=torch.float32))
        if self.config.grid_prune:
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=128,
                contraction_type=ContractionType.AABB
            )

        self.randomized = self.config.randomized
        self.background_color = torch.ones((3,), dtype=torch.float32, device=self.rank)
        self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray
    
    def update_step(self, epoch, global_step):
        update_module_step(self.geometry, epoch, global_step)
        update_module_step(self.variance, epoch, global_step)

        cos_anneal_end = self.config.get('cos_anneal_end', 0)
        self.cos_anneal_ratio = 1.0 if cos_anneal_end == 0 else min(1.0, global_step / cos_anneal_end)

        def occ_eval_fn(x):
            sdf = self.geometry(x, with_grad=False, with_feature=False)
            alpha = torch.sigmoid(- sdf * 80)
            return alpha

        if self.training and self.config.grid_prune:
            self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn, occ_thre=self.config.get('grid_prune_occ_thre', 0.1))

    def isosurface(self):
        mesh = self.geometry.isosurface()
        return mesh

    def get_alpha(self, sdf, normal, dirs, dists):
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(Fn.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     Fn.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[...,None] + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf[...,None] - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
        return alpha

    def get_shadow_ray_alpha(self, sdf, normal, dirs, dists):
        inv_s = self.variance(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter

        true_cos = (dirs * normal).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(Fn.relu(-true_cos * 0.5 + 0.5) * (1.0 - self.cos_anneal_ratio) +
                     Fn.relu(-true_cos) * self.cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        estimated_next_sdf = sdf[...,None] + iter_cos * dists[...,None] * 0.5
        estimated_prev_sdf = sdf[...,None] - iter_cos * dists[...,None] * 0.5

        # ic(estimated_prev_sdf.shape, inv_s.shape)
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).clip(0.0, 1.0)
        return alpha

    def shadow_render_volume_render(self, surface_points, surface_point_to_light_dir, n_samples=128, lnear=1e-2, lfar=0.5):
        # ic(surface_points.shape, surface_point_to_light_dir.shape)
        # sample points uniformly
        t = torch.linspace(lnear, lfar, n_samples, device=surface_points.device, dtype=surface_points.dtype)

        # sample points from surface points towards the point light
        sample_points = surface_points[:, None, :] + t[None, :, None] * surface_point_to_light_dir[:, None, :]

        # reverse the sample points along the second axis
        sample_points = sample_points.flip(1)

        # query opacity values
        shadow_ray_sdf, shadow_ray_sdf_grad, _ = self.geometry(sample_points, with_grad=True, with_feature=True)
        shadow_ray_normal_world_space = Fn.normalize(shadow_ray_sdf_grad, p=2, dim=-1)

        dists = torch.ones_like(t) * (lfar - lnear) / n_samples

        # Both camera space or world space are fine, since only their inner product is used
        shadow_ray_alpha = self.get_shadow_ray_alpha(shadow_ray_sdf, shadow_ray_normal_world_space, surface_point_to_light_dir[:, None, :],dists)[..., None]
        shadow_ray_alpha = shadow_ray_alpha.squeeze()
        n_rays = shadow_ray_alpha.shape[0]
        shadow_ray_weights = shadow_ray_alpha * torch.cumprod(
            torch.cat([torch.ones([n_rays, 1], device=shadow_ray_alpha.device), 1. - shadow_ray_alpha + 1e-7], -1), -1)[:, :-1]
        shadow_map = 1 - shadow_ray_weights.sum(dim=-1, keepdim=True)
        return shadow_map

    def forward_geometry(self, rays_o, rays_d):
        n_rays = rays_o.shape[0]

        with torch.no_grad():
            ray_indices, t_starts, t_ends = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid if self.config.grid_prune else None,
                alpha_fn=None,
                near_plane=None, far_plane=None,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=0.0,
            )

        ray_indices = ray_indices.long()
        t_origins = rays_o[ray_indices]
        t_dirs_world_space = rays_d[ray_indices]
        midpoints = (t_starts + t_ends) / 2.
        positions = t_origins + t_dirs_world_space * midpoints
        dists = t_ends - t_starts

        sdf, sdf_grad, spatial_feature = self.geometry(positions, with_grad=True, with_feature=True)
        normal_world_space = Fn.normalize(sdf_grad, p=2, dim=-1)

        alpha = self.get_alpha(sdf, normal_world_space, t_dirs_world_space, dists)  # Both camera space or world space are fine, since only their inner product is used

        weights = render_weight_from_alpha(alpha, ray_indices=ray_indices, n_rays=n_rays)
        opacity = accumulate_along_rays(weights, ray_indices, values=None, n_rays=n_rays)
        depth = accumulate_along_rays(weights, ray_indices, values=midpoints, n_rays=n_rays)
        num_samples_per_ray = accumulate_along_rays(
            torch.ones_like(weights, dtype=weights.dtype, device=weights.device),
            ray_indices, values=torch.ones_like(weights, dtype=weights.dtype, device=weights.device), n_rays=n_rays)

        if positions.shape[0] == 0: # no sample points
            surface_points = torch.zeros_like(rays_d)
        else:
            surface_points = depth * rays_d + rays_o

        rays_normal_world_space = accumulate_along_rays(weights, ray_indices, values=normal_world_space, n_rays=n_rays)

        geometry_out = {
            "samples_signed_distance": sdf,
            "samples_sdf_grad": sdf_grad, # for Eikonal loss
            "samples_normal_dir_world_space": normal_world_space,
            "samples_view_dir_world_space": t_dirs_world_space,
            "samples_brdf_latent": spatial_feature,
            "samples_weights": weights,
            "samples_positions": positions,

            "rays_opacity": opacity,  # for mask loss
            "rays_depth": depth,
            "rays_surface_points": surface_points,
            "rays_normal_dir_world_space": rays_normal_world_space,
            'rays_valid': opacity > 0,

            "rays_to_samples_indices": ray_indices,
            'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays_o.device),
            "num_samples_per_ray": num_samples_per_ray,
        }
        return geometry_out


    def render_shading(self, geometry_out, rays_light_dir_world_space):
        n_rays = rays_light_dir_world_space.shape[0]

        samples_weights = geometry_out["samples_weights"]
        samples_normal_dir_world_space = geometry_out["samples_normal_dir_world_space"]
        rays_to_samples_indices = geometry_out["rays_to_samples_indices"]
        samples_brdf_latent = geometry_out["samples_brdf_latent"]

        # reverse the view direction to point towards the camera
        samples_view_dir_world_space = -geometry_out["samples_view_dir_world_space"]

        samples_light_dir_world_space = rays_light_dir_world_space[rays_to_samples_indices]
        samples_brdf_values = self.brdf(samples_brdf_latent,
                                        samples_normal_dir_world_space,
                                        samples_view_dir_world_space,
                                        samples_light_dir_world_space)  # (N_samples, 3)

        samples_NoL = torch.sum(samples_normal_dir_world_space * samples_light_dir_world_space, dim=-1,keepdim=True)
        samples_shading_values = samples_brdf_values * Fn.softplus(samples_NoL)  # (N_samples, 3)
        samples_cosine_term = torch.max(samples_NoL, torch.zeros_like(samples_NoL) + 1e-7)  # Only for visualization.

        rays_shading = accumulate_along_rays(samples_weights,
                                              rays_to_samples_indices,
                                              values=samples_shading_values,
                                              n_rays=n_rays)
        rays_brdf_values = accumulate_along_rays(samples_weights, rays_to_samples_indices, values=samples_brdf_values, n_rays=n_rays)
        rays_NoL = accumulate_along_rays(samples_weights, rays_to_samples_indices, values=samples_cosine_term, n_rays=n_rays)
        shading_out = {
            'rays_brdf': rays_brdf_values,  # without light intensity and without shadow
            'rays_NoL': rays_NoL,
            'rays_shading': rays_shading,
        }
        return shading_out


    def render_shadow(self, geometry_out, rays_light_dir_world_space, rays_view_dir_world_space):
        rays_surface_points = geometry_out["rays_surface_points"]
        rays_shadow_rendered = self.shadow_render_volume_render(rays_surface_points,
                                                                rays_light_dir_world_space,
                                                                n_samples=64,
                                                                lnear=0.01,
                                                                lfar=0.5)

        shadow_out = {
            "rays_shadow_rendered": rays_shadow_rendered
        }

        _, surface_features = self.geometry(rays_surface_points, with_grad=False, with_feature=True)
        rays_shadow_refined = self.shadow_mapping(surface_features.detach(), rays_shadow_rendered, -rays_view_dir_world_space)
        shadow_out["rays_shadow_refined"] = rays_shadow_refined

        return shadow_out

    def forward(self, rays, single_light_dir_world_space=None):
        n_rays = rays.shape[0]
        rays_origin, rays_dir_world_space = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        # render geometry-related information
        geometry_out = self.forward_geometry(rays_origin, rays_dir_world_space)

        # get lighting.py information, both in shape (N_rays, 3)
        if single_light_dir_world_space is None:
            # for train/validation, rays_light_index is provided
            # light directions and intensities are indexed from LightingParameters
            rays_light_index = rays[:, 6].long()
            rays_R_c2w = rays[:, 7:16].reshape(-1, 3, 3)
            rays_light_intensity, rays_light_dir_camera_space = self.lighting(rays_light_index)
            rays_light_dir_world_space = (rays_R_c2w @ rays_light_dir_camera_space[..., None])[..., 0]  # (n_rays, 3)
        else:
            # for relighting, rays_light_dir_world_space is provided
            rays_light_intensity = torch.ones((n_rays, 3), device=rays.device)
            rays_light_dir_world_space = single_light_dir_world_space.expand(n_rays, 3)

        # render shading
        shading_out = self.render_shading(geometry_out, rays_light_dir_world_space)

        # render shadow
        shadow_out = self.render_shadow(geometry_out, rays_light_dir_world_space, rays_dir_world_space)

        # combine shading and shadow
        rays_rgb_wo_shadow = shading_out['rays_shading'] * rays_light_intensity
        rays_rgb_w_shadow = rays_rgb_wo_shadow * shadow_out["rays_shadow_refined"]
        rays_rgb_final = rays_rgb_w_shadow + (1.0 - geometry_out['rays_opacity'])

        if single_light_dir_world_space is None:
            # train/validation
            render_out = {
                'samples_sdf_grad': geometry_out['samples_sdf_grad'], # for Eikonal loss
                'rays_opacity': geometry_out['rays_opacity'], # for mask loss
                'rays_rgb': rays_rgb_final, # for color loss

                'rays_rgb_wo_shadow': rays_rgb_wo_shadow,
                'rays_rgb_w_shadow': rays_rgb_w_shadow,
                'rays_depth': geometry_out['rays_depth'],
                'rays_valid': geometry_out['rays_valid'],
                "rays_normal_dir_world_space": geometry_out['rays_normal_dir_world_space'],
                'num_samples': geometry_out['num_samples'],
                "num_samples_per_ray": geometry_out['num_samples_per_ray'],
                **shadow_out
            }
        else:
            # relighting
            render_out = {
                'rays_opacity': geometry_out['rays_opacity'],
                'rays_rgb': rays_rgb_final,
                'rays_rgb_wo_shadow': rays_rgb_wo_shadow,
                "rays_normal_dir_world_space": geometry_out['rays_normal_dir_world_space'],}
        return render_out


    def forward_brdf(self, rays, brdf_sphere_normal, brdf_sphere_fg_mask, light_dir):
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
        geometry_out = self.forward_geometry(rays_o, rays_d)

        rays_valid = geometry_out['rays_valid'][..., 0]
        if not torch.any(rays_valid):
            return

        surface_points = geometry_out["rays_surface_points"]  # (N_rays, 3)
        _, surface_grad, surface_features = self.geometry(surface_points, with_grad=True, with_feature=True)  # (N_rays, 3), (N_rays, 3), (N_rays, 3)
        surface_normal = Fn.normalize(surface_grad, p=2, dim=-1)

        n_brdf_samples = brdf_sphere_fg_mask.sum()
        brdf_sphere_fg_mask_flat = brdf_sphere_fg_mask.reshape(-1).bool()
        brdf_sphere_normal_flat = brdf_sphere_normal.reshape(-1, 3)
        normal_camera_sphere_input = brdf_sphere_normal_flat[brdf_sphere_fg_mask_flat] # (N_samples, 3)
        normal_sphere_res = brdf_sphere_normal.shape[0]

        brdf_sphere_list = []
        for i in tqdm(range(rays_valid.sum())):
            surface_point_feature = surface_features[rays_valid][i] # (3, )

            # repeat the surface point and feature to the number of brdf samples
            surface_point_feature = surface_point_feature.expand(n_brdf_samples, -1)
            t_dirs_brdf = torch.tensor([0., 0., -1.], device=surface_points.device).expand(n_brdf_samples, 3)
            light_dir_brdf = light_dir.expand(n_brdf_samples, 3)
            light_dir_brdf = Fn.normalize(light_dir_brdf, p=2, dim=-1)

            # all vectors in camera space, since angular encoding is used
            # ic(surface_point_feature.shape, normal_camera_sphere_input.shape, t_dirs_brdf.shape, light_dir_brdf.shape)
            brdf_values = self.brdf(surface_point_feature, normal_camera_sphere_input, t_dirs_brdf, light_dir_brdf)  # (N_samples, 3)
            brdf_values = brdf_values/ brdf_values.max()
            brdf_values= torch.pow(brdf_values, 1/2.2)

            # save the brdf values as image
            brdf_sphere = torch.zeros((normal_sphere_res * normal_sphere_res, 3)).to(brdf_values.device)
            brdf_sphere[brdf_sphere_fg_mask_flat] = brdf_values
            brdf_sphere = brdf_sphere.reshape(normal_sphere_res, normal_sphere_res, 3)
            brdf_sphere_list.append(brdf_sphere)

        brdf_sphere_all = torch.stack(brdf_sphere_list, dim=0)  # (N_rays_valid, H, W, 3)
        out = {"rays_surface_latent": surface_features,
               "rays_brdf_sphere": brdf_sphere_all,
               "rays_surface_normal": surface_normal,
               }
        return out

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()
    
    def regularizations(self, out):
        losses = {}
        losses.update(self.geometry.regularizations(out))
        losses.update(self.brdf.regularizations(out))
        return losses

    @torch.no_grad()
    def export(self, export_config):
        mesh = self.isosurface()
        if export_config.export_spatial_latent:
            import umap
            from sklearn.preprocessing import MinMaxScaler

            _, _, feature  = chunk_batch(self.geometry, export_config.chunk_size, False, mesh['v_pos'].to(self.rank), with_grad=True, with_feature=True, with_albedo=False)

            spatial_latent = feature[0].detach().cpu().numpy()

            reducer = umap.UMAP(n_components=3, metric='cosine')
            spatial_latent_3d = reducer.fit_transform(spatial_latent)

            scaler = MinMaxScaler()
            spatial_latent_3d_colors = scaler.fit_transform(spatial_latent_3d)
            mesh['v_rgb'] = torch.from_numpy(spatial_latent_3d_colors).float()
        return mesh

    @torch.no_grad()
    def slice(self, ):
        sdf_plane = self.geometry.slice_plane()

        return sdf_plane