import os

import cv2
import numpy as np
import pyvista as pv
import torch
import torch.nn.functional as F

pv.set_plot_theme('document')
pv.start_xvfb()
import igl
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import umap
from sklearn.preprocessing import MinMaxScaler
## evaluation metrics
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import \
    LearnedPerceptualImagePatchSimilarity as LPIPS
from tqdm import tqdm

from core.registry import REG
from models.utils import chunk_batch
from systems.criterions import (MAE, binary_cross_entropy,
                                chamfer_distance_and_f1_score,
                                scale_invariant_mse)
from systems.utils import parse_optimizer, parse_scheduler, update_module_step
from utils.light_plot_utils import plot_light
from utils.misc import config_to_primitive, get_rank
from utils.mixins import SaverMixin


def rays_to_image(rays_values, fg_mask):
    img_h, img_w = fg_mask.shape[0], fg_mask.shape[1]
    img = torch.zeros((img_h, img_w, rays_values.shape[-1]), device=fg_mask.device)
    img[fg_mask] = rays_values.to(fg_mask.device)
    return img

def set_image_background_black(x, mask):
    x[~mask] = 0.
    return x

gamma_correct = lambda x: x.pow(1/2.2)

@REG.register('system', name="mvscps")
class MvscpsSystem(pl.LightningModule, SaverMixin):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = get_rank()
        self.prepare()
        self.model = REG.build('model', self.config.model)

        self._predict_names = self.config.dataset.predict_targets
        self._predict_handlers = dict()
        for fn_id, fn_name in enumerate(self._predict_names):
            self._predict_handlers[fn_id] = getattr(self, self.config.dataset[fn_name].predict_step_fn_name)

        self.trained_global_step: int | None = None

    def configure_optimizers(self):
        optim = parse_optimizer(self.config.system.optimizer, self.model)
        ret = {
            'optimizer': optim,
        }

        if 'scheduler' in self.config.system:
            ret.update({
                'lr_scheduler': parse_scheduler(self.config.system.scheduler, optim),
            })
        return ret

    def C(self, value):
        if isinstance(value, int) or isinstance(value, float):
            pass
        else:
            value = config_to_primitive(value)
            if not isinstance(value, list):
                raise TypeError('Scalar specification only supports list, got', type(value))
            if len(value) == 3:
                value = [0] + value
            assert len(value) == 4
            start_step, start_value, end_value, end_step = value
            if isinstance(end_step, int):
                current_step = self.global_step
                value = start_value + (end_value - start_value) * max(
                    min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
            elif isinstance(end_step, float):
                current_step = self.current_epoch
                value = start_value + (end_value - start_value) * max(
                    min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
        return value

    def prepare(self):
        self.criterions = {
            'psnr': PSNR(data_range=1.).to(self.rank),
            'mae': MAE(),  # mean angular error for normal map and light directions
            'ssim': SSIM().to(self.rank),
            'lpips': LPIPS(normalize=True).to(self.rank),
            "simse": scale_invariant_mse,
        }
        self.train_num_samples = self.config.model.train_num_rays * (self.config.model.num_samples_per_ray + self.config.model.get('num_samples_per_ray_bg', 0))
        self.train_num_rays = self.config.model.train_num_rays
        self.save_mesh_interval = self.config.model.save_mesh_interval

    def on_load_checkpoint(self, checkpoint):
        self.trained_global_step = int(checkpoint.get("global_step"))
        print(f"[INFO] trained global step = {self.trained_global_step}")

    def on_train_end(self):
        self.trained_global_step = int(self.global_step)
        print(f"[INFO] trained global_step = {self.trained_global_step}")

    def on_train_batch_start(self, batch, batch_idx, unused=0):
        self.dataset = self.trainer.datamodule.train_dataloader().dataset
        update_module_step(self.model, self.current_epoch, self.global_step)

    def training_step(self, batch, batch_idx):
        render_out = self.model(batch['rays'])

        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / (render_out['num_samples'].sum().item()+1)))
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)

        rgb_scale_est = render_out['rays_rgb'][render_out['rays_valid'][...,0]] + 1e-3
        rgb_scale_est = rgb_scale_est.detach()

        if self.config.system.loss.lambda_rgb_weighted_l1 > 0:
            loss_rgb_weighted_l1 = F.l1_loss(render_out['rays_rgb'][render_out['rays_valid'][..., 0]] / rgb_scale_est,
                                              batch['rays_rgb'][render_out['rays_valid'][..., 0]] / rgb_scale_est)
            self.log('train/loss_rgb_weighted_l1', round(loss_rgb_weighted_l1.item(), 4), prog_bar=True)
            loss += loss_rgb_weighted_l1 * self.C(self.config.system.loss.lambda_rgb_weighted_l1)

        if self.config.system.loss.lambda_rgb_weighted_mse > 0:
            loss_rgb_weighted_mse = F.mse_loss(render_out['rays_rgb'][render_out['rays_valid'][..., 0]] / rgb_scale_est,
                                                batch['rays_rgb'][render_out['rays_valid'][..., 0]] / rgb_scale_est)
            self.log('train/loss_rgb_weighted_mse', loss_rgb_weighted_mse.mean(), prog_bar=True)
            loss += loss_rgb_weighted_mse * self.C(self.config.system.loss.lambda_rgb_weighted_mse)

        if self.config.system.loss.lambda_rgb_mse > 0:
            loss_rgb_mse = F.mse_loss(render_out['rays_rgb'][render_out['rays_valid'][..., 0]],
                                       batch['rays_rgb'][render_out['rays_valid'][..., 0]])
            self.log('train/loss_rgb_mse', loss_rgb_mse.mean(), prog_bar=True)
            loss += loss_rgb_mse * self.C(self.config.system.loss.lambda_rgb_mse)

        if self.config.system.loss.lambda_rgb_l1 > 0:
            loss_rgb_l1 = F.l1_loss(render_out['rays_rgb'][render_out['rays_valid'][...,0]], batch['rays_rgb'][render_out['rays_valid'][...,0]])
            self.log('train/loss_rgb_l1', round(loss_rgb_l1.item(), 4), prog_bar=True)
            loss += loss_rgb_l1 * self.C(self.config.system.loss.lambda_rgb_l1)

        loss_eikonal = ((torch.linalg.norm(render_out['samples_sdf_grad'], ord=2, dim=-1) - 1.)**2).mean()
        self.log('train/loss_eikonal', round(loss_eikonal.item(), 4), prog_bar=True)
        loss += loss_eikonal * self.C(self.config.system.loss.lambda_eikonal)

        opacity = torch.clamp(render_out['rays_opacity'].squeeze(-1), 1.e-3, 1.- 1.e-3)
        loss_mask = binary_cross_entropy(opacity, batch['rays_fg_mask'].float())
        self.log('train/loss_mask', round(loss_mask.item(), 4), prog_bar=True)
        loss += loss_mask * (self.C(self.config.system.loss.lambda_mask) if self.dataset.has_mask else 0.0)

        self.log('train/inv_s', round(self.model.variance.inv_s.item(), 3), prog_bar=True)

        for name, value in self.config.system.loss.items():
            if name.startswith('lambda'):
                self.log(f'train_params/{name}', self.C(value))

        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)
        self.log('train/samples_per_ray', render_out['num_samples']/self.train_num_rays, prog_bar=True)
        self.log("global_step", float(self.global_step),
                 on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=False)

        # if dataset has gt normal, evaluate the normal vectors
        if self.dataset.has_gt_normal:
            eval_normal_mask = batch["rays_fg_mask"].bool()
            normal_est = render_out['rays_normal_dir_world_space'].to(batch['rays_normal_dir_world_space'])[eval_normal_mask]
            normal_gt = batch['rays_normal_dir_world_space'].view(-1, 3)[eval_normal_mask]
            normal_est = F.normalize(normal_est, p=2, dim=-1)
            normal_gt = F.normalize(normal_gt, p=2, dim=-1)
            normal_angular_err_degree = torch.acos(torch.clamp(torch.sum(normal_est * normal_gt, dim=-1), -1.0, 1.0))
            normal_angular_err = torch.rad2deg(normal_angular_err_degree).mean()
            self.log('test/mae_normal', round(normal_angular_err.item(), 2), prog_bar=True)

        # if dataset has gt light directions, evaluate the light directions
        if self.dataset.light_dirs_GT is not None:
            light_est = F.normalize(self.model.lighting.light_dir[self.model.lighting.train_light_indices], p=2, dim=-1).detach().cpu().numpy()
            light_est[..., [1,2]] = -light_est[..., [1,2]]
            light_gt = self.dataset.light_dirs_GT[self.model.lighting.train_light_indices]

            # compute the angular error between the estimated light direction and the ground truth light direction in degrees
            light_est = light_est.reshape(-1, 3)
            light_gt = light_gt.reshape(-1, 3)
            light_angular_err = np.mean(np.arccos(np.clip(np.sum(light_est * light_gt, axis=-1), -1, 1)) * 180 / np.pi)
            self.log('test/mae_light', round(light_angular_err.item(), 2), prog_bar=True)

        return {
            'loss': loss
        }

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self.dataset = self.trainer.datamodule.val_dataloader().dataset
        update_module_step(self.model, self.current_epoch, self.global_step)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out = chunk_batch(self.model.forward, self.config.model.ray_chunk, True, batch["rays"])

        fg_mask = batch['image_fg_mask'].bool()

        # validate image
        rgb_render = rays_to_image(out['rays_rgb_w_shadow'], fg_mask)
        rgb_wo_shadow = rays_to_image(out['rays_rgb_wo_shadow'], fg_mask)
        rgb_gt = batch['image_rgb']

        rgb_render_eval = set_image_background_black(rgb_render, fg_mask).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        rgb_gt_eval = set_image_background_black(rgb_gt, fg_mask).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        psnr = self.criterions['psnr'](rgb_render[fg_mask], rgb_gt[fg_mask])  # We calculate psnr only for foreground pixels.
        ssim = self.criterions['ssim'](rgb_render_eval, rgb_gt_eval)
        lpips = self.criterions['lpips'](rgb_render_eval.clamp(0., 1.), rgb_gt_eval.clamp(0., 1.))
        self.log('val/psnr', round(psnr.item(), 2), prog_bar=True, batch_size=1)
        self.log('val/ssim', round(ssim.item(), 2), prog_bar=True, batch_size=1)
        self.log('val/lpips', round(lpips.item(), 2), prog_bar=True, batch_size=1)

        # validate normal map
        rays_normal_dir_camera_space = out['rays_normal_dir_world_space'].to(batch["R_w2c"]) @ batch['R_w2c'].T  # (N_rays, 3)
        rays_normal_dir_camera_space = F.normalize(rays_normal_dir_camera_space, p=2, dim=-1)
        rays_normal_dir_camera_space[..., [1, 2]] = -rays_normal_dir_camera_space[..., [1, 2]]  # opencv to opengl conver

        normal_map_est_camera_space = rays_to_image(rays_normal_dir_camera_space, fg_mask)

        if self.dataset.has_gt_normal:
            normal_map_gt_camera_space = batch['image_normal_camera_space']
            normal_gt_norm = torch.linalg.norm(normal_map_gt_camera_space, dim=-1)
            normal_gt_valid_mask = normal_gt_norm > 0.9

            mae_normal, normal_error_map = self.criterions['mae'](normal_map_est_camera_space, normal_map_gt_camera_space, valid_mask=normal_gt_valid_mask)
            self.log('val/mae_normal', mae_normal, prog_bar=True, batch_size=1)

        # save images
        img_grids = [
            {'type': 'rgb', 'img': normal_map_est_camera_space, 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},
            {'type': 'rgb', 'img': gamma_correct(rgb_gt), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': gamma_correct(rgb_render), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': gamma_correct(rgb_wo_shadow), 'kwargs': {'data_format': 'HWC'}}
        ]
        if self.dataset.has_gt_normal:
            img_grids.insert(0, {'type': 'rgb', 'img': normal_map_gt_camera_space, 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}})
        view_light_idx = batch["val_view_light_idx"]
        self.save_image_grid(f"validation_results/img_normal/it{self.global_step}_{view_light_idx}.png", img_grids)

        # save a slice of the sdf field
        sdf_plane = self.model.slice()
        plt.imshow(sdf_plane, cmap='jet')
        plt.colorbar()
        plt.contour(sdf_plane, levels=[0], colors='black')
        plt.savefig(self.get_save_path(f"validation_results/sdf_slice/it{self.global_step}.png"), transparent=True)
        plt.close()

        # save the light directions and intensity
        light_dir_est = self.model.lighting.light_dir[self.model.lighting.train_light_indices].detach().cpu().numpy()
        light_dir_est[..., [1, 2]] = -light_dir_est[..., [1, 2]]
        light_dir_est = light_dir_est / np.linalg.norm(light_dir_est, axis=-1, keepdims=True)
        light_intensity = self.model.lighting.intensity[self.model.lighting.train_light_indices].detach().cpu().numpy()
        light_intensity_normalized = light_intensity / light_intensity.max()

        fname_plot = self.get_save_path(f"validation_results/light/it{self.global_step}_light_dir.png")
        plot_light(light_dir_est[:, 0], light_dir_est[:, 1], fname_plot, c=light_intensity_normalized.mean(axis=-1))

        # save light intensity and directions as txt file
        np.savetxt(self.get_save_path(f"validation_results/light/it{self.global_step}_light_intensity.txt"), light_intensity)
        np.savetxt(self.get_save_path(f"validation_results/light/it{self.global_step}_light_dir.txt"), light_dir_est)

    def validation_epoch_end(self, outputs):
        pass

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self.dataset = self.trainer.datamodule.test_dataloader().dataset
        update_module_step(self.model, self.current_epoch, self.global_step)

    def test_step(self, batch, batch_idx):
        out = chunk_batch(self.model.forward, self.config.model.ray_chunk, True, batch["rays"])

        fg_mask = batch['image_fg_mask'].bool()

        # validate image
        rgb_render = rays_to_image(out['rays_rgb_w_shadow'], fg_mask)
        rgb_wo_shadow = rays_to_image(out['rays_rgb_wo_shadow'], fg_mask)
        rgb_gt = batch['image_rgb']

        if batch["light_type"] == "test" and self.dataset.light_intensity_GT is not None:
            light_intensity_est = self.model.lighting.intensity
            light_intensity_gt = self.dataset.light_intensity_GT

            intensity_ratio = light_intensity_est / torch.from_numpy(light_intensity_gt).to(light_intensity_est.device)
            light_intensity_scale = intensity_ratio[self.model.lighting.train_light_indices].mean(axis=0).to(rgb_render.dtype)  # intensity_scale

            rgb_render = rgb_render * light_intensity_scale
            rgb_wo_shadow = rgb_wo_shadow * light_intensity_scale

        rgb_render_eval = set_image_background_black(rgb_render, fg_mask).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        rgb_gt_eval = set_image_background_black(rgb_gt, fg_mask).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)

        psnr = self.criterions['psnr'](rgb_render[fg_mask],
                                       rgb_gt[fg_mask])  # We calculate psnr only for foreground pixels.
        ssim = self.criterions['ssim'](rgb_render_eval, rgb_gt_eval)
        lpips = self.criterions['lpips'](rgb_render_eval.clamp(0., 1.), rgb_gt_eval.clamp(0., 1.))
        self.log('test/psnr', round(psnr.item(), 2), prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
        self.log('test/ssim', round(ssim.item(), 2), prog_bar=True, batch_size=1, on_step=True, on_epoch=True)
        self.log('test/lpips', round(lpips.item(), 2), prog_bar=True, batch_size=1, on_step=True, on_epoch=True)

        # validate normal map
        rays_normal_dir_camera_space = out['rays_normal_dir_world_space'].to(batch["R_w2c"]) @ batch[
            'R_w2c'].T  # (N_rays, 3)
        rays_normal_dir_camera_space = F.normalize(rays_normal_dir_camera_space, p=2, dim=-1)
        rays_normal_dir_camera_space[..., [1, 2]] = -rays_normal_dir_camera_space[..., [1, 2]]  # opencv to opengl conver

        normal_map_est_camera_space = rays_to_image(rays_normal_dir_camera_space, fg_mask)

        if self.dataset.has_gt_normal:
            normal_map_gt_camera_space = batch['image_normal_camera_space']
            normal_gt_norm = torch.linalg.norm(normal_map_gt_camera_space, dim=-1)
            normal_gt_valid_mask = normal_gt_norm > 0.9

            mae_normal, normal_error_map = self.criterions['mae'](normal_map_est_camera_space,
                                                                  normal_map_gt_camera_space,
                                                                  valid_mask=normal_gt_valid_mask)
            self.log('test/mae_normal', mae_normal, prog_bar=True, batch_size=1, on_step=True, on_epoch=True)

        # save images
        img_grids = [
            {'type': 'rgb', 'img': normal_map_est_camera_space,
             'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}},
            {'type': 'rgb', 'img': gamma_correct(rgb_gt), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': gamma_correct(rgb_render), 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': gamma_correct(rgb_wo_shadow), 'kwargs': {'data_format': 'HWC'}}
        ]
        if self.dataset.has_gt_normal:
            img_grids.insert(0, {'type': 'rgb', 'img': normal_map_gt_camera_space,
                                 'kwargs': {'data_format': 'HWC', 'data_range': (-1, 1)}})
        view_light_idx = batch["view_light_idx"]
        self.save_image_grid(f"test_results/it{self.trained_global_step}/image_and_normal/it{self.trained_global_step}_{view_light_idx}.png", img_grids)


        # save psnr, ssim, lpips, mae_normal to txt file
        view_type = batch["view_type"]
        light_type = batch["light_type"]
        view_light_type = f"{view_type}_view_{light_type}_light"

        # save psnr and mae to text file
        fname = f"test_results/it{self.trained_global_step}/it{self.trained_global_step}_psnr.txt"
        save_path = self.get_save_path(fname)
        context = (f"{view_light_idx} {view_light_type}: {psnr} \n")
        with open(save_path, "a") as f:
            f.write(context)

        ssim_txt = f"test_results/it{self.trained_global_step}/it{self.trained_global_step}_ssim.txt"
        save_path = self.get_save_path(ssim_txt)
        ssim_context = (f"{view_light_idx} {view_light_type}: {ssim} \n")
        with open(save_path, "a") as f:
            f.write(ssim_context)

        lpips_txt = f"test_results/it{self.trained_global_step}/it{self.trained_global_step}_lpips.txt"
        save_path = self.get_save_path(lpips_txt)
        lpips_context = (f"{view_light_idx} {view_light_type}: {lpips} \n")
        with open(save_path, "a") as f:
            f.write(lpips_context)

        # save normal mae
        if self.dataset.has_gt_normal:
            mae_normal_txt = f"test_results/it{self.trained_global_step}/it{self.trained_global_step}_mae_normal.txt"
            save_path = self.get_save_path(mae_normal_txt)
            mae_context = (f"{view_light_idx} {view_light_type}: {mae_normal} \n")
            with open(save_path, "a") as f:
                f.write(mae_context)

        return_out = {
            'index': batch_idx,
            'view_light_idx': batch['view_light_idx'],
            'view_type': batch['view_type'],
            'light_type': batch['light_type'],
            'psnr': psnr,
            'ssim': ssim,
            'lpips': lpips,
        }
        if self.dataset.has_gt_normal:
            return_out['mae_normal'] = mae_normal
        return return_out


    def test_epoch_end(self, out):
        out = self.all_gather(out)

        psnr_train_view_test_light_list = []
        psnr_test_view_test_light_list = []
        psnr_train_view_train_light_list = []
        psnr_test_view_train_light_list = []
        for step_out in out:
            if step_out['view_type'] == 'train' and step_out['light_type'] == 'test':
                psnr_train_view_test_light_list.append(step_out['psnr'].item())
            elif step_out['view_type'] == 'test' and step_out['light_type'] == 'test':
                psnr_test_view_test_light_list.append(step_out['psnr'].item())
            elif step_out['view_type'] == 'train' and step_out['light_type'] == 'train':
                psnr_train_view_train_light_list.append(step_out['psnr'].item())
            elif step_out['view_type'] == 'test' and step_out['light_type'] == 'train':
                psnr_test_view_train_light_list.append(step_out['psnr'].item())
        if len(psnr_train_view_test_light_list) > 0:
            psnr_train_view_test_light = np.mean(psnr_train_view_test_light_list)
            self.log('test/psnr_train_view_test_light', psnr_train_view_test_light, prog_bar=True)
        if len(psnr_test_view_test_light_list) > 0:
            psnr_test_view_test_light = np.mean(psnr_test_view_test_light_list)
            self.log('test/psnr_test_view_test_light', psnr_test_view_test_light, prog_bar=True)
        if len(psnr_train_view_train_light_list) > 0:
            psnr_train_view_train_light = np.mean(psnr_train_view_train_light_list)
            self.log('test/psnr_train_view_train_light', psnr_train_view_train_light, prog_bar=True)
        if len(psnr_test_view_train_light_list) > 0:
            psnr_test_view_train_light = np.mean(psnr_test_view_train_light_list)
            self.log('test/psnr_test_view_train_light', psnr_test_view_train_light, prog_bar=True)


        # evaluate light intensity
        light_intensity = self.model.lighting.intensity[self.model.lighting.train_light_indices].detach().cpu().numpy()
        if self.dataset.light_intensity_GT is not None:
            light_intensity_gt = self.dataset.light_intensity_GT[self.model.lighting.train_light_indices]
            simse, mse_individual = self.criterions['simse'](
                self.model.lighting.intensity[self.model.lighting.train_light_indices],
                self.dataset.light_intensity_GT[self.model.lighting.train_light_indices])
            self.log('test/simse_light', simse, prog_bar=True)

        # evaluate light direction
        light_dir_est = F.normalize(self.model.lighting.light_dir[self.model.lighting.train_light_indices], p=2,
                                dim=-1).detach().cpu().numpy()
        light_dir_est[..., [1, 2]] = -light_dir_est[..., [1, 2]]  # opencv to opengl convert
        if self.dataset.light_dirs_GT is not None:
            light_dir_gt = self.dataset.light_dirs_GT[self.model.lighting.train_light_indices]
            light_mae = self.criterions['mae'](light_dir_est.reshape(-1, 3), light_dir_gt.reshape(-1, 3))[0]
            self.log('test/mae_light', light_mae, prog_bar=True)

        # save images of light directions and intensity
        fname_light_int = self.get_save_path(f"test_results/it{self.trained_global_step}/light/it{self.trained_global_step}_light_intensity.txt")
        if not os.path.exists(fname_light_int):
            np.savetxt(fname_light_int, light_intensity)
        fname_light_dir = self.get_save_path(f"test_results/it{self.trained_global_step}/light/it{self.trained_global_step}_light_dir.txt")
        if not os.path.exists(fname_light_dir):
            np.savetxt(fname_light_dir, light_dir_est)

        fname_png = self.get_save_path(f"test_results/it{self.trained_global_step}/light/it{self.trained_global_step}_light_dir_est.png")
        light_intensity = light_intensity / light_intensity.max()
        plot_light(light_dir_est[:, 0], light_dir_est[:, 1], fname_png, c=light_intensity.mean(axis=-1))

        if self.dataset.light_dirs_GT is not None:
            fname_plot_gt = self.get_save_path(f"test_results/it{self.trained_global_step}/light/light_dir_GT.png")
            if self.dataset.light_intensity_GT is not None:
                light_intensity_gt = light_intensity_gt / light_intensity_gt.max()
                plot_light(light_dir_gt[:, 0], light_dir_gt[:, 1], fname_plot_gt, c=light_intensity_gt.mean(axis=-1))

        # evaluate mesh quality
        cd, fscore = self.export()
        if self.dataset.gt_mesh_fpath is not None:
            self.log('test/cd', cd, prog_bar=True)
            self.log('test/fscore', fscore, prog_bar=True)


    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx):
        self.dataset = self.trainer.datamodule.predict_dataloader()[dataloader_idx].dataset
        update_module_step(self.model, self.current_epoch, self.global_step)

    def _predict_step_mesh(self, batch, batch_idx):
        C2W_list = batch['C2W'].detach().cpu().numpy()
        K_list = batch['K'].detach().cpu().numpy()
        img_h, img_w = batch['img_h'], batch['img_w']

        mesh = self.model.export(self.config.export)
        mesh["v_pos"] = mesh["v_pos"]
        mesh_fname_obj = f"it{self.trained_global_step}_{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}_obj_space.ply"
        mesh_fpath_obj = self.save_mesh(mesh_fname_obj, **mesh)
        self.remove_isolated_clusters(mesh_fpath_obj)

        mesh["v_pos"] = mesh["v_pos"] * self.dataset.O2W_scale + torch.tensor(self.dataset.O2W_translation)
        mesh_fname = f"it{self.trained_global_step}_{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}_world_space.ply"
        mesh_fpath = self.save_mesh(mesh_fname, **mesh)

        self.remove_isolated_clusters(mesh_fpath)

        mesh = pv.read(mesh_fpath)

        mesh_draw_dir = self.get_save_path(f"it{self.trained_global_step}_mesh_draw")
        os.makedirs(mesh_draw_dir, exist_ok=True)

        print("Visualizing the mesh...")
        for pose_id, c2w in tqdm(enumerate(C2W_list)):
            fy = K_list[pose_id][1, 1]
            mesh_fpath = os.path.join(mesh_draw_dir, f"{pose_id:02d}_eval.png")

            self.draw_mesh(mesh, c2w, fy, img_h, img_w, mesh_fpath)
        print("Visualizing the mesh done.")


    def _predict_step_relighting(self, batch, batch_idx):
        out = chunk_batch(self.model.forward, self.config.model.ray_chunk, True, batch["rays"],
                          single_light_dir_world_space=batch["single_light_dir_world_space"])
        W, H = self.dataset.img_w, self.dataset.img_h

        fg_mask = out['rays_opacity'] > 0.01
        fg_mask = fg_mask.squeeze().view(H, W)

        image_rgb = out['rays_rgb'].view(H, W, 3).detach().cpu().numpy()
        image_rgb[image_rgb > 1] = 1
        image_rgb = image_rgb ** (1 / 2.2)

        image_rgb_wo_shadow = out['rays_rgb_wo_shadow'].view(H, W, 3).detach().cpu().numpy()
        image_rgb_wo_shadow[image_rgb_wo_shadow > 1] = 1
        image_rgb_wo_shadow = image_rgb_wo_shadow ** (1 / 2.2)

        normal_map_world_space = F.normalize(out['rays_normal_dir_world_space'], p=2, dim=-1).view(H, W, 3).detach().cpu().numpy()
        normal_map_world_space[~fg_mask] = -1  # so that the background is black after mapping to [0, 1]

        self.save_rgb_image(f"it{self.trained_global_step}-relighting/normal/{batch_idx}.png", normal_map_world_space,
                            data_range=(-1, 1), data_format="HWC")
        self.save_rgb_image(f"it{self.trained_global_step}-relighting/rgb_shadowed/{batch_idx}.png", image_rgb,
                            data_format="HWC")
        self.save_rgb_image(f"it{self.trained_global_step}-relighting/rgb_unshadowed/{batch_idx}.png",
                            image_rgb_wo_shadow, data_format="HWC")


    def _predict_step_brdf(self, batch, batch_idx):
        out = chunk_batch(self.model.forward_brdf, self.config.model.ray_chunk, True,
                          batch["rays"],
                          brdf_sphere_normal=batch["normal_sphere"],
                          brdf_sphere_fg_mask=batch["normal_sphere_mask"].long(),
                          light_dir=batch['light_dir'], )

        surface_latent = out['rays_surface_latent']
        brdf_spheres_all = out['rays_brdf_sphere']
        surface_normal = out['rays_surface_normal']  # for sanity check that the ray marching is correct

        view_idx = self.dataset.view_light_index_test
        surface_normal_map = np.zeros((self.dataset.fg_mask.shape[0], self.dataset.fg_mask.shape[1], 3), dtype=np.float32)
        surface_normal_map[self.dataset.fg_mask] = surface_normal.detach().cpu().numpy()
        self.save_rgb_image(f"brdf/it{self.trained_global_step}_surface_normal_sanity_check_{view_idx}.png",
                            surface_normal_map, data_range=(-1, 1), data_format="HWC")

        print("Concatenating BRDF spheres into a single BRDF map...")

        brdf_sphere_res = batch["normal_sphere"].shape[0]
        img_h, img_w = self.dataset.fg_mask.shape[0], self.dataset.fg_mask.shape[1]
        brdf_map = np.zeros((img_h * img_w, brdf_sphere_res, brdf_sphere_res, 3), dtype=np.uint8)
        brdf_map[self.dataset.fg_mask.reshape(-1)] = (brdf_spheres_all.detach().cpu().numpy() * 255).astype(np.uint8)  # (h*w, res, res,3)
        brdf_map = brdf_map.reshape(img_h, img_w, brdf_sphere_res, brdf_sphere_res, 3)
        brdf_map = brdf_map.transpose(0, 2, 1, 3, 4).reshape(img_h * brdf_sphere_res, img_w * brdf_sphere_res, 3)

        brdf_map_fpath = self.get_save_path(f"brdf/it{self.trained_global_step}_brdf_map_{view_idx}.png")
        cv2.imwrite(brdf_map_fpath, brdf_map[..., ::-1])
        print(f"Saved BRDF map to {brdf_map_fpath}")

        reducer = umap.UMAP(n_components=3, metric='cosine')
        print(f"Applying UMAP to reduce the dimensionality of BRDF latent vectors to 3D for visualization...")
        print(f"BRDF latent shape: {surface_latent.shape}")
        brdf_latent_reduced = reducer.fit_transform(surface_latent)
        scaler = MinMaxScaler()
        brdf_latent_reduced = scaler.fit_transform(brdf_latent_reduced)

        brdf_latent_map = np.zeros((self.dataset.fg_mask.shape[0], self.dataset.fg_mask.shape[1], 3), dtype=np.float32)
        brdf_latent_map[self.dataset.fg_mask] = brdf_latent_reduced
        brdf_latent_fpath = self.get_save_path(f"brdf/it{self.trained_global_step}_brdf_latent_{view_idx}.png")
        cv2.imwrite(brdf_latent_fpath, (brdf_latent_map[..., ::-1] * 255).astype(np.uint8))
        print(f"Saved BRDF latent map to {brdf_latent_fpath}")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self._predict_handlers[dataloader_idx](batch, batch_idx)

    def on_predict_end(self):
        if self.trainer.is_global_zero:
            for dir_name in ["normal", "rgb_shadowed", "rgb_unshadowed"]:
                self.save_img_sequence(
                        f"it{self.trained_global_step}-relighting/{dir_name}",
                        f"it{self.trained_global_step}-relighting/{dir_name}",
                        '(\d+)\.png',
                        save_format='mp4',
                        fps=15
                    )

    def draw_mesh(self, mesh, c2w, fy, img_h, img_w, mesh_fpath, set_rgb=False):
        c2w = c2w.detach().cpu().numpy() if isinstance(c2w, torch.Tensor) else c2w
        fy = fy.cpu().numpy() if isinstance(fy, torch.Tensor) else fy

        camera_center = c2w[:3, 3]
        principle_axis = c2w[:3, 2]  # z axis
        up_direction = -c2w[:3, 1]  # opencv Y axis is downward, so we need to flip the y axis

        plotter = pv.Plotter(off_screen=True)
        if set_rgb:
            plotter.add_mesh(mesh,
                             rgb=True,
                             diffuse=0.5,
                             ambient=0.5,
                             specular=0.3,
                             smooth_shading=True,
                             )
        else:
            plotter.add_mesh(mesh,
                             color='w',
                             diffuse=0.5,
                             ambient=0.5,
                             specular=0.3,
                             smooth_shading=True,
                             )
        # Vertical FOV
        # https://www.nikonians.org/reviews/fov-tables
        plotter.camera.view_angle = float(np.rad2deg(np.arctan((img_h / 2) / fy) * 2.))
        plotter.camera_position = [camera_center, principle_axis + camera_center, up_direction]
        plotter.enable_eye_dome_lighting()
        plotter.show(window_size=(img_w, img_h), screenshot=mesh_fpath)
    
    def export(self):
        import open3d as o3d
        mesh = self.model.export(self.config.export)
        mesh["v_pos"] = mesh["v_pos"]
        mesh_fname_obj = os.path.join(f"mesh/it{self.trained_global_step}", f"it{self.trained_global_step}_{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}_obj_space.ply")
        mesh_fpath_obj = self.save_mesh(mesh_fname_obj, **mesh)
        self.remove_isolated_clusters(mesh_fpath_obj)

        mesh["v_pos"] = mesh["v_pos"] * self.dataset.O2W_scale + torch.tensor(self.dataset.O2W_translation)
        mesh_fname = os.path.join(f"mesh/it{self.trained_global_step}", f"it{self.trained_global_step}_{self.config.model.geometry.isosurface.method}{self.config.model.geometry.isosurface.resolution}_world_space.ply")
        mesh_fpath = self.save_mesh(mesh_fname, **mesh)

        self.remove_isolated_clusters(mesh_fpath)

        mesh = pv.read(mesh_fpath)

        if self.dataset.gt_mesh_fpath is not None:
            mesh_gt = pv.read(self.dataset.gt_mesh_fpath)
        mesh_draw_dir = self.get_save_path(f"mesh/it{self.trained_global_step}/mesh_visualization")
        os.makedirs(mesh_draw_dir, exist_ok=True)

        if self.dataset.gt_mesh_fpath is not None:
            cd, fscore, eval_mesh_o3d = self.eval_mesh(mesh_fpath, self.dataset.gt_mesh_fpath, self.dataset.C2W_val_mesh, self.dataset.K_val_mesh)
            err_mesh_path = self.get_save_path(f"mesh/it{self.trained_global_step}/it{self.trained_global_step}_gt_mesh_error_map.ply")
            o3d.io.write_triangle_mesh(err_mesh_path, eval_mesh_o3d)
            mesh_err = pv.read(err_mesh_path)

        print("Drawing the mesh from training views ...")
        try:
            img_h = self.dataset.img_h_mesh
            img_w = self.dataset.img_w_mesh
        except:
            img_h = self.dataset.img_h
            img_w = self.dataset.img_w

        for pose_id, c2w in tqdm(enumerate(self.dataset.C2W_val_mesh)):
            fy = self.dataset.K_val_mesh[pose_id][1, 1]
            mesh_fpath = os.path.join(mesh_draw_dir, f"{pose_id:02d}_eval.png")

            self.draw_mesh(mesh, c2w, fy, img_h, img_w, mesh_fpath)
            if self.dataset.gt_mesh_fpath is not None:
                mesh_gt_fpath = os.path.join(mesh_draw_dir, f"{pose_id:02d}_gt.png")
                self.draw_mesh(mesh_gt, c2w, fy, img_h, img_w, mesh_gt_fpath)

                mesh_err_fpath = os.path.join(mesh_draw_dir, f"{pose_id:02d}_err.png")
                self.draw_mesh(mesh_err, c2w, fy, img_h, img_w, mesh_err_fpath, set_rgb=True)

        print("Drawing the mesh from training views done.")
        if self.dataset.gt_mesh_fpath is not None:
            return cd, fscore
        else:
            return None, None

    def eval_mesh(self, eval_mesh_fpath, gt_mesh_fpath, c2w_list, K_list):
        import open3d as o3d
        import trimesh

        assert len(c2w_list) == len(K_list)
        # gt_pts = trimesh.load(self.dataset.gt_pts_path).vertices
        gt_mesh = trimesh.load(gt_mesh_fpath)
        eval_mesh = trimesh.load(eval_mesh_fpath)

        print("Computing the mesh error map ...")
        signed_dist, *_ = igl.signed_distance(gt_mesh.vertices, eval_mesh.vertices, eval_mesh.faces)
        print("Computing the mesh error map done.")
        dists = np.abs(signed_dist)
        dists[dists > 1] = 1
        dists = (dists * 255).astype(np.uint8)
        dists_jet = np.squeeze(cv2.applyColorMap(dists, cv2.COLORMAP_JET))[..., ::-1]

        # create an open3d mesh
        eval_mesh_o3d = o3d.geometry.TriangleMesh()
        eval_mesh_o3d.vertices = o3d.utility.Vector3dVector(gt_mesh.vertices)
        eval_mesh_o3d.triangles = o3d.utility.Vector3iVector(gt_mesh.faces)
        eval_mesh_o3d.vertex_colors = o3d.utility.Vector3dVector(dists_jet.astype(np.float32) / 255.)

        img_height = self.dataset.img_h0
        img_width = self.dataset.img_w0

        # find the closest point on the mesh to the ground truth points
        pts_eval = []
        pts_gt = []
        print("Computing the ray-mesh intersections ...")
        for view_idx in tqdm(range(len(c2w_list))):
            c2w = c2w_list[view_idx]
            K = K_list[view_idx][:3, :3]

            # compute the ray-mesh intersection
            ray_origins = np.tile(c2w[:3, 3], (img_height * img_width, 1))
            xx, yy = np.meshgrid(np.arange(img_width), np.arange(img_height))
            xyz = np.stack((xx.flatten(), yy.flatten(), np.ones_like(xx.flatten())), axis=1)
            ray_directions = np.matmul(np.linalg.inv(K), xyz.T).T
            ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=1, keepdims=True)
            ray_directions = np.matmul(c2w[:3, :3], ray_directions.T).T
            ray_directions = ray_directions / np.linalg.norm(ray_directions, axis=1, keepdims=True)
            ray_directions = ray_directions.astype(np.float32)

            locations_eval, *_ = eval_mesh.ray.intersects_location(ray_origins, ray_directions, multiple_hits=False)
            locations_gt, *_ = gt_mesh.ray.intersects_location(ray_origins, ray_directions, multiple_hits=False)
            pts_eval.append(locations_eval)
            pts_gt.append(locations_gt)
        print("Computing the ray-mesh intersections done.")
        pts_eval = np.concatenate(pts_eval, axis=0)
        pts_gt = np.concatenate(pts_gt, axis=0)
        cd, fscore = chamfer_distance_and_f1_score(pts_gt, pts_eval)
        print("Mesh evaluation done.")
        return cd, fscore, eval_mesh_o3d

    def slice(self):
        import matplotlib.pyplot as plt
        sdf_plane = self.model.slice()
        plt.imshow(sdf_plane, cmap='jet')
        plt.colorbar()
        plt.contour(sdf_plane, levels=[0], colors='black')
        plt.savefig(f"{self.save_dir}/it{self.global_step}-slice.png", transparent=True)

    def remove_isolated_clusters(self, mesh_fpath):
        import copy

        import open3d as o3d

        mesh = o3d.io.read_triangle_mesh(mesh_fpath)
        with o3d.utility.VerbosityContextManager(
                o3d.utility.VerbosityLevel.Debug) as cm:
            triangle_clusters, cluster_n_triangles, cluster_area = (
                mesh.cluster_connected_triangles())
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)

        mesh_eval = copy.deepcopy(mesh)
        largest_cluster_idx = cluster_n_triangles.argmax()
        triangles_to_remove = triangle_clusters != largest_cluster_idx
        mesh_eval.remove_triangles_by_mask(triangles_to_remove)
        mesh_eval.remove_duplicated_triangles()
        mesh_eval.remove_unreferenced_vertices()
        mesh_eval.remove_duplicated_vertices()

        o3d.io.write_triangle_mesh(mesh_fpath, mesh_eval)

        mesh = o3d.io.read_triangle_mesh(mesh_fpath)
        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()
        o3d.io.write_triangle_mesh(mesh_fpath, mesh)
