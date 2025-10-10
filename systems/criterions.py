import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import KDTree


# from https://github.com/junxuan-li/SCPS-NIR/blob/775db12dea996a6a04af46582b3e60422bcba776/utils.py#L16
def scale_invariant_mse(pred_i, gt_i):
    if not isinstance(pred_i, torch.Tensor):
        pred_i = torch.from_numpy(pred_i)
    if not isinstance(gt_i, torch.Tensor):
        gt_i = torch.from_numpy(gt_i)
    # ensure both are in the same device
    pred_i = pred_i.to(gt_i.device)
    # ensure both are float32
    pred_i = pred_i.float()
    gt_i = gt_i.float()

    # Red channel:
    gt_i_c = gt_i[:, :1]
    pred_i_c = pred_i[:, :1]
    scale = torch.linalg.lstsq(pred_i_c, gt_i_c).solution
    ints_ratio1 = (gt_i_c - scale * pred_i_c).abs() / (gt_i_c + 1e-8)
    # Green channel:
    gt_i_c = gt_i[:, 1:2]
    pred_i_c = pred_i[:, 1:2]
    scale = torch.linalg.lstsq(pred_i_c, gt_i_c).solution
    ints_ratio2 = (gt_i_c - scale * pred_i_c).abs() / (gt_i_c + 1e-8)
    # Blue channel:
    gt_i_c = gt_i[:, 2:3]
    pred_i_c = pred_i[:, 2:3]
    scale = torch.linalg.lstsq(pred_i_c, gt_i_c).solution
    ints_ratio3 = (gt_i_c - scale * pred_i_c).abs() / (gt_i_c + 1e-8)

    ints_ratio = (ints_ratio1 + ints_ratio2 + ints_ratio3) / 3
    return ints_ratio.mean().item(), ints_ratio.mean(dim=-1)


class MAE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, valid_mask=None):
        # if input is not a tensor, convert it to a tensor
        if not isinstance(inputs, torch.Tensor):
            inputs = torch.from_numpy(inputs)
        if not isinstance(targets, torch.Tensor):
            targets = torch.from_numpy(targets)
        # compute the angular error between two vectors
        dot = torch.sum(inputs * targets, dim=-1)
        dot = torch.clamp(dot, -1, 1)
        err_map = torch.acos(dot) / np.pi * 180

        if valid_mask is not None:
            value = err_map[valid_mask]
            err_map[~valid_mask] = torch.nan
        else:
            value = err_map

        return torch.mean(value), err_map


def binary_cross_entropy(input, target):
    """
    F.binary_cross_entropy is not numerically stable in mixed-precision training.
    """
    return -(target * torch.log(input) + (1 - target) * torch.log(1 - input)).mean()


def chamfer_distance_and_f1_score(ref_points, eval_points, f_threshold=0.5):
    """
    This function calculates the chamfer distance and f1 score between two sets of points.

    Parameters:
    ref_points (numpy.ndarray): Reference points. A (p, 3) array representing points in the world space.
    eval_points (numpy.ndarray): Points to be evaluated. A (p, 3) array representing points in the world space.
    f_threshold (float, optional): Threshold for f1 score calculation. Default is 0.5mm.

    Returns:
    chamfer_dist (float): The chamfer distance between gt_points and eval_points.
    f_score (float): The f1 score between gt_points and eval_points.
    """
    print("Computing Chamfer distance and f1 score...")
    distance_eval2gt, _ = KDTree(ref_points).query(eval_points, k=1, p=2)   # p=2 for Euclidean distance
    distance_gt2eval, _ = KDTree(eval_points).query(ref_points, k=1, p=2)

    # following Uncertainty-aware deep multi-view photometric stereo
    chamfer_dist = (np.mean(distance_eval2gt) + np.mean(distance_gt2eval))/2

    precision = np.mean(distance_eval2gt < f_threshold)
    recall = np.mean(distance_gt2eval < f_threshold)
    f_score = 2 * precision * recall / (precision + recall)

    return chamfer_dist, f_score