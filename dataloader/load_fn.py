import json

import cv2
import numpy as np
import pyexr
import rawpy

from core.registry import REG


@REG.register("fn", name="diligentmv_img")
def load_img_diligentmv(img_fpath, img_downscale=1):
    img = cv2.imread(img_fpath, cv2.IMREAD_COLOR)[..., ::-1]
    if img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.
    else:
        img = img.astype(np.float32) / 255.
    if img_downscale != 1:
        scale = 1.0 / float(img_downscale)
        img = cv2.resize(
            img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
    return img

@REG.register("fn", name="a7r5_img")
def load_img_a7r5(img_fpath, img_downscale=1):
    raw_params = dict(
        use_camera_wb=True,
        half_size=False,
        no_auto_bright=True,
        output_bps=16,
        user_flip=0,
    )

    with rawpy.imread(img_fpath) as raw:
        img = raw.postprocess(**raw_params)
        img = img[:6376, :9600]  # crop the black border
        img = img[20:-20, 33:-63]  # align the image to the jpg image, which is used for the camera calibration and mask segmentation
        img = img.astype(np.float32) / 65535.

    if img_downscale != 1:
        target_size = (int(img.shape[1] / img_downscale), int(img.shape[0] / img_downscale))
        img = cv2.resize(img, dsize=target_size, interpolation=cv2.INTER_CUBIC)
    return img


@REG.register("fn", name="mask")
def load_mask(mask_fpath, img_downscale=1):
    mask = cv2.imread(mask_fpath, cv2.IMREAD_GRAYSCALE)  # (H, W)
    if img_downscale != 1:
        target_size = (int(mask.shape[1] / img_downscale), int(mask.shape[0] / img_downscale))
        mask = cv2.resize(mask, dsize=target_size, interpolation=cv2.INTER_NEAREST)
    mask = mask.astype(bool)
    return mask

@REG.register("fn", name="diligentmv_normal")
def load_normal(normal_fpath):
    normal_img = pyexr.read(normal_fpath)[..., :3]
    normal_img /= (np.linalg.norm(normal_img, axis=-1, keepdims=True) + 1e-6)
    return normal_img

@REG.register("fn", name="load_camera")
def load_cameras(cam_fpath):
    with open(cam_fpath) as f:
        return json.load(f)

@REG.register("fn", name="select_camera")
def select_camera(view_light_index, cams, img_downscale=1):
    C2W = np.array(cams[f'C2W_{view_light_index}'])[:3]  # (3, 4)
    K = np.array(cams[f'K_{view_light_index}'])[:3, :3]  # (3, 3)
    K[:2] /= img_downscale
    return C2W, K

@REG.register("fn", name="diligentmv_light")
def load_lights(lights_fpath):
    return np.loadtxt(lights_fpath)