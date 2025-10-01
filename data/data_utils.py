from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from scipy.ndimage import center_of_mass


def fill_holes_in_mask(mask_fpath):
    """
    Fill holes in a binary mask image using OpenCV's floodFill method.

    Args:
        mask_fpath (str): Path to the binary mask image.

    Returns:
        np.ndarray: The binary mask with holes filled.
    """
    # Load the binary mask
    binary_image = cv2.imread(mask_fpath, cv2.IMREAD_GRAYSCALE)

    # Create a mask that is 2 pixels larger than the original image
    h, w = binary_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    # Copy the original image to avoid modifying it
    filled_image = binary_image.copy()

    # Use floodFill to fill the foreground (0, 0) point is background
    cv2.floodFill(filled_image, mask, (0, 0), 255)

    # Invert the filled image to keep only the original foreground
    filled_inv = cv2.bitwise_not(filled_image)

    # Combine the original image with the filled inverted image to fill small holes
    output = binary_image | filled_inv

    return output

def load_K_Rt_from_P(P):
    K, R, t, *_ = cv2.decomposeProjectionMatrix(P)  # W2C
    # CAUTION:
    # t is the camera position in world coordinate.
    # R is the world-to-camera rotation matrix
    # https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#gaaae5a7899faa1ffdf268cd9088940248
    K = K / K[2, 2]

    C2W = np.eye(4, dtype=np.float32)
    C2W[:3, :3] = R.T
    C2W[:3, 3] = (t[:3] / t[3])[:, 0]

    return K, C2W

def are_rays_intersecting_sphere(ray_origin, ray_directions, sphere_center, radius):
    """
    Determine whether multiple rays (starting from a shared origin) intersect
    or come within a given radius of a sphere.

    Parameters:
        ray_origin: (3,) numpy array - common origin of all rays
        ray_directions: (N, 3) numpy array - each row is a ray direction vector
        sphere_center: (3,) numpy array - center of the sphere
        radius: float - radius of the sphere

    Returns:
        hits: (N,) boolean array - True if the corresponding ray intersects or
              gets within the radius of the sphere, False otherwise
    """
    o = ray_origin[None, :]                  # (1, 3) - expand dims for broadcasting
    v = ray_directions                       # (N, 3) - ray directions
    c = sphere_center[None, :]               # (1, 3) - expand dims for broadcasting
                             # (1, 3) - vector from ray origin to sphere center
    v_norm = v / np.linalg.norm(v, axis=1, keepdims=True)  # (N, 3) - normalize directions
    t_prime = v_norm @ (c - o).T  # (N, 1) - distance from sphere center to ray
    x_prime = o + t_prime * v_norm            # (N, 3) - closest point along each ray to sphere center

    dists = np.linalg.norm(x_prime - c, axis=1)                  # (N,) - distances to sphere center

    # If t_prime < 0, override with distance from origin to center
    o_to_c_dist = np.linalg.norm(c - o)                   # scalar
    dists = np.where(t_prime[:, 0] < 0, o_to_c_dist, dists)           # (N,)

    # Check if distance is within radius of the sphere
    hits = dists <= radius                                # (N,)
    return hits

def convert_numpy(obj):
    """Recursively convert NumPy arrays in a dictionary to Python lists"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(element) for element in obj]
    return obj

def normalize_camera(P_list):
    A_camera_normalize = 0
    b_camera_normalize = 0
    for P in P_list:
        _, R, t, *_ = cv2.decomposeProjectionMatrix(P)
        camera_center = (t[:3] / t[3])[:, 0] # in world coordinates

        vi = R[2][:, None]  # the camera's principal axis in world coordinates
        Vi = vi @ vi.T
        A_camera_normalize += np.eye(3) - Vi
        b_camera_normalize += camera_center.T @ (np.eye(3) - Vi)
    offset = np.linalg.lstsq(A_camera_normalize, np.squeeze(b_camera_normalize), rcond=None)[0]
    return offset

def scene_normalization(P_list, mask_list, fg_area_ratio=5):
    assert len(P_list) == len(mask_list), "P_list and mask_list must have the same length"

    fg_area = np.sum([mask.sum() for mask in mask_list])
    print(f"Foreground area: {fg_area} pixels")

    with ThreadPoolExecutor() as executor:
        def compute_system_matrix(view_idx):
            K, R, o, *_ = cv2.decomposeProjectionMatrix(P_list[view_idx])
            # NOTE: R is the W2C rotation matrix, while o is the camera center in world coordinates.
            o = (o[:3] / o[3])[:, 0]
            K = K / K[2, 2]
            K_inv = np.linalg.inv(K)

            fg_com = center_of_mass(mask_list[view_idx])  # compute the center of mass of the mask
            fg_com = fg_com[::-1]  # swap x, y to match the camera coordinate system
            mi = K_inv @ np.array([[fg_com[0]], [fg_com[1]], [1]])  # convert to camera coordinates
            mi = mi / np.linalg.norm(mi)  # normalize the direction vector
            mi = R.T @ mi
            Mi = mi @ mi.T
            A_term = np.eye(3) - Mi
            b_term = o.T @ (np.eye(3) - Mi)
            return A_term, b_term
        results = executor.map(compute_system_matrix, range(len(P_list)))
    A, b = zip(*results)  # A: (N, 3, 3), b: (N, 3)
    A = np.sum(A, axis=0)  # shape: (3, 3)
    b = np.sum(b, axis=0)  # shape: (3,)
    d_O2W = np.linalg.lstsq(A, np.squeeze(b), rcond=None)[0]
    print("O2W translation:", d_O2W)

    with ThreadPoolExecutor() as executor:
        def compute_sphere_area(P):
            K, R, o, *_ = cv2.decomposeProjectionMatrix(P)
            # NOTE: R is the W2C rotation matrix, while o is the camera center in world coordinates.
            K = K / K[2, 2]
            f_x = K[0, 0]
            o = (o[:3] / o[3])[:, 0]  # camera position in world coordinates

            O_C = R @ (d_O2W - o)  # the object-space origin in the camera space
            Z = O_C[2]
            return f_x ** 2 / Z ** 2

        temp_list = executor.map(compute_sphere_area, P_list)

    sum_temp = sum(temp_list) * np.pi
    s_O2W = np.sqrt(fg_area_ratio * fg_area / sum_temp)
    print("O2W scale:", s_O2W)
    return s_O2W, d_O2W


def visualize_scene_normalization(P, mask, s_O2W, d_O2W, fpath, downscale_factor=8):
    u_O = P[..., :3] @ d_O2W + P[..., 3]  # shape: (3)
    u_O = u_O[:2] / u_O[-1]  # shape: (2)

    u_O //= downscale_factor
    u_O = u_O.astype(int)
    # print(u_O)
    fg_center = np.array(center_of_mass(mask)) // downscale_factor

    # compute whether the pixel is bounded in the unit sphere
    img_h, img_w = mask.shape
    img_h //= downscale_factor
    img_w //= downscale_factor
    xx, yy = np.meshgrid(np.arange(img_w), np.arange(img_h))  # x: right, y: down
    u_homo = np.stack((xx, yy, np.ones_like(xx)), -1)
    u_homo = u_homo.reshape(-1, 3)

    K, R, cam_center, *_ = cv2.decomposeProjectionMatrix(P)
    K = K / K[2, 2]
    K[0, 0] //= downscale_factor
    K[1, 1] //= downscale_factor
    K[0, 2] //= downscale_factor
    K[1, 2] //= downscale_factor

    cam_center = (cam_center[:3] / cam_center[3])[:, 0]

    ray_dir = (R.T @ (np.linalg.inv(K) @ u_homo.T)).T

    hits = are_rays_intersecting_sphere(cam_center, ray_dir, d_O2W, radius=s_O2W)
    hit_map = hits.reshape(img_h, img_w)

    mask = cv2.resize(mask.astype(np.uint8), (img_w, img_h), cv2.INTER_NEAREST).astype(bool)
    gray_area = hit_map * (~mask)

    mask = cv2.cvtColor(mask.astype("uint8") * 255, cv2.COLOR_GRAY2BGR)
    mask[gray_area] = 128
    circle_size = img_h // 100
    mask = cv2.circle(mask, (int(fg_center[1]), int(fg_center[0])), circle_size, (0, 0, 255),
                      -1)
    # draw a cross for the world origin
    mask = cv2.line(mask, (u_O[0] - circle_size, u_O[1]),
                    (u_O[0] + circle_size, u_O[1]), (255, 0, 0), circle_size//2)
    mask = cv2.line(mask, (u_O[0], u_O[1] - circle_size),
                    (u_O[0], u_O[1] + circle_size), (255, 0, 0), circle_size//2)

    # save the mask to the target dir
    cv2.imwrite(fpath, mask)