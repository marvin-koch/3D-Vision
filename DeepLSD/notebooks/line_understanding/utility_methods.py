import os
import glob
import numpy as np
import cv2
import h5py


import numpy as np


def find_file(image_dir, image_id,  pattern, cam_view):

    search_pattern = os.path.join(image_dir, image_id, "images", cam_view, pattern)
    print(search_pattern)
    files = glob.glob(search_pattern, recursive=True)
    return files[0] if files else None


def load_color_image(image_dir, image_id,  frame_str, cam_view):

    color_file = find_file(image_dir, image_id,  f"frame.{frame_str}.color.jpg", cam_view)
    if color_file is None:
        print("Color image not found in", image_dir, "with camera view", cam_view)
        return None
    img = cv2.imread(color_file)
    if img is None:
        print("Failed to load", color_file)
        return None
    # Convert from BGR to RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_depth_map(image_dir, image_id, frame_str, cam_view):

    depth_file = find_file(image_dir,  image_id, f"frame.{frame_str}.depth_meters.hdf5", cam_view)
    if depth_file is None:
        print("Depth file not found in", image_dir, "with camera view", cam_view)
        return None
    with h5py.File(depth_file, 'r') as f:
        depth = np.array(f['dataset'])
    return depth.astype(np.float32)


def load_normal_map(image_dir,  image_id, frame_str, cam_view):

    normal_file = find_file(image_dir, image_id,  f"frame.{frame_str}.normal_world.hdf5", cam_view)
    if normal_file is None:
        print("Normal file not found in", image_dir, "with camera view", cam_view)
        return None
    with h5py.File(normal_file, 'r') as f:
        normal = np.array(f['dataset'])
    return normal.astype(np.float32)

def load_world_coordinates(image_dir,  image_id, frame_str, cam_view):

    wc_file = find_file(image_dir, image_id,  f"frame.{frame_str}.position.hdf5", cam_view)
    if wc_file is None:
        print("Normal file not found in", image_dir, "with camera view", cam_view)
        return None
    with h5py.File(wc_file, 'r') as f:
        #print("Keys in the postion file:", list(f.keys()))
        wc = np.array(f['dataset'])
        #print("Shape of position data:", wc.shape)
    return wc.astype(np.float32)


#****************************************************************************************************
#****************************************************************************************************
#****************************************************************************************************


def compute_variation(mapping, k, normalize=False):
    """
    Computes the Sobel variation of a mapping (depth or normal) using a kernel size k.
    Optionally ormalizes the result by subtracting the mean and dividing by the standard deviation.
    """
    grad_x = cv2.Sobel(mapping, cv2.CV_64F, 1, 0, ksize=k)
    grad_y = cv2.Sobel(mapping, cv2.CV_64F, 0, 1, ksize=k)
    variation = np.sqrt(grad_x**2 + grad_y**2)

    normalized = variation
    if normalize: 
        mean = np.mean(variation)
        std_dev = np.std(variation)
        normalized = (variation - mean) / std_dev
    return normalized

def sobel_line_neighborhood(sobel_depth, sobel_normal, line, thickness=5):
    """
    """
    x1, y1 = int(round(line[0, 0])), int(round(line[0, 1]))
    x2, y2 = int(round(line[1, 0])), int(round(line[1, 1]))
    # Draw a thicker line mask to capture the neighborhood.
    mask_depth = cv2.line(np.zeros_like(sobel_depth), (x1, y1), (x2, y2), 1, thickness=thickness)
    mask_normal = cv2.line(np.zeros_like(sobel_normal), (x1, y1), (x2, y2), 1, thickness=thickness)
    return mask_depth * sobel_depth, mask_normal * sobel_normal


def raydepth2depth(raydepth, K):
    K_inv = np.linalg.inv(K)
    h, w = raydepth.shape[0], raydepth.shape[1]
    grids = np.meshgrid(np.arange(w), np.arange(h))
    coords_homo = [grids[0].reshape(-1), grids[1].reshape(-1), np.ones(h * w)]
    coords_homo = np.stack(coords_homo)
    coeffs = np.linalg.norm(K_inv @ coords_homo, axis=0)
    coeffs = coeffs.reshape(h, w)
    depth = raydepth / coeffs
    return depth