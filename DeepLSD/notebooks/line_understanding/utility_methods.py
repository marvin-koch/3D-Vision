import os
import glob
import numpy as np
import cv2
import h5py
import json

import numpy as np


def find_file(image_dir, image_id,  pattern, cam_view):

    search_pattern = os.path.join(image_dir, image_id, "images", cam_view, pattern)
    print(search_pattern)
    files = glob.glob(search_pattern, recursive=True)
    return files[0] if files else None

def find_file_moge_gt(image_dir,  pattern):

    search_pattern = os.path.join(image_dir, pattern)
    print(search_pattern)
    files = glob.glob(search_pattern, recursive=True)
    return files[0] if files else None

def load_color_image_moge_gt(image_dir):
    return load_color_image(image_dir, "", "", "", dataset="moge_gt")

def load_color_image(image_dir, image_id,  frame_str, cam_view, dataset="hypersim"):

    file_name = ""
    if dataset=="hypersim":
        file_name = f"frame.{frame_str}.color.jpg"
        color_file = find_file(image_dir, image_id, file_name, cam_view)

    elif dataset =="moge_gt":
        file_name = f"image.jpg"
        color_file = find_file_moge_gt(image_dir, file_name)

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

def load_depth_map_png(image_dir):

    depth_file = find_file_moge_gt(image_dir, f"depth.png")
    if depth_file is None:
        print("Depth file not found in", image_dir)
        return None
    depth = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE)
    
    return np.array(depth).astype(np.float32)

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

def load_K(image_dir,  image_id, frame_str, cam_view):

    normal_file = find_file(image_dir, image_id,  f"frame.{frame_str}.K.hdf5", cam_view)
    if normal_file is None:
        print("Normal file not found in", image_dir, "with camera view", cam_view)
        return None
    with h5py.File(normal_file, 'r') as f:
        normal = np.array(f['dataset'])
    return normal.astype(np.float32)

def load_mask(image_dir,  image_id, frame_str, cam_view):

    normal_file = find_file(image_dir, image_id,  f"frame.{frame_str}.mask.hdf5", cam_view)
    if normal_file is None:
        print("Normal file not found in", image_dir, "with camera view", cam_view)
        return None
    with h5py.File(normal_file, 'r') as f:
        normal = np.array(f['dataset'])
    return normal.astype(np.float32)

def load_intrinsics_json(image_dir):
    K_file = find_file_moge_gt(image_dir, f"meta.json")
    # Parse the JSON data
    with open(K_file, 'r') as file:
        data = json.load(file)
    # Convert the 'intrinsics' list into a NumPy array
    intrinsics_matrix = np.array(data["intrinsics"])
    
    return intrinsics_matrix.astype(np.float64)

def reconstruct_3d_from_depth(depth, intrinsics):
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Create meshgrid of pixel coordinates
    x, y = np.meshgrid(np.arange(W), np.arange(H))

    # Convert to normalized camera coordinates
    x_cam = (x - cx) / fx
    y_cam = (y - cy) / fy

    # Backproject to 3D
    X = x_cam * depth
    Y = y_cam * depth
    Z = depth

    # Stack into 3D points (H, W, 3)
    points_3d = np.stack((X, Y, Z), axis=-1)
    return points_3d.astype(np.float32)

def calculate_normal_map_from_depth(depth_map, ksize=3):
    # Compute gradients
    grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=ksize)

    # Compute normal vectors
    normal_x = -grad_x
    normal_y = -grad_y
    normal_z = np.ones_like(depth_map)

    # Normalize the normals
    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm

    # Convert to RGB
    # normal_map = np.stack([(normal_x + 1) / 2 * 255,
    #                     (normal_y + 1) / 2 * 255,
    #                     (normal_z + 1) / 2 * 255], axis=-1).astype(np.float64)
    normal_map = np.stack([normal_x, normal_y, normal_z], axis=-1).astype(np.float64)

    return normal_map
#****************************************************************************************************
#****************************************************************************************************
#****************************************************************************************************


def compute_variation(mapping, k, depth=False):
    """
    Computes the Sobel variation of a mapping (depth or normal) using a kernel size k.
    Normalizes the result by subtracting the mean and dividing by the standard deviation.
    """
   
    grad_x = cv2.Sobel(mapping, cv2.CV_64F, 1, 0, ksize=k)
    grad_y = cv2.Sobel(mapping, cv2.CV_64F, 0, 1, ksize=k)

    variation = np.sqrt(grad_x**2 + grad_y**2)

    return variation
"""
    # mean = np.mean(variation)
    # std_dev = np.std(variation)
    # normalized = (variation - mean) / std_dev
    # return normalized
    max_depth = 399742.56
    min_depth = 2.83e-12

    if depth:
        norm = (variation - min_depth) /(max_depth -min_depth)
    else:
        norm = (variation - 0) / (6.043567e+14 - 0)
    return norm
"""

def compute_variation_laplace(mapping, k, depth=False):
    """
    Computes the Sobel variation of a mapping (depth or normal) using a kernel size k.
    Normalizes the result by subtracting the mean and dividing by the standard deviation.
    """
   
    laplacian = cv2.Laplacian(mapping, cv2.CV_64F, ksize=k)
    
    # Take the absolute value to measure the magnitude of variation
    variation = np.abs(laplacian)

    return variation

def sigmoid(x, lam=10, tau=0.01):
    """
    Compute sigmoid function for soft thresholding.
    - lam: scaling factor (higher = sharper transition)
    - tau: threshold shift
    """
    return 1 / (1 + np.exp(-lam * (x - tau)))


def sobel_line(sobel_depth, sobel_normal, line, trim_ratio=0.25):
    """
    Computes the Sobel response in a trimmed neighborhood along a given line.

    Parameters:
        
    sobel_depth: 2D array of Sobel-filtered depth image.
    sobel_normal: 2D array of Sobel-filtered normal image.
    line: 2x2 array representing endpoints [[x1, y1], [x2, y2]].
    trim_ratio: Ratio (0-0.5) to trim from both ends of the line.

        Returns:
        
    Tuple of masked Sobel depth and normal in the trimmed line neighborhood."""
    p1 = np.array(line[0], dtype=np.float32)
    p2 = np.array(line[1], dtype=np.float32)

    # Direction vector and length
    direction = p2 - p1
    length = np.linalg.norm(direction)
    unit_dir = direction / (length + 1e-8)

    # Shorten line by trim_ratio from both ends
    trim_len = length * trim_ratio
    new_p1 = p1 + unit_dir * trim_len
    new_p2 = p2 - unit_dir * trim_len

    # Convert to integer pixel coordinates
    x1, y1 = int(round(new_p1[0])), int(round(new_p1[1]))
    x2, y2 = int(round(new_p2[0])), int(round(new_p2[1]))

    # Create masks
    mask_depth = cv2.line(np.zeros_like(sobel_depth), (x1, y1), (x2, y2), 1, 1)
    mask_normal = cv2.line(np.zeros_like(sobel_normal), (x1, y1), (x2, y2), 1, 1)
    return mask_depth * sobel_depth, mask_normal * sobel_normal

def sobel_line_neighborhood(sobel_depth, sobel_normal, line, thickness=1):
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