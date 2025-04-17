import cv2
import numpy as np

def compute_variation_laplace(mapping, k, depth=False):
    """
    Computes the Sobel variation of a mapping (depth or normal) using a kernel size k.
    Normalizes the result by subtracting the mean and dividing by the standard deviation.
    """
   
    laplacian = cv2.Laplacian(mapping, cv2.CV_64F, ksize=k)
    
    # Take the absolute value to measure the magnitude of variation
    variation = np.abs(laplacian)

    return variation


def compute_variation(mapping, k, depth=False):
    """
    Computes the Sobel variation of a mapping (depth or normal) using a kernel size k.
    Normalizes the result by subtracting the mean and dividing by the standard deviation.
    """
   
    grad_x = cv2.Sobel(mapping, cv2.CV_64F, 1, 0, ksize=k)
    grad_y = cv2.Sobel(mapping, cv2.CV_64F, 0, 1, ksize=k)

    variation = np.sqrt(grad_x**2 + grad_y**2)

    return variation

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



# ===========================
# Plotting Function

# ===========================
# Reprojection Function
def reproject_depth_to_points(depth_map, intrinsics):
    """
    Reproject a depth map to 3D camera coordinates using camera intrinsics.
    
    Args:
        depth_map (np.ndarray): (H, W) depth map.
        intrinsics (np.ndarray): (3, 3) camera intrinsics.
    
    Returns:
        np.ndarray: 3D point cloud of shape (H, W, 3).
    """
    H, W = depth_map.shape
    if intrinsics[0, 0] < 1:  # heuristic check
        intrinsics[0, 0] *= W  # fx
        intrinsics[0, 2] *= W  # cx
        intrinsics[1, 1] *= H  # fy
        intrinsics[1, 2] *= H  # cy
    print("???")
    i, j = np.indices((H, W))
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]
    
    X = (j - cx) * depth_map / fx
    Y = (i - cy) * depth_map / fy
    Z = depth_map
    points = np.stack((X, Y, Z), axis=-1)
    return points

# ===========================
# Normal Map Computation Function
def compute_normal_map_from_points(points, ksize=3):
    """
    Compute a normal map from a 3D point cloud using spatial gradients.
    
    Args:
        points (np.ndarray): 3D points of shape (H, W, 3).
        mask (np.ndarray, optional): Binary mask (H, W) where valid pixels == 1.
        ksize (int, optional): Kernel size for the Sobel operator.
        
    Returns:
        np.ndarray: Normal map of shape (H, W, 3) with unit normals.
    """
    points = np.ascontiguousarray(points.astype(np.float32))

    grad_x = np.zeros_like(points, dtype=np.float32)
    grad_y = np.zeros_like(points, dtype=np.float32)
    
    for channel in range(3):
        channel_data = np.ascontiguousarray(points[..., channel])
        grad_x[..., channel] = cv2.Sobel(channel_data, cv2.CV_32F, 1, 0, ksize=ksize)
        grad_y[..., channel] = cv2.Sobel(channel_data, cv2.CV_32F, 0, 1, ksize=ksize)
    
    # Compute cross product to obtain normals.
    normals = np.cross(grad_x, grad_y)
    
    # Normalize the normals.
    norm = np.linalg.norm(normals, axis=2, keepdims=True)
    norm[norm == 0] = 1  # Prevent division by zero.
    normals = normals / norm
    
        
    return normals



def compute_plane_point(point, normal):
    """
    Compute plane coefficients from a 3D point and its normal vector.
    """
    denom = np.linalg.norm(normal)
    normal = normal / denom  # Normalize the normal vector
    a, b, c = normal
    d = -np.dot(normal, point)
    return np.array([a, b, c, d])  # Return plane coefficients

def calculate_plane_for_map(normal_map, world_coordinates):
    """
    Calculate a plane for every pixel in the normal map using the corresponding world coordinate.
    """
    plane_map = []
    for y in range(normal_map.shape[0]):
        for x in range(normal_map.shape[1]):
            plane_map.append(compute_plane_point(world_coordinates[y, x], normal_map[y, x]))
    return np.array(plane_map).reshape(normal_map.shape[0], normal_map.shape[1], 4)


def compute_plane_point(point, normal):
    """
    Compute plane coefficients from a 3D point and its normal vector.
    """
    denom = np.linalg.norm(normal)
    normal = normal / denom  # Normalize the normal vector
    a, b, c = normal
    d = -np.dot(normal, point)
    return np.array([a, b, c, d])  # Return plane coefficients

def calculate_plane_for_map(normal_map, world_coordinates):
    """
    Calculate a plane for every pixel in the normal map using the corresponding world coordinate.
    """
    plane_map = []
    for y in range(normal_map.shape[0]):
        for x in range(normal_map.shape[1]):
            plane_map.append(compute_plane_point(world_coordinates[y, x], normal_map[y, x]))
    return np.array(plane_map).reshape(normal_map.shape[0], normal_map.shape[1], 4)

import matplotlib.pyplot as plt

def visualize_plane_clusters(labels_2d, title):
    """
    Visualize plane cluster labels as a color-coded image.
    labels_2d: (H, W) integer labels.
    """
    plt.figure()
    unique_labels = np.unique(labels_2d)
    num_labels = len(unique_labels)

    # Generate random colors for each cluster
    # shape (num_labels, 3)
    colors = np.random.rand(num_labels, 3)

    # Build color image
    H, W = labels_2d.shape
    color_img = np.zeros((H, W, 3), dtype=np.float32)
    for i, label in enumerate(unique_labels):
        color_img[labels_2d == label] = colors[i]

    plt.imshow(color_img)
    plt.title(f"Plane Clusters  {title}")
    plt.axis("off")
    plt.show()


import matplotlib.pyplot as plt

def visualize_plane_clusters(labels_2d, title):
    """
    Visualize plane cluster labels as a color-coded image.
    labels_2d: (H, W) integer labels.
    """
    plt.figure()
    unique_labels = np.unique(labels_2d)
    num_labels = len(unique_labels)

    # Generate random colors for each cluster
    # shape (num_labels, 3)
    colors = np.random.rand(num_labels, 3)

    # Build color image
    H, W = labels_2d.shape
    color_img = np.zeros((H, W, 3), dtype=np.float32)
    for i, label in enumerate(unique_labels):
        color_img[labels_2d == label] = colors[i]

    plt.imshow(color_img)
    plt.title(f"Plane Clusters  {title}")
    plt.axis("off")
    plt.show()


def get_sobel_edges(img, ksize=3):
    """
    Computes the Sobel operator for an image.
    Args:
        img (np.array): Grayscale image.
        ksize (int): Kernel size for Sobel operator.
    Returns:
        np.array: Absolute gradient magnitude image.
    """
    # Compute gradients along x and y directions
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=ksize)
    
    # Calculate the gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    # Convert to 8-bit image
    grad_magnitude = cv2.convertScaleAbs(grad_magnitude)
    
    return grad_magnitude

def threshold_edges(edges, thresh_val):
    """
    Thresholds the edge map to produce a binary mask.
    Args:
        edges (np.array): Edge map image.
        thresh_val (int): Threshold value.
    Returns:
        np.array: Binary image.
    """
    _, binary = cv2.threshold(edges, thresh_val, 255, cv2.THRESH_BINARY)
    return binary


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


import random

def fit_plane_from_points(pts):
    # Fit plane from three points using SVD or cross-product.
    # Select three points
    p1, p2, p3 = pts
    # Compute the normal via cross product
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    norm_length = np.linalg.norm(normal)
    if norm_length == 0:
        return None
    normal = normal / norm_length
    # Compute d as -dot(normal, p1)
    d = -np.dot(normal, p1)
    return normal, d

def compute_distance_to_plane(points, normal, d):
    # Calculate point-to-plane distance.
    distances = np.abs(np.dot(points, normal) + d)
    return distances

def ransac_plane_fit(points, num_iterations=100, threshold=0.01, min_inliers_ratio=0.5):
    best_inliers = []
    best_model = None

    n_points = points.shape[0]
    for _ in range(num_iterations):
        # Randomly choose three points that are non-collinear
        sample_indices = random.sample(range(n_points), 3)
        sample_pts = points[sample_indices]
        res = fit_plane_from_points(sample_pts)
        if res is None:
            continue
        normal, d = res

        # Compute distances for all points
        distances = compute_distance_to_plane(points, normal, d)
        inliers = distances < threshold
        num_inliers = np.sum(inliers)

        if num_inliers > len(best_inliers) and num_inliers > min_inliers_ratio * n_points:
            best_inliers = inliers
            best_model = (normal, d)

    return best_model, best_inliers


from collections import Counter

def find_line_planes(lines, segmentation_map, get_line_pixels_func):
    """
    For each line, determine the most common plane label by sampling pixels from the segmentation map.
    """
    

    
    line_labels = []
    for line in lines:
        pixel_coords = get_line_pixels_func(line, segmentation_map)
        labels = [segmentation_map[y, x] for x, y in pixel_coords]
        
        most_common_label = max(set(labels), key=labels.count)  # Find the most common label
        
        line_labels.append(most_common_label)
        

    return line_labels


def get_line_pixels_trim(line, maps, trim_ratio=0.25):
    """
    Get all pixel coordinates along a line using cv2.line.
    """
  

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

    height, width = maps.shape[:2]

    blank_image = np.zeros((height, width), dtype=np.uint8)
    cv2.line(blank_image, (x1, y1), (x2, y2), color=255, thickness=10) #TODO it was 3 before, we changed it to 1
    
    y_coords, x_coords = np.where(blank_image == 255)
    return list(zip(x_coords, y_coords))


import os
import cv2
import numpy as np
import torch 

from numpy import linalg as LA
import matplotlib.pyplot as plt

from deeplsd.geometry.viz_2d import plot_images

import networkx as nx
import cv2
import torch
import time

def create_optimal_offset_lines_fast(line, normal_map, offset_amount=1.0, num_samples=100, angle_steps=36):
    p1, p2 = line
    H, W = normal_map.shape[:2]
    xs = np.linspace(p1[0], p2[0], num_samples)
    ys = np.linspace(p1[1], p2[1], num_samples)
    base_pts = np.stack([xs, ys], axis=1)

    best_score = -np.inf
    best_d = None

    for theta in np.linspace(0, np.pi, angle_steps, endpoint=False):
        d = np.array([np.cos(theta), np.sin(theta)])
        offset_vec = offset_amount * d
        pts1 = base_pts + offset_vec
        pts2 = base_pts - offset_vec

        # Vectorized rounding & clipping
        coords1 = np.round(pts1).astype(np.int32)
        coords2 = np.round(pts2).astype(np.int32)
        coords1[:, 0] = np.clip(coords1[:, 0], 0, W - 1)
        coords1[:, 1] = np.clip(coords1[:, 1], 0, H - 1)
        coords2[:, 0] = np.clip(coords2[:, 0], 0, W - 1)
        coords2[:, 1] = np.clip(coords2[:, 1], 0, H - 1)

        normals1 = normal_map[coords1[:, 1], coords1[:, 0]]  # [N, 3]
        normals2 = normal_map[coords2[:, 1], coords2[:, 0]]  # [N, 3]

        score = np.sum(np.linalg.norm(normals1 - normals2, axis=1))
        if score > best_score:
            best_score = score
            best_d = d

    if best_d is None:
        # Handle the case where no valid offset vector was found
        print("Warning: No valid offset found!")
        # You can decide on a default value or return a default behavior
        best_d = np.array([0.0, 0.0])  # Default direction (no offset)

    offset = offset_amount * best_d
    return np.array([p1 + offset, p2 + offset]), np.array([p1 - offset, p2 - offset])


def compute_shifted_line_fast(line, depth_map, w, h, offset=1.0, num_samples=100):
    tangent = line[1] - line[0]
    t_norm = tangent / np.linalg.norm(tangent)
    perp = np.array([-t_norm[1], t_norm[0]]) * offset

    xs = np.linspace(line[0, 0], line[1, 0], num_samples)
    ys = np.linspace(line[0, 1], line[1, 1], num_samples)
    points = np.stack([xs, ys], axis=1)

    pos_pts = points + perp
    neg_pts = points - perp

    pos_coords = np.round(pos_pts).astype(np.int32)
    neg_coords = np.round(neg_pts).astype(np.int32)
    pos_coords[:, 0] = np.clip(pos_coords[:, 0], 0, w - 1)
    pos_coords[:, 1] = np.clip(pos_coords[:, 1], 0, h - 1)
    neg_coords[:, 0] = np.clip(neg_coords[:, 0], 0, w - 1)
    neg_coords[:, 1] = np.clip(neg_coords[:, 1], 0, h - 1)

    depth_pos = depth_map[pos_coords[:, 1], pos_coords[:, 0]]
    depth_neg = depth_map[neg_coords[:, 1], neg_coords[:, 0]]

    avg_depth_pos = np.mean(depth_pos)
    avg_depth_neg = np.mean(depth_neg)

    chosen_offset = perp if avg_depth_pos < avg_depth_neg else -perp
    return line + chosen_offset



from sklearn.linear_model import RANSACRegressor
import numpy as np

def fit_normal_ransac(normals, threshold_deg=15):
    """
    Fit dominant normal direction using RANSAC.
    Assumes normals are unit vectors.
    Returns the estimated dominant normal and a boolean inlier mask.
    """
    if normals.shape[0] < 3:
        return None, None

    # Convert 
    # to cosine similarity threshold
    threshold_rad = np.deg2rad(threshold_deg)
    cos_thresh = np.cos(threshold_rad)

    best_normal = None
    max_inliers = 0
    best_inlier_mask = None

    # RANSAC loop (manual, for unit vectors)
    for _ in range(100):  # or fewer, depending on performance tradeoff
        i = np.random.choice(len(normals))
        candidate = normals[i]
        candidate /= np.linalg.norm(candidate)

        cos_angles = normals @ candidate
        inlier_mask = cos_angles > cos_thresh

        num_inliers = np.sum(inlier_mask)
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_normal = candidate
            best_inlier_mask = inlier_mask

    return best_normal, best_inlier_mask


def ransac_plane_equations(planes, angle_thresh_deg=10, dist_thresh=0.02, iterations=100):
    # Normalize all planes
    if len(planes) < 3:
        return None, None

    angle_thresh_rad = np.deg2rad(angle_thresh_deg)
    cos_thresh = np.cos(angle_thresh_rad)

    best_inliers = []
    best_plane = None

    for _ in range(iterations):
        ref_idx = np.random.randint(0, len(planes))
        ref_plane = planes[ref_idx]
        ref_normal = ref_plane[:3]
        ref_d = ref_plane[3]

        # Compare all planes to the reference
        dot_products = planes[:, :3] @ ref_normal
        angles_ok = np.abs(dot_products) > cos_thresh

        # Compare distance offsets (from origin)
        d_diffs = np.abs(planes[:, 3] - ref_d)
        distances_ok = d_diffs < dist_thresh

        inliers = angles_ok & distances_ok

        if np.sum(inliers) > np.sum(best_inliers):
            best_inliers = inliers
            best_plane = ref_plane

    if best_plane is None or np.sum(best_inliers) < 3:
        return None, None

    # Average inlier planes and renormalize
    inlier_planes = planes[best_inliers]
    rep_plane = np.mean(inlier_planes, axis=0)
    rep_plane /= np.linalg.norm(rep_plane[:3])

    return rep_plane, best_inliers

