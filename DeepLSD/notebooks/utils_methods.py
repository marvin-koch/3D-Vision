
import os
import glob
import numpy as np
import cv2
import h5py


from sklearn.neighbors import NearestNeighbors


import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN


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

def load_world_coordinates_map(image_dir,  image_id, frame_str, cam_view):

    position_file = find_file(image_dir, image_id,  f"frame.{frame_str}.position.hdf5", cam_view)
    if position_file is None:
        print("Position file not found in", image_dir, "with camera view", cam_view)
        return None
    with h5py.File(position_file, 'r') as f:
        position = np.array(f['dataset'])
    return position.astype(np.float32)

#****************************************************************************************************
#****************************************************************************************************
#****************************************************************************************************


def compute_variation(mapping, k, normalize=False):
    """
    Computes the Sobel variation of a mapping (depth or normal) using a kernel size k.
    Normalizes the result by subtracting the mean and dividing by the standard deviation.
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



#****************************************************************************************************
#****************************************************************************************************
#****************************************************************************************************


def overlay_lines_on_image_custom(image, lines, is_struct, line_color_struct, line_color_text):
    """
    Overlays predicted lines on the image using custom colors.
    """
    overlay = image.copy()
    for i, line in enumerate(lines):
        x1, y1 = int(round(line[0, 0])), int(round(line[0, 1]))
        x2, y2 = int(round(line[1, 0])), int(round(line[1, 1]))
        color = line_color_struct if is_struct[i] else line_color_text
        cv2.line(overlay, (x1, y1), (x2, y2), color, thickness=2)
    return overlay



#****************************************************************************************************
#****************************************************************************************************
#****************************************************************************************************




def pixel_to_3d(x, y, depth, K):
    """
    Convert pixel coordinate a 3D point in camera coordinates.
    """
    f = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]
    X = (x - cx) * depth / f
    Y = (y - cy) * depth / f
    Z = depth
    return np.array([X, Y, Z])

def compute_plane_features(normal_map, depth_map, K, spatial_weight=0.5, depth_weight=1.0):
    """
    Compute per-pixel features for plane clustering.
    
    For each valid pixel (depth > 0), we build a feature vector:
      [n_x, n_y, n_z, d, spatial_weight * (x / width), spatial_weight * (y / height)]
    """
    h, w = depth_map.shape
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    
    # Flatten arrays
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    depth_flat = depth_map.flatten()
    normals_flat = normal_map.reshape(-1, 3)
    
    # Valid pixels: depth > 0
    valid_mask_flat = depth_flat > 0
    xx_valid = xx_flat[valid_mask_flat]
    yy_valid = yy_flat[valid_mask_flat]
    depth_valid = depth_flat[valid_mask_flat]
    normals_valid = normals_flat[valid_mask_flat]
    
    num_valid = depth_valid.shape[0]
    points_3d = np.zeros((num_valid, 3), dtype=np.float32)
    for i in range(num_valid):
        x = xx_valid[i]
        y = yy_valid[i]
        d_val = depth_valid[i]
        points_3d[i, :] = pixel_to_3d(x, y, d_val, K)
        
    # Compute plane parameter: d = - n dot P
    d_params = -np.sum(normals_valid * points_3d, axis=1)
    # Normalize spatial coordinates
    spatial_x = (xx_valid / float(w)).reshape(-1, 1)
    spatial_y = (yy_valid / float(h)).reshape(-1, 1)

     # Normalize depth values to [0,1] (using maximum valid depth)
    norm_depth = (depth_valid.reshape(-1, 1) / (np.max(depth_valid) + 1e-8))
    
    # Assemble feature vector: normals, plane parameter, weighted spatial coordinates, and weighted normalized depth.
    features = np.hstack((normals_valid,
                          d_params.reshape(-1, 1),
                          spatial_weight * spatial_x,
                          spatial_weight * spatial_y,
                          depth_weight * norm_depth))
                          
    coords = np.stack((yy_valid, xx_valid), axis=1)  # (row, col)
    
    valid_mask = valid_mask_flat.reshape(h, w)
    return features, coords, valid_mask

def cluster_planes(features, coords, image_shape, eps=0.2, min_samples=10, sample_rate=0.2, threshold=10000):
    """
    Cluster plane features using DBSCAN on a random sample of features.
    Returns:
      segmentation_map: A 2D array where each valid pixel is assigned a cluster label.
                        Noise pixels are labeled -1.
    """
    # Normalize the features (zero mean, unit variance)
    norm_features = (features - np.mean(features, axis=0)) / (np.std(features, axis=0) + 1e-8)
    
    n_points = norm_features.shape[0]
    if sample_rate < 1.0:
        n_sample = int(n_points * sample_rate)
        sample_indices = np.random.choice(n_points, size=n_sample, replace=False)
    else:
        sample_indices = np.arange(n_points)
    
    sample_features = norm_features[sample_indices]
    
    # Run DBSCAN on the sampled (normalized) features
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    sample_labels = db.fit_predict(sample_features)
    
    # Use nearest neighbor search to assign a label to every feature
    nbrs = NearestNeighbors(n_neighbors=1, metric='euclidean').fit(sample_features)
    distances, nn_indices = nbrs.kneighbors(norm_features)
    distances = distances.flatten()
    nn_indices = nn_indices.flatten()
    
    # Assign label
    full_labels = np.array([sample_labels[idx] if dist <= threshold else -1           # ***************************
                              for idx, dist in zip(nn_indices, distances)])
    
    h, w = image_shape
    segmentation_map = -1 * np.ones((h, w), dtype=np.int32)
    for i, label in enumerate(full_labels):
        r, c = int(coords[i, 0]), int(coords[i, 1])
        segmentation_map[r, c] = label
        
    # Post-process: for each cluster (ignoring noise), use connected components to split spatially disjoint regions.
    final_segmentation = -1 * np.ones((h, w), dtype=np.int32)
    new_label = 0
    unique_labels = np.unique(segmentation_map)
    for label in unique_labels:
        if label == -1:
            continue
        mask = (segmentation_map == label).astype(np.uint8)
        num_components, comps = cv2.connectedComponents(mask, connectivity=8)
        for comp in range(1, num_components):  # Skip background (component 0)
            final_segmentation[comps == comp] = new_label
            new_label += 1
    return final_segmentation


def overlay_plane_segmentation(color_img, segmentation_map):
    """
    Overlay segmentation boundaries on the color image.
    
    This function finds pixels at boundaries between different planar regions and draws them in green.
    """
    overlay = color_img.copy()
    boundaries = np.zeros(segmentation_map.shape, dtype=np.uint8)
    h, w = segmentation_map.shape
    # Detect boundaries by checking if a pixel's neighbors have a different label.
    for r in range(1, h-1):
        for c in range(1, w-1):
            center = segmentation_map[r, c]
            if center == -1:
                continue
            if (segmentation_map[r-1, c] != center or segmentation_map[r+1, c] != center or
                segmentation_map[r, c-1] != center or segmentation_map[r, c+1] != center):
                boundaries[r, c] = 255
    # Draw boundaries in green on the overlay image.
    overlay[boundaries == 255] = [0, 255, 0]
    return overlay



def colorize_segmentation_map(segmentation_map):
    """
    Given a 2D segmentation map, assign a random color to each unique plane label.
    """
    unique_labels = np.unique(segmentation_map)
    print("Unique labels:", len(unique_labels))
    label_to_color = {}
    for label in unique_labels:
        if label == -1:
            label_to_color[label] = np.array([0, 0, 0], dtype=np.uint8)
        else:
            # Generate a random color (RGB)
            label_to_color[label] = np.random.randint(0, 256, size=3, dtype=np.uint8)
    h, w = segmentation_map.shape
    colorized = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in label_to_color.items():
        mask = segmentation_map == label
        colorized[mask] = color
    return colorized




#****************************************************************************************************
#****************************************************************************************************
#****************************************************************************************************

