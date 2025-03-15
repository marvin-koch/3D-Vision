
import os
import glob
import numpy as np
import cv2
import h5py


import cv2
import numpy as np
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
    Convert pixel coordinate (x,y) with depth value into a 3D point in camera coordinates.
    """
    f = K[0, 0]
    cx = K[0, 2]
    cy = K[1, 2]
    X = (x - cx) * depth / f
    Y = (y - cy) * depth / f
    Z = depth
    return np.array([X, Y, Z])

def compute_plane_features(normal_map, depth_map, K, spatial_weight=0.5):
    """
    Compute per-pixel features for plane clustering.
    
    For each valid pixel (depth > 0), we build a feature vector:
      [n_x, n_y, n_z, d, spatial_weight * (x / width), spatial_weight * (y / height)]
    where d = - n · P is the plane parameter computed from the 3D point P.
    
    Returns:
      features: Array of shape (N, 6) for valid pixels.
      coords: Array of shape (N, 2) with (row, col) coordinates for valid pixels.
      valid_mask: A boolean mask of shape (h, w) indicating valid pixels.
    """
    h, w = depth_map.shape
    # Create grid of pixel coordinates
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))
    xx = xx.astype(np.float32)
    yy = yy.astype(np.float32)
    # Flatten arrays
    xx_flat = xx.flatten()
    yy_flat = yy.flatten()
    depth_flat = depth_map.flatten()
    normals_flat = normal_map.reshape(-1, 3)
    # Valid pixels: depth > 0 (and optionally non-NaN)
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
    # Assemble feature vector: normals, plane parameter, and weighted spatial coordinates
    features = np.hstack((normals_valid, d_params.reshape(-1, 1), spatial_weight * spatial_x, spatial_weight * spatial_y))
    coords = np.stack((yy_valid, xx_valid), axis=1)  # (row, col)
    
    valid_mask = valid_mask_flat.reshape(h, w)
    return features, coords, valid_mask

def cluster_planes(features, coords, image_shape, eps=0.5, min_samples=50):
    """
    Cluster plane features using DBSCAN and then perform connected component analysis
    on each cluster to split spatially disjoint regions.
    
    Parameters:
      eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other.
      min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
    
    Returns:
      segmentation_map: A 2D array (shape: image height x width) where each unique integer
                        label corresponds to a distinct planar region.
      Pixels with invalid depth are labeled -1.
    """
    # Run DBSCAN on the features
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean')
    labels = db.fit_predict(features)
    
    h, w = image_shape
    segmentation_map = -1 * np.ones((h, w), dtype=np.int32)
    # Assign the cluster label to each valid pixel (using coords)
    for idx, label in enumerate(labels):
        r, c = int(coords[idx, 0]), int(coords[idx, 1])
        segmentation_map[r, c] = label
        
    # Post-process: for each cluster (ignoring noise, i.e. label -1), split by connected components.
    final_segmentation = -1 * np.ones((h, w), dtype=np.int32)
    new_label = 0
    unique_labels = np.unique(segmentation_map)
    for label in unique_labels:
        if label == -1:
            continue
        mask = (segmentation_map == label).astype(np.uint8)
        num_components, comps = cv2.connectedComponents(mask, connectivity=8)
        for comp in range(1, num_components):  # Skip background component 0
            final_segmentation[comps == comp] = new_label
            new_label += 1
    return final_segmentation

def overlay_plane_segmentation(color_img, segmentation_map):
    """
    Overlay segmentation boundaries on the color image.
    
    This function detects the edges between different planar regions and draws them in green.
    """
    overlay = color_img.copy()
    boundaries = np.zeros(segmentation_map.shape, dtype=np.uint8)
    h, w = segmentation_map.shape
    # Simple boundary detection: mark pixel if neighbor label is different.
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

def process_image_with_plane_detection(image_dir, image_id, frame_str, spatial_weight=0.5, eps=0.5, min_samples=50, display_result=True):
    """
    New function to detect physical planes in the image using per-pixel plane estimation and DBSCAN clustering.
    
    It uses your existing functions (load_color_image, load_depth_map, load_normal_map) to load data.
    Then, it computes per-pixel plane features, clusters them with DBSCAN, performs a connected component
    analysis to separate spatially disjoint surfaces, and overlays the resulting plane boundaries on the color image.
    
    Parameters:
      image_dir, image_id, frame_str: Same as in your original functions.
      spatial_weight: Weight for the spatial (2D) coordinates in the feature vector.
      eps: The maximum distance between two samples for DBSCAN.
      min_samples: The minimum number of samples in a neighborhood for DBSCAN.
      display_result: If True, displays the composite image.
      
    Returns:
      composite_with_planes: The color image with overlaid plane segmentation boundaries.
    """
    # Use the same camera views as in process_image
    cam_view_color = "scene_cam_00_final_preview"
    cam_view_geom = "scene_cam_00_geometry_hdf5"
    
    # Load images using your original functions.
    color_img = load_color_image(image_dir, image_id, frame_str, cam_view_color)
    normal_map = load_normal_map(image_dir, image_id, frame_str, cam_view_geom)
    depth_map = load_depth_map(image_dir, image_id, frame_str, cam_view_geom)
    
    if color_img is None or depth_map is None or normal_map is None:
        print(f"Missing data in {image_dir}; skipping plane detection.")
        return None
    
    h, w = color_img.shape[:2]
    # Compute camera intrinsic matrix (same as in process_image)
    fov_x = np.pi / 3 
    f = w / (2 * np.tan(fov_x / 2))
    default_K = np.array([[f, 0, w / 2],
                          [0, f, h / 2],
                          [0, 0, 1]])
    
    # Compute per-pixel plane features.
    features, coords, valid_mask = compute_plane_features(normal_map, depth_map, default_K, spatial_weight=spatial_weight)
    
    # Cluster features to obtain a segmentation map using DBSCAN.
    segmentation_map = cluster_planes(features, coords, (h, w), eps=eps, min_samples=min_samples)
    
    # Overlay the plane segmentation boundaries on the original color image.
    composite_with_planes = overlay_plane_segmentation(color_img, segmentation_map)
    
    if display_result:
        plt.figure(figsize=(10, 10))
        plt.imshow(composite_with_planes)
        plt.title(f"{os.path.basename(image_dir)} - Frame {frame_str}: Plane Detection Overlay")
        plt.axis("off")

        # Ensure the output directory exists
        output_dir = "data/results"
        os.makedirs(output_dir, exist_ok=True)

        # Create a filename for the saved image
        output_filename = f"{os.path.basename(image_dir)}_frame_{frame_str}_overlay.png"
        output_path = os.path.join(output_dir, output_filename)

        # Save the figure without displaying it
        plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
        plt.close()  # Close the figure to free up memory
    
    return composite_with_planes