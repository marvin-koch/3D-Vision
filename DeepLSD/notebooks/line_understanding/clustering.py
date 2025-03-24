import numpy as np
import cv2
import hdbscan
from sklearn.neighbors import NearestNeighbors

def cluster_coplanar_points(features, world_coordinates, eps=0.5, min_samples=5, sample_rate=1, threshold=1):
    """
    Cluster coplanar lines using HDBSCAN.
    """
    if len(features) == 0:
        return []

    h, w = world_coordinates.shape[:2]
    n_points = features.shape[0]

    if sample_rate < 1.0:
        n_sample = int(n_points * sample_rate)
        sample_indices = np.random.choice(n_points, size=n_sample, replace=False)
    else:
        sample_indices = np.arange(n_points)
    
    sample_features = features[sample_indices]
    sample_labels = hdbscan.HDBSCAN(
        approx_min_span_tree=False, 
        cluster_selection_epsilon=0.01, 
        min_cluster_size=10, 
        core_dist_n_jobs=-1, 
        allow_single_cluster=False
    ).fit_predict(sample_features)
    
    nbrs = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(sample_features)
    distances, nn_indices = nbrs.kneighbors(features)
    distances = distances.flatten()
    nn_indices = nn_indices.flatten()
    
    full_labels = np.array([
        sample_labels[idx] if dist <= threshold else -1 
        for idx, dist in zip(nn_indices, distances)
    ])
    
    segmentation_map = full_labels.reshape((h, w))
    
    # Post-process: Split spatially disjoint regions within each cluster.
    final_segmentation = -1 * np.ones((h, w), dtype=np.int32)
    new_label = 0
    unique_labels = np.unique(segmentation_map)
    
    for label in unique_labels:
        if label == -1:
            continue
        mask = (segmentation_map == label).astype(np.uint8)
        kernel_size = 1 if np.sum(mask) < 100 else 5  # Smaller clusters get less dilation.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dilated_mask = cv2.dilate(mask, kernel, iterations=3)
        num_components, comps = cv2.connectedComponents(dilated_mask, connectivity=8)
        for comp in range(1, num_components):  # Skip background.
            final_segmentation[(mask == 1) & (comps == comp)] = new_label
            new_label += 1
            
    return final_segmentation, segmentation_map



def find_line_planes(lines, segmentation_map, get_line_pixels_func):
    """
    For each line, determine the most common plane label by sampling pixels from the segmentation map.
    """
    line_labels = []
    for line in lines:
        pixel_coords = get_line_pixels_func(line, segmentation_map)
        labels = [segmentation_map[y, x] for x, y in pixel_coords]
        most_common_label = max(set(labels), key=labels.count)
        line_labels.append(most_common_label)
    return line_labels
