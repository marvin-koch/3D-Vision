import numpy as np
import cv2
import hdbscan
import time
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer

def cluster_coplanar_points(features, world_coordinates, approx_min_span_tree=True, 
        cluster_selection_epsilon=0.01, 
        min_cluster_size=10, 
        allow_single_cluster=False, sample_rate=1, threshold=1):
    """
    Cluster coplanar lines using HDBSCAN.
    """
    
    start = time.time()

    if len(features) == 0:
        return []
    # Create an imputer to fill NaN values with the mean of the column
    if np.any(np.isnan(features)):
        imputer = SimpleImputer(strategy='mean')
        features = imputer.fit_transform(features)
        
        
    h, w = world_coordinates.shape[:2]
    
    print("World_coordinates", h, w)
    n_points = features.shape[0]

    if sample_rate < 1:
        n_sample = int(n_points * sample_rate)
        sample_indices = np.random.choice(n_points, size=n_sample, replace=False)
    else:
        sample_indices = np.arange(n_points)
    
    sample_features = features[sample_indices]
    
    print("start hdbscan")
    sample_labels = hdbscan.HDBSCAN(
        approx_min_span_tree=approx_min_span_tree, 
        cluster_selection_epsilon=cluster_selection_epsilon, 
        min_cluster_size=min_cluster_size, 
        core_dist_n_jobs=-1, 
        allow_single_cluster=allow_single_cluster
        
    ).fit_predict(sample_features)

    end = time.time()
    length = end - start 
    
    print("HDBSCAN :", length, "seconds!")
    start = time.time()

    if sample_rate < 1.0:
        print("start nearestneighbors")

        nbrs = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(sample_features)
        distances, nn_indices = nbrs.kneighbors(features)
        distances = distances.flatten()
        nn_indices = nn_indices.flatten()
        """
        full_labels = np.array([
            sample_labels[idx] if dist <= threshold else -1 
            for idx, dist in zip(nn_indices, distances)
        ])
        
        segmentation_map = full_labels.reshape((h, w))
        """
        full_labels = np.where(distances <= threshold, sample_labels[nn_indices], -1)
    else:
        full_labels = sample_labels
    
    segmentation_map = full_labels.reshape((h, w))
    
    # Post-process: Split spatially disjoint regions within each cluster.
    final_segmentation = -1 * np.ones((h, w), dtype=np.int32)
    new_label = 0
    unique_labels = np.unique(segmentation_map)
    
    print("start dilation")
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

    for label in unique_labels:
        if label == -1:
            continue
        mask = (segmentation_map == label).astype(np.uint8)
        #mask = cv2.medianBlur(mask, 3)

        kernel = kernel_small if np.sum(mask) < 100 else kernel_large # Small clusters = less dilation

        dilated_mask = cv2.dilate(mask, kernel, iterations=3)

        num_components, comps = cv2.connectedComponents(dilated_mask, connectivity=8)
        for comp in range(1, num_components):  # Skip background (component 0)
            final_segmentation[(mask == 1) & (comps == comp)] = new_label

            new_label += 1
    
    end = time.time()
    length = end - start 
    
    print("Dilation :", length, "seconds!")
    return final_segmentation, segmentation_map



def find_line_planes(lines, segmentation_map, get_line_pixels_func):
    """
    For each line, determine the most common plane label by sampling pixels from the segmentation map.
    """
    
    start = time.time()

    
    line_labels = []
    for line in lines:
        pixel_coords = get_line_pixels_func(line, segmentation_map)
        labels = [segmentation_map[y, x] for x, y in pixel_coords]
        
        most_common_label = max(set(labels), key=labels.count)  # Find the most common label
        
        line_labels.append(most_common_label)
        
    
    end = time.time()
    length = end - start 

    print("Find Line Planes :", length, "seconds!")
    
    return line_labels

