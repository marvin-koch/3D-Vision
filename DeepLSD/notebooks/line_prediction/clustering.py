import numpy as np
import cv2
import hdbscan
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer

def cluster_coplanar_points(features, world_coordinates, valid_mask,  approx_min_span_tree=False, cluster_selection_epsilon=0.01, 
        min_cluster_size=10, 
        allow_single_cluster=False, sample_rate=1, threshold=1):
    """
    Cluster coplanar lines using DBSCAN.
    """
    if len(features) == 0:
        return []

    # Feature vector: (plane normal x, y, z, centroid x, y, z)
    # feature_matrix = np.array([np.hstack((plane[:3], centroid)) for plane, centroid, _ in line_features])
    # DBSCAN Clustering
    h,w = world_coordinates.shape[:2]
    n_points = features.shape[0]

    if sample_rate < 1.0:
        n_sample = int(n_points * sample_rate)
        sample_indices = np.random.choice(n_points, size=n_sample, replace=False)
    else:
        sample_indices = np.arange(n_points)
    

    sample_features = features[valid_mask > 0]

    sample_features = sample_features.reshape(-1, 4)
    features = features.reshape(-1,4)

    #0.17
    #sample_labels = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean', algorithm="ball_tree").fit_predict(sample_features)
    sample_labels = hdbscan.HDBSCAN(
        approx_min_span_tree=approx_min_span_tree, 
        cluster_selection_epsilon=cluster_selection_epsilon, 
        min_cluster_size=min_cluster_size, 
        core_dist_n_jobs=-1, 
        allow_single_cluster=allow_single_cluster
    ).fit_predict(sample_features)
    
    nbrs = NearestNeighbors(n_neighbors=1, metric="euclidean").fit(sample_features)
    distances, nn_indices = nbrs.kneighbors(features)
    distances = distances.flatten()
    nn_indices = nn_indices.flatten()
    
    # Assign label
    full_labels = np.array([sample_labels[idx] if dist <= threshold else -1           # ***************************
                              for idx, dist in zip(nn_indices, distances)])
    
    # h, w = world_coordinates.shape
    # segmentation_map = -1 * np.ones((h, w), dtype=np.int32)
    # for i, label in enumerate(full_labels):
    #     r, c = int(coords[i, 0]), int(coords[i, 1])
    #     segmentation_map[r, c] = label
    segmentation_map = full_labels.reshape((h,w))
    
    # Post-process: for each cluster (ignoring noise), use connected components to split spatially disjoint regions.
    final_segmentation = -1 * np.ones((h, w), dtype=np.int32)
    new_label = 0
    unique_labels = np.unique(segmentation_map)

    for label in unique_labels:
        if label == -1:
            continue
        mask = (segmentation_map == label).astype(np.uint8)
        #mask = cv2.medianBlur(mask, 3)

        kernel_size = 1 if np.sum(mask) < 100 else 5  # Small clusters = less dilation
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        dilated_mask = cv2.dilate(mask, kernel, iterations=3)

        num_components, comps = cv2.connectedComponents(dilated_mask, connectivity=8)
        for comp in range(1, num_components):  # Skip background (component 0)
            final_segmentation[(mask == 1) & (comps == comp)] = new_label

            new_label += 1
            

    return final_segmentation, segmentation_map



def find_line_planes(lines, segmentation_map,valid_mask, get_line_pixels_func):
 
    line_labels = []
    for line in lines:
        pixel_coords = get_line_pixels_func(line, valid_mask, segmentation_map)

        labels = []
        for x, y in pixel_coords:
            labels.append(segmentation_map[y, x])
            
        if len(labels) == 0:
            print("No labels found for line")
            line_labels.append(-1)
        else: 
            most_common_label = max(set(labels), key=labels.count)
            line_labels.append(most_common_label)

    return line_labels