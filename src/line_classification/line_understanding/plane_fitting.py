import numpy as np
import random


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

def ransac_plane_fit(points,
                     num_iterations=50,
                     threshold=0.03,
                     min_inliers_ratio=0.8):
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

