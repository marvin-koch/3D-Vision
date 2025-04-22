import logging
import cv2
import numpy as np

def compute_normal_map_from_points(
    points: np.ndarray, ksize: int = 3
) -> np.ndarray:
    """
    Compute a normal map from a 3D point cloud using Sobel gradients.

    Args:
        points (np.ndarray): Array of shape (H, W, 3).
        ksize (int): Sobel kernel size.

    Returns:
        np.ndarray: Normal map (H, W, 3) of unit normals (float32).
    """
    if points.ndim != 3 or points.shape[2] != 3:
        logging.error("Invalid points shape %s, expected (H, W, 3).", points.shape)
        h, w = points.shape[:2] if points.ndim >= 2 else (0, 0)
        return np.zeros((h, w, 3), dtype=np.float32)

    pts = np.ascontiguousarray(points, dtype=np.float32)
    gx = np.stack([
        cv2.Sobel(pts[..., c], cv2.CV_32F, 1, 0, ksize=ksize)
        for c in range(3)
    ], axis=-1)
    gy = np.stack([
        cv2.Sobel(pts[..., c], cv2.CV_32F, 0, 1, ksize=ksize)
        for c in range(3)
    ], axis=-1)

    normals = np.cross(gy, gx)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    normals = (normals / norm).astype(np.float32)
    normals[norm.squeeze() == 0] = 0
    return normals


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

