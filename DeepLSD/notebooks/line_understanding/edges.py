import cv2
import numpy as np

from typing import Any, Optional, Tuple, List, Dict


def compute_variation_laplace(mapping, k, depth=False):
    """
    Computes the Sobel variation of a mapping (depth or normal) using a kernel size k.
    Normalizes the result by subtracting the mean and dividing by the standard deviation.
    """
   
    laplacian = cv2.Laplacian(mapping, ddepth=-1, ksize=k)
    
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



def compute_edge_maps(
    normal_map: np.ndarray,
    depth_map: np.ndarray,
    thresh_n: float,
    thresh_d: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute sobel and thresholded edge maps for normal and depth.
    """
    sobel_n = np.linalg.norm(compute_variation(normal_map, k=27), axis=2)
    tn = threshold_edges(sobel_n, thresh_n)
    tn = cv2.convertScaleAbs(tn)
    tn = cv2.morphologyEx(tn, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    tn = cv2.erode(tn, np.ones((3,3), np.uint8), iterations=2)

    sobel_d = compute_variation_laplace(depth_map, k=3, depth=True)
    td = threshold_edges(sobel_d, thresh_d)
    td = cv2.convertScaleAbs(td)
    td = cv2.morphologyEx(td, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    td = cv2.dilate(td, np.ones((3,3), np.uint8), iterations=3)

    combined = cv2.bitwise_or(tn, td)
    combined = np.nan_to_num(combined, nan=0.0)
    return sobel_n, tn, sobel_d, td, combined
