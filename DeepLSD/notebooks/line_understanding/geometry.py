import numpy as np
import cv2

def get_line_pixels(line, maps):
    """
    Get all pixel coordinates along a line using cv2.line.
    """
    x1, y1 = map(int, line[0])
    x2, y2 = map(int, line[1])
    height, width = maps.shape[:2]

    blank_image = np.zeros((height, width), dtype=np.uint8)
    cv2.line(blank_image, (x1, y1), (x2, y2), color=255, thickness=3)
    
    y_coords, x_coords = np.where(blank_image == 255)
    return list(zip(x_coords, y_coords))

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
    return np.array(plane_map)
