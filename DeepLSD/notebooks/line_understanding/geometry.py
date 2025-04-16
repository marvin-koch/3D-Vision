import numpy as np
import cv2
import time
def get_line_pixels(line, maps):
    """
    Get all pixel coordinates along a line using cv2.line.
    """
    x1, y1 = map(int, line[0])
    x2, y2 = map(int, line[1])
    height, width = maps.shape[:2]

    blank_image = np.zeros((height, width), dtype=np.uint8)
    cv2.line(blank_image, (x1, y1), (x2, y2), color=255, thickness=1) #TODO it was 3 before, we changed it to 1
    
    y_coords, x_coords = np.where(blank_image == 255)
    return list(zip(x_coords, y_coords))

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
    cv2.line(blank_image, (x1, y1), (x2, y2), color=255, thickness=3) #TODO it was 3 before, we changed it to 1
    
    y_coords, x_coords = np.where(blank_image == 255)
    return list(zip(x_coords, y_coords))

def compute_plane_point(point, normal, dataset="hypersim"):
    """
    Compute plane coefficients from a 3D point and its normal vector.
    """
    denom = np.linalg.norm(normal)
    normal = normal / (denom + 1e-12) # Normalize the normal vector
    x,y,z = point
    a, b, c = normal
    d = -np.dot(normal, point)
    
    # if d < 0:
    #     a *= -1
    #     b *= -1
    #     c *= -1
    #     d *= -1
        
    #d = np.log1p(d)
       
    point_norm = np.linalg.norm([x, y])
    if point_norm > 0:
        x_scaled = x / point_norm
        y_scaled = y / point_norm
    else:
        x_scaled, y_scaled = 0.0, 0.0
    
    if dataset=="hypersim":
        return np.array([a, b, c, d])  # Return plane coefficients
    else:
        return np.array([a, b, c, d ,x_scaled,y_scaled])  # Return plane coefficients

def calculate_plane_for_map(normal_map, world_coordinates, dataset="hypersim"):
    """
    Calculate a plane for every pixel in the normal map using the corresponding world coordinate.
    """
    
    start = time.time()

    plane_map = []
    for y in range(normal_map.shape[0]):
        for x in range(normal_map.shape[1]):
            plane_map.append(compute_plane_point(world_coordinates[y, x], normal_map[y, x], dataset=dataset))
            

    end = time.time()
    length = end - start 
    
    print("Calculating plane for each pixel :", length, "seconds!")

    return np.array(plane_map)
