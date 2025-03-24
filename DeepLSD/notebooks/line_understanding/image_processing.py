import os
import cv2
import numpy as np
import torch
from numpy import linalg as LA
import matplotlib.pyplot as plt
from line_understanding.utility_methods import (
    raydepth2depth, load_color_image, load_depth_map,
    load_normal_map, load_world_coordinates, compute_variation, sobel_line_neighborhood
)


def create_optimal_offset_lines(line, normal_map, offset_amount=1.0, num_samples=100, angle_steps=36):
    """
    For a given structural line and normal map, search for the best offset direction.
    Returns a pair of offset lines that maximize the difference between normals.
    """
    p1, p2 = line
    xs = np.linspace(p1[0], p2[0], num_samples)
    ys = np.linspace(p1[1], p2[1], num_samples)
    best_score = -np.inf
    best_d = None
    for theta in np.linspace(0, np.pi, angle_steps, endpoint=False):
        d = np.array([np.cos(theta), np.sin(theta)])
        pts1 = np.stack([xs, ys], axis=1) + offset_amount * d
        pts2 = np.stack([xs, ys], axis=1) - offset_amount * d
        H, W = normal_map.shape[:2]
        normals1 = []
        normals2 = []
        for pt in pts1:
            x = int(round(pt[0]))
            y = int(round(pt[1]))
            x = np.clip(x, 0, W-1)
            y = np.clip(y, 0, H-1)
            normals1.append(normal_map[y, x, :])
        for pt in pts2:
            x = int(round(pt[0]))
            y = int(round(pt[1]))
            x = np.clip(x, 0, W-1)
            y = np.clip(y, 0, H-1)
            normals2.append(normal_map[y, x, :])
        normals1 = np.array(normals1)
        normals2 = np.array(normals2)
        diffs = np.linalg.norm(normals1 - normals2, axis=1)
        score = np.sum(diffs)
        if score > best_score:
            best_score = score
            best_d = d
    new_line1 = np.array([p1 + offset_amount * best_d, p2 + offset_amount * best_d])
    new_line2 = np.array([p1 - offset_amount * best_d, p2 - offset_amount * best_d])
    return new_line1, new_line2




def process_image(image_dir, image_id, frame_str, net, device,
                  depth_thresh, normal_thresh, thickness,
                  method="neighborhood", depth_normal_func=np.max,
                  depth_normal_func_str="Max", norm_agg_func=np.sum,
                  struct_color=(0, 0, 255), text_color=(255, 0, 0)):
    """
    Process a single image:
      - Load image data.
      - Run line detection using DeepLSD.
      - Compute variation maps.
      - Classify lines as structural or non-structural.
      - Draw structural lines (with optimal offsets) and non-structural lines.
    
    Returns:
      composite_after: the image with drawn lines,
      new_lines_array: an array of new (drawn) lines,
      color_img, normal_map, world_coordinates_map: raw data,
      line_info: a list of dictionaries containing per-line metadata.
    """
    # Load image data using helper functions.
    cam_view_color = "scene_cam_00_final_preview"
    cam_view_geom = "scene_cam_00_geometry_hdf5"
    color_img = load_color_image(image_dir, image_id, frame_str, cam_view_color)
    normal_map = load_normal_map(image_dir, image_id, frame_str, cam_view_geom)
    depth_map = load_depth_map(image_dir, image_id, frame_str, cam_view_geom)
    world_coordinates_map = load_world_coordinates(image_dir, image_id, frame_str, cam_view_geom)
    
    if color_img is None or depth_map is None or normal_map is None:
        print(f"Missing data in {image_dir}; skipping processing.")
        return None, None, None, None, None, None

    h, w = color_img.shape[:2]
    fov_x = np.pi / 3 
    f = w / (2 * np.tan(fov_x / 2))
    default_K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    
    depth_map = raydepth2depth(depth_map, default_K)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
    
    # Detect lines with DeepLSD.
    input_tensor = torch.tensor(gray_img, dtype=torch.float32, device=device)[None, None] / 255.
    with torch.no_grad():
        out = net({'image': input_tensor})
        pred_lines = out['lines'][0]
        if isinstance(pred_lines, torch.Tensor):
            pred_lines = pred_lines.cpu().numpy()
    
    # Compute variation maps.
    sobel_depth_map = compute_variation(depth_map, 11)
    sobel_normal_map = compute_variation(normal_map, 27)
    sobel_normal_map = norm_agg_func(sobel_normal_map, axis=2)
        
    # Classify each predicted line.
    is_struct = []
    for l in pred_lines:
        line_for_class = l.reshape(2, 2) if l.shape == (4,) else l
        ld_neigh, ln_neigh = sobel_line_neighborhood(sobel_depth_map, sobel_normal_map, line_for_class, thickness=thickness)
        max_depth = depth_normal_func(ld_neigh)
        max_normal = depth_normal_func(ln_neigh)
        depth_bool = max_depth > depth_thresh
        normal_bool = max_normal > normal_thresh
        is_struct.append(depth_bool or normal_bool)

    print(f"[{method.capitalize()} Method] {os.path.basename(image_dir)}: Detected {len(pred_lines)} lines; {sum(is_struct)} structural.")

    non_structural_color = text_color
    structural_color1 = (128, 0, 128)  # purple
    structural_color2 = (0, 165, 255)   # orange

    composite_after = color_img.copy()
    new_lines_list = []  # List of all drawn lines (offsets for structural, original for textural)
    line_info = []       # Metadata for each base line

    for i, l in enumerate(pred_lines):
        line = l.reshape(2, 2) if l.shape == (4,) else l
        if is_struct[i]:
            offset_amount = 1.0
            line1, line2 = create_optimal_offset_lines(line, normal_map, offset_amount=offset_amount)
            # Record indices of the offset lines in new_lines_list
            idx1 = len(new_lines_list)
            new_lines_list.append(line1)
            idx2 = len(new_lines_list)
            new_lines_list.append(line2)
            # Record one entry for the base line with its two offsets
            line_info.append({
                "base_line": line.tolist(),
                "type": "structural",
                "offset_lines": [line1.tolist(), line2.tolist()],
                "new_line_indices": [idx1, idx2]
            })
            new_thickness = thickness + 1
            cv2.line(composite_after,
                     (int(round(line1[0, 0])), int(round(line1[0, 1]))),
                     (int(round(line1[1, 0])), int(round(line1[1, 1]))),
                     structural_color1, new_thickness)
            cv2.line(composite_after,
                     (int(round(line2[0, 0])), int(round(line2[0, 1]))),
                     (int(round(line2[1, 0])), int(round(line2[1, 1]))),
                     structural_color2, new_thickness)
        else:
            idx = len(new_lines_list)
            new_lines_list.append(line)
            line_info.append({
                "base_line": line.tolist(),
                "type": "textural",
                "new_line_indices": [idx]
            })
            cv2.line(composite_after,
                     (int(round(line[0, 0])), int(round(line[0, 1]))),
                     (int(round(line[1, 0])), int(round(line[1, 1]))),
                     non_structural_color, thickness)

    new_lines_array = np.array(new_lines_list)
    return composite_after, new_lines_array, color_img, normal_map, world_coordinates_map, line_info

