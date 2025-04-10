import os
import cv2
import numpy as np
import torch 
import time

from numpy import linalg as LA
import matplotlib.pyplot as plt

from .utility_methods import (
    raydepth2depth, load_color_image, load_depth_map, load_depth_map_png, calculate_normal_map_from_depth, reconstruct_3d_from_depth,
    load_normal_map, load_world_coordinates, compute_variation, sobel_line, sigmoid, load_intrinsics_json, load_color_image_moge_gt, compute_variation_laplace,
    compute_normal_map_from_world
)
from deeplsd.geometry.viz_2d import plot_images
def quantize_normals(normals, bins=10):
    """
    Quantize unit normals into discrete bins over the sphere.
    
    Parameters:
      normals : numpy.ndarray
          A [H, W, 3] array of unit normals.
      bins : int
          Number of bins per axis (total number of bins = bins^2).
          
    Returns:
      quantized : numpy.ndarray
          A [H, W] array containing quantized bin indices.
    """
    # Extract components of the normals
    x, y, z = normals[..., 0], normals[..., 1], normals[..., 2]
    

    # Convert Cartesian to spherical coordinates:
    # theta: polar angle (0 to pi)
    theta = np.arccos(np.clip(z, -1, 1))
    # phi: azimuth (from -pi to pi)
    phi = np.arctan2(y, x)

    # Quantize theta and phi into discrete bins:
    # Multiply by bins and then use floor to get the bin index.
    theta_bin = np.floor(theta / np.pi * bins).astype(int)
    phi_bin   = np.floor((phi + np.pi) / (2 * np.pi) * bins).astype(int)

    # Clip indices to ensure they are within [0, bins-1]
    theta_bin = np.clip(theta_bin, 0, bins - 1)
    phi_bin   = np.clip(phi_bin, 0, bins - 1)

    # Combine the two bin indices into a single index:
    quantized = theta_bin * bins + phi_bin
    return quantized


import numpy as np

def dequantize_normals(quantized, bins=10):
    """
    Convert quantized bin indices back into approximate normal vectors.
    
    Parameters:
      quantized : numpy.ndarray
          A [H, W] array of quantized bin indices.
      bins : int
          Number of bins per axis used in the original quantization.
          
    Returns:
      normals : numpy.ndarray
          A [H, W, 3] array of approximate unit normals reconstructed from the bins.
    """
    # Recover the theta and phi bin indices from the combined quantized value.
    theta_bin = quantized // bins
    phi_bin = quantized % bins

    # Compute the center of each bin in spherical coordinates.
    # Each theta bin covers an interval of pi / bins, so we take the center value.
    theta = (theta_bin + 0.5) * (np.pi / bins)
    # Each phi bin covers an interval of (2*pi) / bins; adjust for the offset of -pi.
    phi = (phi_bin + 0.5) * (2 * np.pi / bins) - np.pi

    # Convert the spherical coordinates back to Cartesian coordinates.
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    # Stack the Cartesian coordinates into a [H, W, 3] array.
    normals = np.stack([x, y, z], axis=-1)
    return normals

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




def compute_shifted_line(line, depth_map, w, h, offset=1.0, num_samples=100):
    """
    Computes a shifted version of the input line by moving it in the direction 
    (perpendicular to the line) where the average depth is lower.
    """
    # Compute the unit tangent vector.
    tangent = line[1] - line[0]
    t_norm = tangent / np.linalg.norm(tangent)
    # Compute a perpendicular direction and scale it by the offset.
    perp = np.array([-t_norm[1], t_norm[0]]) * offset
    
    # Generate sample points along the line.
    xs = np.linspace(line[0, 0], line[1, 0], num_samples)
    ys = np.linspace(line[0, 1], line[1, 1], num_samples)
    points = np.stack([xs, ys], axis=1)

    depth_pos = []
    depth_neg = []
    # For each sample point, compute the depth on both sides.
    for pt in points:
        # Positive offset side.
        x_pos = int(round(pt[0] + perp[0]))
        y_pos = int(round(pt[1] + perp[1]))
        x_pos = np.clip(x_pos, 0, w - 1)
        y_pos = np.clip(y_pos, 0, h - 1)
        depth_pos.append(depth_map[y_pos, x_pos])
        
        # Negative offset side.
        x_neg = int(round(pt[0] - perp[0]))
        y_neg = int(round(pt[1] - perp[1]))
        x_neg = np.clip(x_neg, 0, w - 1)
        y_neg = np.clip(y_neg, 0, h - 1)
        depth_neg.append(depth_map[y_neg, x_neg])
    
    avg_depth_pos = np.mean(depth_pos)
    avg_depth_neg = np.mean(depth_neg)
    # Choose the offset direction with lower average depth.
    chosen_offset = perp if avg_depth_pos < avg_depth_neg else -perp
    shifted_line = line + chosen_offset
    return shifted_line

def normalize_img(img):
    """Normalize image to [0, 1] range."""
    return (img - img.min()) / (img.max() - img.min())

def process_image(image_dir, image_id, frame_str, net, device,
            depth_thresh=125, normal_thresh=1.25 * 1e7, thickness=1, structural_thresh=0.6,
            method="neighborhood", normal_func=np.max, depthfunc=np.max,
            depth_normal_func_str="Max", norm_agg_func=np.linalg.norm,
            struct_color=(0, 0, 255), text_color=(255, 0, 0), dataset="hypersim"):

    # Load image data using helper functions.
    cam_view_color = "scene_cam_00_final_preview"
    cam_view_geom = "scene_cam_00_geometry_hdf5"
    gt = "gt"
    moge = "moge"
    
    start = time.time()


    if dataset == "hypersim":
        color_img = load_color_image(image_dir, image_id, frame_str, cam_view_color)
        h, w = color_img.shape[:2]

        normal_map = load_normal_map(image_dir, image_id, frame_str, cam_view_geom)
        depth_map = load_depth_map(image_dir, image_id, frame_str, cam_view_geom)
        fov_x = np.pi / 3 
        f = w / (2 * np.tan(fov_x / 2))
        default_K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
        
        if default_K[0, 0] < 1:  # heuristic check
            print("correct K")
            default_K[0, 0] *= w  # fx
            default_K[0, 2] *= w  # cx
            default_K[1, 1] *= h  # fy
            default_K[1, 2] *= h  # cy   
            
        depth_map = raydepth2depth(depth_map, default_K)
        world_coordinates_map = load_world_coordinates(image_dir, image_id, frame_str, cam_view_geom)
        
    elif dataset == "moge_gt":
        """
        color_img = load_color_image_moge_gt(image_dir)
        h, w = color_img.shape[:2]

        depth_map = load_depth_map_png(image_dir)
        normal_map = calculate_normal_map_from_depth(depth_map, ksize=11)
        default_K = load_intrinsics_json(image_dir)
        depth_map = raydepth2depth(depth_map, default_K)

        world_coordinates_map = reconstruct_3d_from_depth(depth_map, default_K)
        """
        color_img = load_color_image_moge_gt(image_dir)
        h, w = color_img.shape[:2]
        color_img = cv2.resize(color_img, dsize=(int(w/2), int(h/2)),interpolation=cv2.INTER_CUBIC)
        depth_map = load_depth_map_png(image_dir)
        depth_map = cv2.resize(depth_map,dsize=(int(w/2), int(h/2)),interpolation=cv2.INTER_CUBIC)
        
        h,w = color_img.shape[:2]

        plot_images([normalize_img(depth_map)], ["Depth Original"], cmaps='gray')

        # d = 15 # e.g., 5, 9, 15 - adjust based on noise scale

        # sigmaColor = 25

        # sigmaSpace = 25 # Often similar to d/2 or d

        depth_map_cleaned = cv2.medianBlur(depth_map, 5)
        
        plot_images([normalize_img(depth_map_cleaned)], ["Depth Cleaned"], cmaps='gray')

        default_K = load_intrinsics_json(image_dir)
        
        if default_K[0, 0] < 1:  # heuristic check
            default_K[0, 0] *= w  # fx
            default_K[0, 2] *= w  # cx
            default_K[1, 1] *= h  # fy
            default_K[1, 2] *= h  # cy   
                 
        #depth_map = raydepth2depth(depth_map, default_K)
        world_coordinates_map_def = reconstruct_3d_from_depth(depth_map, default_K)

        world_coordinates_map = reconstruct_3d_from_depth(depth_map_cleaned, default_K)
        
        normal_map_def = compute_normal_map_from_world(world_coordinates_map_def, ksize=1)

        normal_map = compute_normal_map_from_world(world_coordinates_map, ksize=1)
        
        
        normal_map_cleaned = cv2.medianBlur(normal_map.astype(np.float32), 5)


        norms = np.linalg.norm(normal_map_cleaned, axis=2, keepdims=True)
        norms[norms < 1e-6] = 1.0
        normal_map_cleaned = normal_map_cleaned / norms
                
                
        #normal_map_cleaned = dequantize_normals(quantize_normals(normal_map_def, bins=10),bins=10)
        
        
        #normal_map = calculate_normal_map_from_depth(depth_map.astype(np.float32), ksize=1)
       
        # # Convert depth to float32 if it isn't already.
        # depth_map_float = depth_map.astype(np.float32)

        # param = {"d": 15, "sigmaColor": 25, "sigmaSpace": 25}

        # d = param["d"]
        # sigmaColor = param["sigmaColor"]
        # sigmaSpace = param["sigmaSpace"]
        
        # #Apply bilateral filter with the current parameters.
        # depth_map_filtered = cv2.bilateralFilter(depth_map_float, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)

         
        print(color_img.shape)
        print(depth_map.shape)
        print(world_coordinates_map.shape)
        print(normal_map.shape)
        # Plot the filtered depth maps side by side.
        #plot_images([normalize_img(depth_map_filtered)], ["Filtered depth"], cmaps='gray')
        
        normal_map_viz = (normal_map_cleaned + 1) / 2
        plot_images([(normal_map_def + 1) / 2], ["Original Normal map"], cmaps='gray')

        plot_images([(normal_map + 1) / 2], ["Normal map with cleaned depth"], cmaps='gray')

        plot_images([normal_map_viz], ["Normal map with cleaned depth and bins"], cmaps='gray')
       
        
    if color_img is None or depth_map is None or normal_map is None:
        print(f"Missing data in {image_dir}; skipping processing.")
        return None, None, None, None, None, None, None, None, None



    gray_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)
    
    
    end = time.time()
    length = end - start 
    start = time.time()

    print("Loading :", length, "seconds!")

    # Detect lines with DeepLSD.
    input_tensor = torch.tensor(gray_img, dtype=torch.float32, device=device)[None, None] / 255.
    with torch.no_grad():
        out = net({'image': input_tensor})
        pred_lines = out['lines'][0]
        if isinstance(pred_lines, torch.Tensor):
            pred_lines = pred_lines.cpu().numpy()
            
   
    end = time.time()
    length = end - start 

    print("DeepLSD :", length, "seconds!")
    start = time.time()

    # Compute variation maps.
    sobel_depth_map = compute_variation_laplace(depth_map,11, depth=True)
    sobel_normal_map = compute_variation(normal_map, 27)
    sobel_normal_map = norm_agg_func(sobel_normal_map, axis=2)
    
    plot_images([sobel_depth_map], ["Depth sobel"], cmaps='gray')
    plot_images([sobel_normal_map], ["Normal sobel"], cmaps='gray')
     
    end = time.time()
    length = end - start 
    print("Sobel :", length, "seconds!")
    start = time.time()

    # Classify each predicted line.
    is_struct = []
    is_depth_seperated  = []
    scores = []
    max_depths = []
    max_normals = []

    sobel_depth_map = np.nan_to_num(sobel_depth_map, nan=0.0)
    sobel_normal_map = np.nan_to_num(sobel_normal_map, nan=0.0)

    for l in pred_lines:
        ld, ln = sobel_line(sobel_depth_map, sobel_normal_map, l, trim_ratio=0.25)
   
        max_depth = depthfunc(ld)
        max_normal = normal_func(ln)
        

                
        depth_sigmoid = sigmoid(max_depth, lam=0.01, tau=depth_thresh)
        normal_sigmoid = sigmoid(max_normal, lam=0.01, tau=normal_thresh)
        scores.append(max(normal_sigmoid, depth_sigmoid))

        
        max_depths.append(max_depth)
        max_normals.append(max_normal)
   
        is_depth_seperated.append(depth_sigmoid > 0.5)
    
    print("Print Scores")
    print("Scores", scores)
    # Sort the list
    sorted_values_depth = sorted(max_depths)
    plt.figure()
    plt.scatter(range(len(sorted_values_depth)), sorted_values_depth, color='b', marker='o')
    plt.axhline(y=depth_thresh, color='r', linestyle='--')

    plt.xlabel('Rank')
    plt.ylabel('Value')
    plt.title('Scatter Plot of Sorted Max Depths')
    plt.show()
    
    
    sorted_values_normal = sorted(max_normals)
    plt.figure()
    plt.scatter(range(len(sorted_values_normal)), sorted_values_normal, color='b', marker='o')
    plt.axhline(y=normal_thresh, color='r', linestyle='--')

    plt.xlabel('Rank')
    plt.ylabel('Value')
    plt.title('Scatter Plot of Sorted Max Normals')
    plt.show()

    sorted_values_scores = sorted(scores)
    plt.figure()
    plt.scatter(range(len(sorted_values_scores)), sorted_values_scores, color='r', marker='o')
    plt.xlabel('Rank')
    plt.ylabel('Value')
    plt.title('Scatter Plot of Sorted Scores')
    plt.show()

    is_struct = [s > structural_thresh for s in scores]
    
    print(f"[{method.capitalize()} Method] {os.path.basename(image_dir)}: Detected {len(pred_lines)} lines; {sum(is_struct)} structural.")


     
    end = time.time()
    length = end - start 
    print("Struct vs Textural :", length, "seconds!")
    start = time.time()


    non_structural_color = text_color
    structural_color1 = (128, 0, 128)  # purple
    structural_color2 = (0, 165, 255)   # orange

    composite_after = color_img.copy()
    new_lines_list = []  # List of all drawn lines (offsets for structural, original for textural)
    line_info = []       # Metadata for each base line

    for i, l in enumerate(pred_lines):
        line = l.reshape(2, 2) if l.shape == (4,) else l

        # Case 1: Structural line in low depth variation => split
        if is_struct[i] and not is_depth_seperated[i]:
            offset_amount = 1.0
            line1, line2 = create_optimal_offset_lines(line, normal_map, offset_amount=offset_amount)
            idx1 = len(new_lines_list)
            new_lines_list.append(line1)
            idx2 = len(new_lines_list)
            new_lines_list.append(line2)
            line_info.append({
                "base_line": line.tolist(),
                "score": scores[i],
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

        # Case 2: Structural line in high depth variation => shift by 1 pixel
        elif is_struct[i] and is_depth_seperated[i]:
            shifted_line = compute_shifted_line(line, depth_map, w, h, offset=1.0, num_samples=100)
            idx = len(new_lines_list)
            new_lines_list.append(shifted_line)
            line_info.append({
                "base_line": line.tolist(),
                "score": scores[i],
                "new_line_indices": [idx],
                "shifted": True
            })
            cv2.line(composite_after,
                     (int(round(shifted_line[0, 0])), int(round(shifted_line[0, 1]))),
                     (int(round(shifted_line[1, 0])), int(round(shifted_line[1, 1]))),
                     struct_color, thickness)

        # Case 3: Textural lines => let as is
        else:
            idx = len(new_lines_list)
            new_lines_list.append(line)
            line_info.append({
                "base_line": line.tolist(),
                "score": scores[i],
                "new_line_indices": [idx]
            })
            cv2.line(composite_after,
                     (int(round(line[0, 0])), int(round(line[0, 1]))),
                     (int(round(line[1, 0])), int(round(line[1, 1]))),
                     non_structural_color, thickness)


    end = time.time()
    length = end - start 
    
    print("Splitting Lines :", length, "seconds!")
    
    
    composite_after_rgb = cv2.cvtColor(composite_after, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(composite_after_rgb)
    plt.axis("off")
    plt.show()

    new_lines_array = np.array(new_lines_list)
    
    
    
    return composite_after, new_lines_array, color_img, normal_map_cleaned, world_coordinates_map, line_info, scores, is_struct, pred_lines
