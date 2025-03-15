from utils_methods import raydepth2depth, load_color_image, load_depth_map, load_normal_map, compute_variation, sobel_line_neighborhood, overlay_lines_on_image_custom
import os
import numpy as np
import cv2
import torch
import itertools
import numpy as np
import matplotlib.pyplot as plt
from deeplsd.utils.tensor import batch_to_device
from deeplsd.models.deeplsd_inference import DeepLSD
from deeplsd.geometry.viz_2d import plot_lines, plot_images


def process_image(image_dir, image_id, frame_str, net, device,
                  depth_thresh, normal_thresh, thickness,
                  method="neighborhood", depth_normal_func = np.max,
                  depth_normal_func_str = "Max", norm_agg_func=np.sum,
                  struct_color=(0, 0, 255), text_color=(255, 0, 0),):
    """
    Process a single image.

    Parameters:
      image_dir (str): Path to the image folder.
      frame_str (str): Frame number (e.g., "0001").
      net: The DeepLSD model.
      device: Torch device.
      depth_thresh (float): Threshold for depth variation.
      normal_thresh (float): Threshold for normal variation.
      thickness (int): Thickness (for neighborhood method).
      method (str): 'neighborhood' or 'original'. Determines which classification method to use.
      struct_color (tuple): Color for structural lines (default blue).
      text_color (tuple): Color for textural lines (default red).
      display_result (bool): If True, displays the composite image.
      plot_metrics (bool): If True, plots the max variation metrics for each detected line.

    Returns:
      composite (np.array): The composite image with overlaid lines, or None if data is missing.
    """
    # Load the image data
    cam_view_color = "scene_cam_00_final_preview"
    cam_view_geom = "scene_cam_00_geometry_hdf5"

    # Load image
    color_img = load_color_image(image_dir, image_id, frame_str, cam_view_color)
    # Load normal map
    normal_map = load_normal_map(image_dir, image_id, frame_str, cam_view_geom)
    # Load Depth map (DOUBLE CHECK) 
    depth_map = load_depth_map(image_dir, image_id, frame_str, cam_view_geom)
    
    h, w = color_img.shape[:2]
    fov_x = np.pi / 3 
    f = w / (2 * np.tan(fov_x / 2))
    fov_y = 2 * np.arctan(h / (2 * f))
    default_K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]])
    R180x = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

    depth_map = raydepth2depth(depth_map, default_K)


    
    if color_img is None or depth_map is None or normal_map is None:
        print(f"Missing data in {image_dir}; skipping processing.")
        return None

    gray_img = cv2.cvtColor(color_img, cv2.COLOR_RGB2GRAY)

    
    # Detect lines with DeepLSD.
    input_tensor = torch.tensor(gray_img, dtype=torch.float32, device=device)[None, None] / 255.
    with torch.no_grad():
        out = net({'image': input_tensor})
        pred_lines = out['lines'][0]
        if isinstance(pred_lines, torch.Tensor):
            pred_lines = pred_lines.cpu().numpy()
    
    
    
    # Compute Sobel variation maps for depth and normal.
    sobel_depth_map = compute_variation(depth_map, 11)
    sobel_normal_map = compute_variation(normal_map, 27)
    sobel_normal_map = norm_agg_func(sobel_normal_map, axis=2)


    plot_images([sobel_depth_map], ["Depth sobel"], cmaps='gray')
    plot_images([(sobel_normal_map)], ['Normal sobel'], cmaps='gray')

        
        
    # Classify each predicted line and collect max metrics.
    is_struct = []
    depth_max_vals = []  # List to store max depth variation per line.
    normal_max_vals = []  # List to store max normal variation per line.
    
    
    for l in pred_lines:
        ld_neigh, ln_neigh = sobel_line_neighborhood(sobel_depth_map, sobel_normal_map, l, thickness=thickness)
        max_depth = depth_normal_func(ld_neigh)
        max_normal = depth_normal_func(ln_neigh)
        depth_max_vals.append(max_depth)
        normal_max_vals.append(max_normal)
        depth_bool = max_depth > depth_thresh
        normal_bool = max_normal > normal_thresh
        is_struct.append(depth_bool or normal_bool)


    print(f"[{method.capitalize()} Method] {os.path.basename(image_dir)}: Detected {len(pred_lines)} lines; {sum(is_struct)} classified as structural.")

    
    
    # Create a composite image using custom overlay (with custom colors).
    composite = overlay_lines_on_image_custom(color_img, pred_lines, is_struct,
                                               line_color_struct=struct_color,
                                               line_color_text=text_color)
    
    
    
    # Optionally display the composite image.
    if display_result:
        plt.figure(figsize=(10, 10))
        plt.imshow(composite)
        plt.title(f"{os.path.basename(image_dir)} - Frame {frame_str}: Structural vs Textural")
        plt.axis("off")
        plt.show()
    
    
    
    # Optionally, plot the max variation metrics.
    if plot_metrics and len(depth_max_vals) > 0:
        plt.figure(figsize=(8, 6))
        # Plot structural lines in one color and textural in another.
        for i in range(len(depth_max_vals)):
            if is_struct[i]:
                plt.scatter(depth_max_vals[i], normal_max_vals[i], c='blue', marker='o', label='Structural' if i==0 else "")
            else:
                plt.scatter(depth_max_vals[i], normal_max_vals[i], c='red', marker='x', label='Textural' if i==0 else "")
        plt.xlabel(f"{depth_normal_func_str} Depth Variation")
        plt.ylabel(f"{depth_normal_func_str} Normal Variation")
        plt.title(f"{os.path.basename(image_dir)} - Frame {frame_str} - Thickness {thickness}: Variation Metrics")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    return composite




#****************************************************************************************************
#****************************************************************************************************
#****************************************************************************************************
#****************************************************************************************************
#****************************************************************************************************
#****************************************************************************************************




def plot_process(image_ids, base_data_dir, frame_str, depth_thresholds, normal_thresholds,
            thickness_values, line_detection_model, device="cpu", n_columns=2, depth_normal_func=np.max, depth_normal_func_str="Max",
            norm_agg_func=np.sum):

    for image_id in image_ids:
        image_dir = os.path.join(base_data_dir, image_id)
        print(f"\n[Parameter Comparison] Processing image: {image_id} for frame {frame_str}")

        # Lists to store the composite images and their corresponding titles
        composite_images = []
        titles = []
        plot = True # Only plot the first time
        # Iterate over the Cartesian product of all parameter arrays
        for d_thresh, n_thresh, t_val in itertools.product(depth_thresholds, normal_thresholds, thickness_values):
            # Process the image with the current set of parameters

            composite = process_image(
                image_dir, image_id, frame_str, line_detection_model, device,
                depth_thresh=d_thresh,
                normal_thresh=n_thresh,
                thickness=t_val,
                method="neighborhood",
                depth_normal_func=depth_normal_func,
                depth_normal_func_str=depth_normal_func_str,
                norm_agg_func=norm_agg_func,
                struct_color=(0, 0, 255),  # blue
                text_color=(255, 0, 0),    # red
                display_result=False,
                plot_metrics=plot
            )
            plot=False # no need to plot more than once

            if composite is not None:
                composite_images.append(composite)
                titles.append(f"d={d_thresh}, n={n_thresh}, t={t_val}")

        # If we got any composite images, create a grid for comparison
        n_composites = len(composite_images)
        if n_composites > 0:
            # Define the grid with 2 columns
            n_cols = n_columns
            n_rows = int(np.ceil(n_composites / n_cols))
            fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))
            # Flatten the axes array in case it is multidimensional.
            axs = np.array(axs).flatten()
            for i, comp in enumerate(composite_images):
                axs[i].imshow(comp)
                axs[i].set_title(titles[i])
                axs[i].axis("off")
            # Hide any unused subplots.
            for j in range(n_composites, len(axs)):
                axs[j].axis("off")
            plt.suptitle(f"{image_id} - Frame {frame_str} Parameter Comparison", fontsize=16)
            plt.tight_layout()
            plt.show()

            # save the grid of composite images.
            grid_out_path = os.path.join(image_dir, f"parameter_comparison_{frame_str}.png")
            fig.savefig(grid_out_path)
            print(f"Saved parameter comparison grid to {grid_out_path}")
