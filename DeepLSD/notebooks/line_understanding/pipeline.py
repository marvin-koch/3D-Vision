import os
import matplotlib.pyplot as plt
import numpy as np
from line_understanding.geometry import calculate_plane_for_map, get_line_pixels
from line_understanding.clustering import cluster_coplanar_points, find_line_planes
from line_understanding.image_processing import process_image
from line_understanding.visualization import plot_coplanar_lines, plot_lines_confidence

def process_image_pipeline(image_id, frame_str, net, device, 
                           base_data_dir="data", 
                           depth_thresh=0.05, 
                           normal_thresh=0.5, 
                           thickness=1, 
                           depth_normal_func=lambda x: x.max(), 
                           norm_agg_func=lambda x, axis: np.linalg.norm(x, axis=axis)):
    """
    Process a single image and compute all necessary data.
    
    Returns a dictionary with:
      - composite_after: the image with drawn lines,
      - pred_lines: detected lines,
      - img: original color image,
      - normals: normal map,
      - world_coordinates: world coordinates map,
      - plane_map: computed plane map,
      - segmentation_map: processed segmentation map,
      - original_map: original clustering result,
      - labels: plane labels for each line,
      - image_dir: full path of the image directory.
    """
    # Process image and get additional line_info from process_image.
    image_dir = os.path.join(base_data_dir, image_id)
    composite_after, pred_lines, img, normals, world_coordinates, line_info, scores, isstruct, original_lines = process_image(
        image_dir, image_id, frame_str, net, device,
        depth_thresh, normal_thresh, thickness,
        depth_normal_func=depth_normal_func,
        norm_agg_func=norm_agg_func
    )
    if composite_after is None:
        print(f"Skipping image {image_id} due to missing data.")
        return None

    # Compute the plane map.
    plane_map = calculate_plane_for_map(normals, world_coordinates)

    # Cluster the plane map.
    segmentation_map, original_map = cluster_coplanar_points(
        plane_map, world_coordinates, sample_rate=1, threshold=1
    )

    # Determine coplanarity label for each line.
    all_coplanarity_labels = find_line_planes(pred_lines, segmentation_map, get_line_pixels)

    coplanarity_labels_original_lines = []
    # Update each line_info entry with its corresponding coplanarity labels.
    for entry in line_info:
        indices = entry.pop("new_line_indices")  # Remove indices after use
        if len(indices) == 1:
            coplanarity_labels_original_lines.append([all_coplanarity_labels[indices[0]]])
            entry["coplanarity_labels"] = all_coplanarity_labels[indices[0]]
        else:
            coplanarity_labels_original_lines.append([all_coplanarity_labels[i] for i in indices])
            entry["coplanarity_labels"] = [all_coplanarity_labels[i] for i in indices]

    # Create coplanar matrix.
    N = len(coplanarity_labels_original_lines)
    coplanarity_matrix = np.zeros((N, N), dtype=int)

    for i in range(N):
        for j in range(N):
            for label in coplanarity_labels_original_lines[i]:
                if label in coplanarity_labels_original_lines[j] and label != -1:
                    coplanarity_matrix[i, j] = 1

    # Then return a dictionary that includes line_info:
    return {
        "image_dir": image_dir,
        "composite_after": composite_after,
        "pred_lines": pred_lines,  
        "img": img,
        "normals": normals,
        "world_coordinates": world_coordinates,
        "plane_map": plane_map,
        "segmentation_map": segmentation_map,
        "original_map": original_map,
        "coplanarity_labels": all_coplanarity_labels,  # if needed
        "line_info": line_info,
        "coplanarity_matrix": coplanarity_matrix,
        "scores": scores,
        "original_lines": original_lines    
        
    }




def plot_pipeline_results(processed_data, frame_str):
    """
    Plot the segmentation maps and coplanar lines using the processed data.
    """
    if processed_data is None:
        print("No data to plot.")
        return

    image_dir = processed_data["image_dir"]
    segmentation_map = processed_data["segmentation_map"]
    original_map = processed_data["original_map"]
    img = processed_data["img"]
    pred_lines = processed_data["pred_lines"]
    labels = processed_data["coplanarity_labels"]  # updated key

    # Plot original clustering result.
    plt.figure(figsize=(10, 10))
    plt.imshow(original_map)
    plt.title(f"{os.path.basename(image_dir)} - Frame {frame_str}: Colored Plane Segmentation")
    plt.axis("off")
    plt.show()

    # Plot processed segmentation map.
    plt.figure(figsize=(10, 10))
    plt.imshow(segmentation_map)
    plt.title(f"{os.path.basename(image_dir)} - Frame {frame_str}: Processed Segmentation Map")
    plt.axis("off")
    plt.show()

    # Visualize the coplanar lines.
    plot_coplanar_lines(pred_lines, labels, img)
    
    plot_lines_confidence(img, processed_data["original_lines"], processed_data["scores"])
