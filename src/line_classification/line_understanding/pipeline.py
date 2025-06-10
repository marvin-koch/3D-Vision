import os
import time
from pathlib import Path
from typing import Any, Optional, Tuple, List, Dict

import cv2
import numpy as np
import torch

from line_understanding.visualization import *
from line_understanding.save import save_lines_to_json
from line_understanding.dataloader import HypersimLoader, ETH3DLoader, ScanNetLoader
from line_understanding.edges import *
from line_understanding.geometry import *
from line_understanding.plane_fitting import *
from line_understanding.lines import *


def load_data(
    image_id: str,
    frame_str: str,
    dataset: str,
    data_root: Path
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load color image, depth map, world coordinates, and normal map.
    """
    data_dir = data_root / image_id
    if dataset == "hypersim":
        loader = HypersimLoader(data_dir)
        color_img = loader.load_color_image(image_id, frame_str, "scene_cam_00_final_preview")
        depth_map = loader.load_depth(image_id, frame_str, "scene_cam_00_geometry_hdf5")
        h, w = depth_map.shape
        fov_x = np.pi / 3
        f = w / (2 * np.tan(fov_x / 2))
        K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]])
        depth_map = loader.raydepth2depth(depth_map, K)
        world_coords = reproject_depth_to_points(depth_map, K)
        normal_map = loader.load_normal(image_id, frame_str, "scene_cam_00_geometry_hdf5")
    elif dataset == "eth3d" or dataset=="diode":
        
        data_dir = data_dir / frame_str
        loader = ETH3DLoader(data_dir)
        color_img = loader.load_color_image()
        h,w = color_img.shape[:2]
        
        color_img = cv2.resize(color_img, dsize=(int(w/2), int(h/2)),interpolation=cv2.INTER_CUBIC)

        depth_map = loader.load_depth_png()
        depth_map = cv2.resize(depth_map,dsize=(int(w/2), int(h/2)),interpolation=cv2.INTER_CUBIC)

        # plot_images([depth_map], ["Depth Cleaned"], cmaps='gray')
        h_new,w_new = color_img.shape[:2]

        K = loader.load_intrinsics()
        
        K[0, 0] *= w_new  # fx
        K[0, 2] *= w_new  # cx
        K[1, 1] *= h_new  # fy
        K[1, 2] *= h_new  # cy   
        depth_map_cleaned = cv2.medianBlur(depth_map.astype(np.float32), 5)

        world_coords = reproject_depth_to_points(depth_map_cleaned, K)
        
        X_channel = world_coords[..., 0]
        Y_channel = world_coords[..., 1]
        Z_channel = world_coords[..., 2]
        # plot_images([X_channel, Y_channel, Z_channel],
        #             ["X Channel", "Y Channel", "Z Channel"],
        #             cmaps=['viridis', 'viridis', 'viridis'])
        normal_map = compute_normal_map_from_points(world_coords, ksize=1)
        
   
        normal_map = cv2.medianBlur(normal_map.astype(np.float32), 5)

        # plot_images([normal_map], ["Normal Cleaned"], cmaps='gray')

    elif dataset=="scannet":
        loader = ScanNetLoader(data_dir)
        color_img = loader.load_color((frame_str))
        depth_map = loader.load_depth((frame_str))
        
        print(color_img.shape, depth_map.shape)
        H_img, W_img = color_img.shape[:2]

        H, W = depth_map.shape[:2]
        color_img = cv2.resize(color_img, (W, H), interpolation=cv2.INTER_LINEAR)
        K = loader.load_intrinsics()
        depth_map_cleaned = cv2.bilateralFilter(depth_map, 
                                      d=25, 
                                      sigmaColor=200, 
                                      sigmaSpace=200)

      
        depth_map_cleaned = cv2.medianBlur(depth_map_cleaned.astype(np.float32), 3)


        world_coords = reproject_depth_to_points(depth_map_cleaned, K)
        
        X_channel = world_coords[..., 0]
        Y_channel = world_coords[..., 1]
        Z_channel = world_coords[..., 2]
        # plot_images([X_channel, Y_channel, Z_channel],
        #             ["X Channel", "Y Channel", "Z Channel"],
        #             cmaps=['viridis', 'viridis', 'viridis'])

        normal_map = compute_normal_map_from_points(world_coords, ksize=11)
        normal_map = cv2.medianBlur(normal_map.astype(np.float32), 5)
        # normal_map = loader.load_mesh_normals(frame_str)

        
        plot_images([normal_map], ["Normal Cleaned"], cmaps='gray')
        
       

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return color_img, depth_map, world_coords, normal_map



def process_image(
    base_dir: str,
    image_id: str,
    frame_str: str,
    net: Any,
    device: torch.device,
    thickness: int = 1,
    thresh_normal: float = 8.2e13,
    thresh_depth: float = 0.2,
    struct_color: Tuple[int, int, int] = (0, 0, 255),
    text_color: Tuple[int, int, int] = (255, 0, 0),
    dataset: str = "hypersim",
    file_path: Optional[str] = None,
    plot: bool = False
) -> Tuple[Any, ...]:
    """
    Main pipeline: loads data, detects lines, computes edges, classifies, draws/splits,
    clusters, fits planes, merges, labels, saves results, and optionally plots.
    """
    start = time.time()
    print(f"Start: {start}")

    data_root = Path(base_dir)
    
    color_img, depth_map, world_coords, normal_map = load_data(
        image_id, frame_str, dataset, data_root
    )
    print(f"Loaded and normals computed: {time.time() - start}")

    lines, features, downsample = detect_lines(
        color_img, net, device
    )
    print(f"DeepLSD time: {time.time() - start}")

    sobel_n, tn, sobel_d, td, combined_edges = compute_edge_maps(
        normal_map, depth_map, thresh_normal, thresh_depth
    )
    print(f"Edges computed: {time.time() - start}")

    is_struct, is_depth_sep = classify_lines(lines, tn, td)
    print(f"Line classification done: {time.time() - start}")
    
    print(color_img.shape, depth_map.shape)

    comp_rgb, new_lines, line_info = draw_and_split(
        color_img, lines, is_struct, is_depth_sep,
        features, downsample, normal_map, depth_map,
        struct_color, text_color, thickness=thickness
    )
    print(f"Lines drawn and split: {time.time() - start}")

    # Connected components
    binary_mask = cv2.bitwise_not(combined_edges)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, np.ones((2,2), np.uint8))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
    num_labels, labels_im = cv2.connectedComponents(binary_mask)
    print(f"CC computed: {time.time() - start}, labels: {num_labels}")

    # RANSAC plane fitting
    cluster_planes: Dict[int, Any] = {}
    for lbl in np.unique(labels_im):
        if lbl == 0: continue
        pts = world_coords[labels_im == lbl]
        if pts.shape[0] < 50: continue
        
        model, inliers = ransac_plane_fit(pts, num_iterations=2000, threshold=0.7, min_inliers_ratio=0.3)

        # model, inliers = ransac_plane_fit(
        #     pts, num_iterations=100, threshold=0.04, min_inliers_ratio=0.8
        # )
        if model is None: continue
        in_pts = pts[inliers]
        errs = compute_distance_to_plane(in_pts, model[0], model[1])
        if errs.mean() > 0.4: continue
        A = np.c_[in_pts[:,0], in_pts[:,1], np.ones(in_pts.shape[0])]
        B = in_pts[:,2]
        sol, *_ = np.linalg.lstsq(A, B, rcond=None)
        n_ls = np.array([sol[0], sol[1], -1.0])
        n_ls /= np.linalg.norm(n_ls)
        cluster_planes[lbl] = {'ls_model': (n_ls, sol[2]), 'inliers_mask': inliers}
    print(f"RANSAC done: {time.time() - start}")

    # Filter clusters
    new_labels = labels_im.copy()
    for lbl in np.unique(labels_im):
        if lbl != 0 and lbl not in cluster_planes:
            new_labels[new_labels == lbl] = 0

    # Build RAG and merge planes
    import networkx as nx
    def are_planes_similar(p1, p2,
                            normal_thresh=0.98,
                            dist_thresh=0.02) -> bool:
        n1, d1 = p1['ls_model']; n2, d2 = p2['ls_model']
        return (abs(n1.dot(n2)) >= normal_thresh) and (abs(d1 - d2) <= dist_thresh)

    G = nx.Graph()
    labels = list(cluster_planes.keys())
    for lbl in labels:
        G.add_node(lbl)
    masks = {lbl: (new_labels == lbl).astype(np.uint8) for lbl in labels}
    dil = {lbl: cv2.dilate(masks[lbl], np.ones((12,12), np.uint8), iterations=7)
           for lbl in labels}
    
    for i, l1 in enumerate(labels):
        for l2 in labels[i+1:]:
            if np.any(dil[l1] & masks[l2]) and are_planes_similar(
               cluster_planes[l1], cluster_planes[l2], 0.99, 0.01):
                G.add_edge(l1, l2)
    merged_groups = list(nx.connected_components(G))

    # Create merged map
    merged_map = np.zeros_like(labels_im)
    new_lbl = 1
    for group in merged_groups:
        for lbl in group:
            merged_map[labels_im == lbl] = new_lbl
        new_lbl += 1
    print(f"Planes merged: {time.time() - start}")


  
    # Coplanarity labeling
    dilated_map = merged_map.copy()
    for lbl in np.unique(merged_map):
        if lbl == 0: continue
        mask_lbl = (merged_map == lbl).astype(np.uint8)
        dmask = cv2.dilate(mask_lbl, np.ones((3,3), np.uint8), iterations=3)
        dilated_map[dmask == 1] = lbl

    line_planes = find_line_planes(new_lines, dilated_map, get_line_pixels_trim)
    copl_labels: List[List[int]] = []
    for entry in line_info:
        inds = entry.pop("new_line_indices")
        labs = [line_planes[i] for i in inds]
        entry["coplanarity_labels"] = labs
        copl_labels.append(labs)

    N = len(copl_labels)
    copl_matrix = np.zeros((N, N), dtype=int)
    for i in range(N):
        for j in range(N):
            if any(l in copl_labels[j] for l in copl_labels[i] if l > 0):
                copl_matrix[i, j] = 1

    # Save results
    save_lines_to_json(
        image_id, frame_str, line_info, copl_matrix,
        file_path_str=file_path
    )

    # Optional plotting
    if plot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 3, figsize=(15, 10))
        axs = axes.ravel()
        axs[0].imshow(sobel_n, cmap='gray'); axs[0].set_title('Sobel Normal'); axs[0].axis('off')
        axs[1].imshow(tn, cmap='gray'); axs[1].set_title('Thresholded Normal'); axs[1].axis('off')
        axs[2].imshow(sobel_d, cmap='gray'); axs[2].set_title('Sobel Depth'); axs[2].axis('off')
        axs[3].imshow(td, cmap='gray'); axs[3].set_title('Thresholded Depth'); axs[3].axis('off')
        axs[4].imshow(color_img); plot_lines_bool(axs[4], color_img, lines, is_correct=is_struct)
        axs[4].set_title('Line Matches'); axs[4].axis('off')
        axs[5].imshow(comp_rgb); axs[5].set_title('Line Splitting'); axs[5].axis('off')
        def make_color(img): return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[6].imshow(make_color(color_map(labels_im, num_labels))); axs[6].set_title('Connected Components'); axs[6].axis('off')
        axs[7].imshow(make_color(color_map(new_labels, num_labels))); axs[7].set_title('Valid Clusters'); axs[7].axis('off')
        axs[8].imshow(make_color(color_map(merged_map, num_labels))); axs[8].set_title('Merged Planes'); axs[8].axis('off')
        axs[9].imshow(make_color(color_map(dilated_map, num_labels))); axs[9].set_title('Dilated'); axs[9].axis('off')
        plot_coplanar_lines(axs[10], new_lines, line_planes, color_img); axs[10].set_title('Line Coplanarity'); axs[10].axis('off')
        axs[11].imshow(make_color(color_img)); axs[11].set_title('Original Image'); axs[11].axis('off')
        plt.tight_layout(); plt.show()




