import os
import json
import orjson

import numpy as np
import torch
import cv2
from torch_geometric.data import Data, Dataset
# from notebooks.models.dataset_utils import extract_line_feature_ROIAlign,sample_lines_grid
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F

from typing import Optional
import logging

def _load_image(filepath: str, color_conversion: Optional[int] = None) -> Optional[np.ndarray]:
    """Loads an image using OpenCV."""
    if not os.path.exists(filepath):
        logging.error(f"Image file not found: {filepath}")
        return None
    try:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED) # Load as is (handles color, grayscale, alpha)
        if img is None:
            logging.error(f"Failed to load image (cv2.imread returned None): {filepath}")
            return None
        if color_conversion is not None:
            img = cv2.cvtColor(img, color_conversion)
        return img
    except Exception as e:
        logging.error(f"Error loading image {filepath}: {e}")
        return None
    
def line_geometry(line_pts: torch.Tensor):
    """
    line_pts : [N, 2, 2]  (x1,y1,x2,y2 per line)
    returns   :  ϕ_node   [N, 5]
    [mid_x, mid_y, dir_x, dir_y, length]
    """
    p1, p2   = line_pts[:, 0], line_pts[:, 1]           # [N,2]  [N,2]
    mid      = 0.5 * (p1 + p2)                          # [N,2]
    vec      = p2 - p1
    length   = vec.norm(dim=1, keepdim=True)            # [N,1]
    dir_norm = F.normalize(vec, dim=1)                  # [N,2]
    return torch.cat([mid, dir_norm, length], dim=1)    # [N,5]

import os, json, logging
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from line_sampler import LineSampler, EdgeSampler  # <-- your LightningModule
from line_sampler import extract_line_feature_ROIAlign

class GraphDatasetInductive(Dataset):
    def __init__(self, json_dir, roi_output_size=(64, 64), method="sample", device=None, edge_sample_size = (7,3)):
        super().__init__()
        json_files = [
            os.path.join(json_dir, f)
            for f in os.listdir(json_dir)
            if f.endswith('.json')
        ]
        self.filter_json_files = []
        for jf in json_files:
            with open(jf, 'r') as f:
                data = orjson.loads(f.read())

            img_path = "../" + data.get('file_path')
            # try to load once
                    # == FAST EXISTENCE CHECK ==
            if img_path and os.path.exists(img_path):
                self.filter_json_files.append(jf)
            else:
                logging.warning(f"Skipping {jf} because image not found: {img_path}")
                
                
        self.roi_output_size = roi_output_size
        self.method = method
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.edge_sample_size = edge_sample_size
        num_samples_edge,width_edge = self.edge_sample_size

        self.edge_sampler = EdgeSampler(num_samples_u=num_samples_edge,num_samples_v=width_edge)

        # Only instantiate the sampler if we're going to use it
        if self.method == "sample":
            num_samples, width = self.roi_output_size
            self.sampler = LineSampler(
                num_samples=num_samples,
                width=width
            )

    def __len__(self):
        return len(self.filter_json_files)

    def __getitem__(self, idx):
        # load JSON
        with open(self.filter_json_files[idx], 'r') as f:
            graph_data = orjson.loads(f.read())


        lines = graph_data.get('lines', [])
        if not lines:
            raise ValueError('lines from the graph_data is empty')

        coplanarity_matrix = graph_data.get('coplanarity_matrix', [[]])
        file_path_img     = graph_data.get('file_path')

        # build node embeddings + labels
        feats, labels, line_coords = [], [], []
        for ln in lines:
            feats.append(np.array(ln['embedding_DeepLSD']))
            labels.append(ln.get('struct_score', 0.5))
            line_coords.append(ln.get('coordinates'))
        x_emb = torch.tensor(np.vstack(feats), dtype=torch.float)
        coords = torch.tensor(line_coords, dtype=torch.float)
        y     = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
        N     = x_emb.size(0)


        img =_load_image(filepath="../" + file_path_img, color_conversion=cv2.COLOR_BGR2RGB)
        # === Feature extraction ===
        if self.method == "roi":
            roi_features = extract_line_feature_ROIAlign(
                img=img,
                lines=line_coords,
                output_size=self.roi_output_size,
                plot_results=True
            )
        else:  # self.method == "sample"
            # 1) load & prep image tensor
            img_np = img
            img_t  = (
                torch.tensor(img_np, dtype=torch.float32, device=self.device)
                     .div(255.0)
                     .permute(2, 0, 1)  # C,H,W
            )

            # 2) prep lines tensor
            lines_t = torch.tensor(line_coords, dtype=torch.float32, device=self.device)

            # 3) sample
            #    returns (N, C, num_samples, width)
            roi_features = self.sampler.sample_lines_grid(
                img=img_t,
                lines=lines_t,
                align_corners=True
            )

        # sanity‐check
        if roi_features is None or roi_features.shape[0] != N:
            logging.warning(
                f"ROI feature issue for {self.filter_json_files[idx]}; "
                f"got {None if roi_features is None else roi_features.shape}, expected ({N}, …)."
            )
            raise ValueError('ROI feature extraction failed.')

        # === build graph ===
        

        # Use line coordinates to determine k-NN (e.g., 7 nearest neighbors)
        coords_center = np.mean(line_coords, axis=1)  # shape: (N, 2)
        nbrs = NearestNeighbors(n_neighbors=7, algorithm='auto').fit(coords_center)
        distances, indices = nbrs.kneighbors(coords_center)

        edge_list, edge_labels = [], []
        full_edge_index, full_edge_labels = [], []

        for i in range(N):
            for j in range(N):  
                full_edge_index.append([i, j])
                full_edge_labels.append(coplanarity_matrix[i][j])
                
            for j in indices[i]:  
                edge_list.append([i, j])
                edge_labels.append(coplanarity_matrix[i][j])
                
                
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_labels = torch.tensor(edge_labels, dtype=torch.float).unsqueeze(1)           

        full_edge_index = torch.tensor(full_edge_index, dtype=torch.long).t().contiguous()
        full_edge_labels = torch.tensor(full_edge_labels, dtype=torch.float).unsqueeze(1)
        
        
        


        # edge_list, edge_labels = [], []
        # for i in range(N):
        #     for j in range(N):
        #         edge_list.append([i, j])
        #         edge_labels.append(coplanarity_matrix[i][j])
        # edge_index  = torch.tensor(edge_list,  dtype=torch.long).t().contiguous()
        # edge_labels = torch.tensor(edge_labels, dtype=torch.float).unsqueeze(1)

        return Data(
            x=x_emb,
            y=y,
            coordinates=coords,
            geo=line_geometry(coords),
            edge_index=edge_index,
            edge_labels=edge_labels,
            full_edge_index=full_edge_index,
            full_edge_labels=full_edge_labels,
            roi_features=roi_features,
            edge_attr = self.edge_sampler((
                torch.tensor(img_np, dtype=torch.float32, device=self.device)
                     .div(255.0)
                     .permute(2, 0, 1)  # C,H,W
            ), coords)[0]
        )
