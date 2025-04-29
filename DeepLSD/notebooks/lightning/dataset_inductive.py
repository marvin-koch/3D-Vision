import os
import json
import numpy as np
import torch
import cv2
from torch_geometric.data import Data, Dataset
# from notebooks.models.dataset_utils import extract_line_feature_ROIAlign,sample_lines_grid

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
    
# def get_line_feature(image_path, line_coordinates, output_size = (64,64), method="roi"):
#     """
#     method is roi or sample
#     output size for sample is interpreted as num_samples = output_size[0], width = output_size[1]
#     """
#     img = _load_image(filepath = image_path, color_conversion=cv2.COLOR_BGR2RGB)
#     if method == "roi":
#         return extract_line_feature_ROIAlign(img=img,lines=line_coordinates,output_size=output_size,plot_results=True)
#     elif method == "sample":
#         img = torch.tensor(img).div(255.0).permute(2, 0, 1)
#         return sample_lines_grid(img=img,lines=torch.tensor(line_coordinates,dtype=torch.float32),num_samples=output_size[0],width=output_size[1],)
#     logging.error(f"Method not found: {method}. Returning None.")
#     return None

# class GraphDatasetInductive(Dataset):
#     def __init__(self, json_dir, roi_output_size=(64, 64), method="roi"):
#         super().__init__()
#         self.json_files = [
#             os.path.join(json_dir, f)
#             for f in os.listdir(json_dir)
#             if f.endswith('.json')
#         ]
#         self.roi_output_size = roi_output_size
#         self.method = method
#         # self.json_files = self.json_files[:20] # Limit for testing

#     def __len__(self):
#         return len(self.json_files)

#     def __getitem__(self, idx):
#         # load JSON
#         with open(self.json_files[idx], 'r') as f:
#             graph_data = json.load(f)

#         lines = graph_data.get('lines', [])
#         if not lines: # Handle cases with no lines
#              # Return an empty Data object or None, depending on desired behavior
#              raise ValueError('lines from the graph_data is empty')
             

#         coplanarity_matrix = graph_data.get('coplanarity_matrix', [[]])
#         file_path_img = graph_data.get('file_path')

#         # build node features (embeddings) + labels
#         feats, labels,line_coords = [], [], []
#         for ln in lines:
#             emb = np.array(ln['embedding_DeepLSD'])
#             score = ln.get('struct_score', 0.5)
#             line_coords.append(ln.get('coordinates'))
#             feats.append(emb)
#             labels.append(score)

#         x_emb = torch.tensor(np.vstack(feats), dtype=torch.float)
#         y = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
#         N = x_emb.size(0)

#         # Extract ROI Align features
#         # line_coords_np = np.array(line_coords)
#         line_coords_np = np.array(line_coords)
#         #mprint(line_coords_np.shape)
#         # img = _load_image(filepath=file_path_img, color_conversion=cv2.COLOR_BGR2RGB)
#         # roi_features shape: (N, C, H, W)
#         roi_features = get_line_feature(
#             image_path=file_path_img,
#             line_coordinates=line_coords_np,
#             output_size=self.roi_output_size,
#             method=self.method
#         )

#         # Handle potential mismatch or failure in ROI extraction
#         if roi_features is None or roi_features.shape[0] != N:
#              print(f"Warning: ROI feature issue for {self.json_files[idx]}. Features shape: {roi_features.shape if roi_features is not None else 'None'}, Expected N: {N}. Skipping graph.")
#              raise ValueError('lines from the graph_data is empty')


#         # fully connected graph + coplanarity labels
#         edge_list, edge_labels = [], []
#         for i in range(N):
#             for j in range(N):
#                 edge_labels.append(coplanarity_matrix[i][j])
#                 # if i == j: continue
#                 edge_list.append([i, j])

#         edge_index  = torch.tensor(edge_list,  dtype=torch.long).t().contiguous()
#         edge_labels = torch.tensor(edge_labels, dtype=torch.float).unsqueeze(1)

#         # Create Data object, storing roi_features as a separate attribute
#         data_object = Data(x=x_emb, y=y, edge_index=edge_index, edge_labels=edge_labels, roi_features=roi_features)

#         return data_object
    


import os, json, logging
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from line_edge_sampler import LineSampler  # <-- your LightningModule
from line_edge_sampler import extract_line_feature_ROIAlign
from line_edge_sampler import EdgeSampler
class GraphDatasetInductive(Dataset):
    def __init__(self, json_dir, roi_output_size=(64, 64), method="roi", device=None, edge_sample_size = (32,8),):
        super().__init__()
        self.json_files = [
            os.path.join(json_dir, f)
            for f in os.listdir(json_dir)
            if f.endswith('.json')
        ]
        self.roi_output_size = roi_output_size
        self.method = method
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.edge_sample_size = edge_sample_size

        # Only instantiate the sampler if we're going to use it
        if self.method == "sample":
            num_samples, width = self.roi_output_size
            self.sampler = LineSampler(
                num_samples=num_samples,
                width=width
            )

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        # load JSON
        with open(self.json_files[idx], 'r') as f:
            graph_data = json.load(f)

        lines = graph_data.get('lines', [])
        if not lines:
            raise ValueError('lines from the graph_data is empty')

        coplanarity_matrix = graph_data.get('coplanarity_matrix', [[]])
        file_path_img     = graph_data.get('file_path')
        file_path_img = file_path_img.replace("/work/scratch/maurdu/", "/work/scratch/maurdu/data/", 1) if file_path_img.startswith("/work/scratch/maurdu/") and not file_path_img.startswith("/work/scratch/maurdu/data/") else file_path_img
        # build node embeddings + labels
        feats, labels, line_coords = [], [], []
        for ln in lines:
            feats.append(np.array(ln['embedding_DeepLSD']))
            labels.append(ln.get('struct_score', 0.5))
            line_coords.append(ln.get('coordinates'))
        x_emb = torch.tensor(np.vstack(feats), dtype=torch.float)
        y     = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
        N     = x_emb.size(0)

        # === Feature extraction ===
        if self.method == "roi":
            roi_features = extract_line_feature_ROIAlign(
                img=_load_image(filepath=file_path_img, color_conversion=cv2.COLOR_BGR2RGB),
                lines=line_coords,
                output_size=self.roi_output_size,
                plot_results=True
            )
        else:  # self.method == "sample"
            # 1) load & prep image tensor
            img_np = _load_image(filepath=file_path_img, color_conversion=cv2.COLOR_BGR2RGB)
            img_t  = (
                torch.tensor(img_np, dtype=torch.float32)
                     .div(255.0)
                     .permute(2, 0, 1)  # C,H,W
            )

            # 2) prep lines tensor
            lines_t = torch.tensor(line_coords, dtype=torch.float32)

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
                f"ROI feature issue for {self.json_files[idx]}; "
                f"got {None if roi_features is None else roi_features.shape}, expected ({N}, …)."
            )
            raise ValueError('ROI feature extraction failed.')
                # Use line coordinates to determine k-NN (e.g., 7 nearest neighbors)
        line_coords_np = np.array(line_coords)
        from sklearn.neighbors import NearestNeighbors
        coords_center = np.mean(line_coords_np, axis=1)  # shape: (N, 2)
        nbrs = NearestNeighbors(n_neighbors=7, algorithm='auto').fit(coords_center)
        distances, indices = nbrs.kneighbors(coords_center)

        edge_list, edge_labels = [], []

        for i in range(N):
            for j in indices[i]:  
                edge_list.append([i, j])
                edge_labels.append(coplanarity_matrix[i][j])

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_labels = torch.tensor(edge_labels, dtype=torch.float).unsqueeze(1)


        # fully connected graph  + edge attributes
        num_samples_edge,width_edge = self.edge_sample_size
        edge_sampler = EdgeSampler(num_samples_u=num_samples_edge,num_samples_v=width_edge)
        
        edge_index_full, edge_attr = None, None
        edge_attr,edge_index_full = edge_sampler.sample_edges(img=img_t,lines=lines_t)
        # for i in range(N):
        #     for j in range(N):
        #         if i == j: continue
        #         edge_index_full.append([i, j])
        edge_index_full = torch.tensor(edge_index_full, dtype=torch.long).t().contiguous()
        


        return Data(x=x_emb, y=y, edge_index=edge_index, edge_labels=edge_labels, roi_features=roi_features, edge_index_full=edge_index_full, edge_attr = edge_attr)