import os
import json
import orjson

import numpy as np
import torch
import torch.nn as nn

import cv2
from torch_geometric.data import Data, Dataset
#from notebooks.models.dataset_utils import extract_line_feature_ROIAlign,sample_lines_grid, extract_fixed_oriented_patches, extract_oriented_feature_patches
from line_sampler import extract_fixed_oriented_patches, extract_oriented_feature_patches
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


def _load_array_from_json_or_npy(field, json_dict, desc="array"):
    """
    Helper: either reads an inline list from json_dict[field]
    or loads from json_dict[f"{field}_path"] via np.load.
    Returns a NumPy array.
    """
    if field in json_dict:
        return np.array(json_dict[field])
    path_field = f"{field}_path"
    if path_field in json_dict:
        arr_path = json_dict[path_field]
        if not os.path.exists(arr_path):
            raise FileNotFoundError(f"{desc} file not found: {arr_path}")
        return np.load(arr_path)
    raise KeyError(f"Neither '{field}' nor '{path_field}' found in JSON.")

import os, json, logging
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from line_sampler import LineSampler, EdgeSampler  # <-- your LightningModule
from line_sampler import extract_line_feature_ROIAlign

class GraphDatasetInductive(Dataset):
    def __init__(self, json_dir, roi_output_size=(64, 64), method="sample", device=None, edge_sample_size = (32,16)):
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

          
            img_path = data.get('file_path')
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
        # # load JSON
        # with open(self.filter_json_files[idx], 'r') as f:
        #     graph_data = orjson.loads(f.read())


        # lines = graph_data.get('lines', [])
        # if not lines:
        #     raise ValueError('lines from the graph_data is empty')

        # coplanarity_matrix = graph_data.get('coplanarity_matrix', [[]])
        # file_path_img     = graph_data.get('file_path')

        # # build node embeddings + labels
        # feats, labels, line_coords = [], [], []
        # for ln in lines:
        #     feats.append(np.array(ln['embedding_DeepLSD']))
        #     labels.append(ln.get('struct_score', 0.5))
        #     line_coords.append(ln.get('coordinates'))
        # x_emb = torch.tensor(np.vstack(feats), dtype=torch.float)
        # coords = torch.tensor(line_coords, dtype=torch.float)
        # y     = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
        # N     = x_emb.size(0)
        
        # --- 1) load JSON ---
        with open(self.filter_json_files[idx], 'rb') as f:
            graph_data = orjson.loads(f.read())

        # --- 2) load coplanarity matrix ---
        coplanarity_matrix = _load_array_from_json_or_npy(
            "coplanarity_matrix", graph_data, desc="Coplanarity matrix"
        )  # shape: (N,N)

        # --- 3) load file_path & image ---
        file_path_img = graph_data.get("file_path", None)
        img = _load_image(filepath=file_path_img, color_conversion=cv2.COLOR_BGR2RGB)

        # --- 4) load per-line features & coords & labels ---
        lines = graph_data.get("lines", [])
        if not lines:
            raise ValueError("No lines found in JSON.")

        labels, line_coords = [], [], []
        for ln in lines:
    
            # struct score
            labels.append(ln.get("struct_score", 0.0))
            # coords always inline
            line_coords.append(ln["coordinates"])

        coords = torch.tensor(line_coords, dtype=torch.float)
        y      = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
        N      = x_emb.size(0)



        img =_load_image(filepath=file_path_img, color_conversion=cv2.COLOR_BGR2RGB)
        downsample_ratio = 2
        feature_map = _load_array_from_json_or_npy(
            "feature_map", ln, desc="Line embedding"
        )
        img_np = img
        img_t  = (
            torch.tensor(img_np, dtype=torch.float32)
                .div(255.0)
                .permute(2, 0, 1)  # C,H,W
        )

        # 2) prep lines tensor
        lines_t = torch.tensor(line_coords, dtype=torch.float32)       
        strip_img = extract_fixed_oriented_patches(
            img           = img_np,                    # (H,W,C) NumPy
            lines         = line_coords,               # (N,2,2)
            patch_size    = (64, 128),                 # pixels on *image*
            draw_line     = True,                     # draw line on patch
        )                                              # → (N, 3, 64, 128)


        # ---------------------------------------------------------------- #
        # the *corresponding* patch on the feature map
        # ---------------------------------------------------------------- #
        strip_fmap = extract_oriented_feature_patches(
            feature_map       = feature_map,              # (1,C,Hf,Wf) torch
            lines             = line_coords,           # same pixel coords!
            patch_size_img    = (64, 128),             # same number you used above
            downsample_ratio  = downsample_ratio,       # e.g. 4 or 8
            draw_line         = False
        )       

        

        # === build graph ===
        

        # Use line coordinates to determine k-NN (e.g., 7 nearest neighbors)
        # coords_center = np.mean(line_coords, axis=1)  # shape: (N, 2)
        # nbrs = NearestNeighbors(n_neighbors=7, algorithm='auto').fit(coords_center)
        # distances, indices = nbrs.kneighbors(coords_center)



        # # reuse your line_geometry to get [mid_x,mid_y,dir_x,dir_y,length]
        geo = line_geometry(coords)       # (N,5)
        
        # 2) helper to compute segment‐to‐segment distances
        def seg_seg_dist(p1, p2, q1, q2, eps=1e-8):
            # p1,p2: (N,2); q1,q2: (N,2) – here we compute N×N all-pairs
            # expand dims for broadcasting
            P1 = p1[:,None]  # (N,1,2)
            P2 = p2[:,None]  # (N,1,2)
            Q1 = q1[None,:]  # (1,N,2)
            Q2 = q2[None,:]  # (1,N,2)
            def proj(X, A, B):
                t = torch.clamp(((X-A)*(B-A)).sum(-1,keepdim=True) /
                                (((B-A)**2).sum(-1,keepdim=True)+eps), 0,1)
                return A + t*(B-A)
            # four point‐to‐segment cases
            d1 = ((Q1 - proj(Q1, P1, P2))**2).sum(-1)
            d2 = ((Q2 - proj(Q2, P1, P2))**2).sum(-1)
            d3 = ((P1 - proj(P1, Q1, Q2))**2).sum(-1)
            d4 = ((P2 - proj(P2, Q1, Q2))**2).sum(-1)
            return torch.sqrt(torch.min(torch.min(d1,d2), torch.min(d3,d4)))  # (N,N)

        # endpoints for seg‐seg
        p1, p2 = coords[:,0], coords[:,1]  # each (N,2)
        D = seg_seg_dist(p1,p2, p1,p2)    # (N,N)

        # 6) build k‐NN graph from D
        k = 7  # number of neighbors
        # topk returns self in position 0, so grab 1:k+1
        knn = D.topk(k+1, largest=False).indices[:,1:]  # (N,k)

        edge_list, edge_labels = [], []
        full_edge_index, full_edge_labels = [], []

        for i in range(N):
            for j in range(N):  
                full_edge_index.append([i, j])
                full_edge_labels.append(coplanarity_matrix[i][j])
                
            # for j in knn[i]:  
            #     edge_list.append([i, j])
            #     edge_labels.append(coplanarity_matrix[i][j])
                
                
        # edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        # edge_labels = torch.tensor(edge_labels, dtype=torch.float).unsqueeze(1)           

        full_edge_index = torch.tensor(full_edge_index, dtype=torch.long).t().contiguous()
        full_edge_labels = torch.tensor(full_edge_labels, dtype=torch.float).unsqueeze(1)
        
        
        
        # edge_attr = self.edge_sampler((
        #     torch.tensor(img_np, dtype=torch.float32)
        #         .div(255.0)
        #         .permute(2, 0, 1)  # C,H,W
        # ), coords)[0]
        # edge_attr = edge_attr
        # N = coords.shape[0]
        # src, dst = edge_index              # each is length E_local
        # flat_idx_local = src * N + dst     # vectorized  i*N + j


      # 8) Directly sample edge attributes only for these local edges
        img_t = torch.tensor(img_np, dtype=torch.float32).div(255).permute(2,0,1)
     

        # 1) flatten knn into a single edge_index of shape (2, E):
        src = torch.arange(N, device=coords.device).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = knn.reshape(-1)
        local_edge_index = torch.stack([src, dst], dim=0)   # (2, N*k)

        # 2) build the M = N*k “quad” coords exactly as before,
        #    but here we pass only (start_i, end_i, end_j, start_j)
        lines_i = coords[src]   # (E, 2, 2)
        lines_j = coords[dst]   # (E, 2, 2)
        quads = torch.stack([
            lines_i[:,0],  # start_i
            lines_i[:,1],  # end_i
            lines_j[:,1],  # end_j
            lines_j[:,0],  # start_j
        ], dim=1)         # (E, 4, 2)

        # 3) now call the sampler and overwrite both the edge_attr and edge_index
        edge_attr = self.edge_sampler(img_t, quads)

        return Data(
            y=y,
            coordinates=coords,
            geo=geo,
            edge_index=local_edge_index,
            full_edge_index=full_edge_index,
            full_edge_labels=full_edge_labels,
            strip_img=strip_img,
            strip_fmap=strip_fmap,
            edge_attr = edge_attr,
            # edge_dist = D,
            #flat_idx_local=flat_idx_local,
            # img_path = file_path_img
        )
