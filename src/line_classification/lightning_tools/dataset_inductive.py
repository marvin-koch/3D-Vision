import os
import json
import orjson

import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple

import cv2
from torch_geometric.data import Data, Dataset
# from notebooks.models.dataset_utils import extract_line_feature_ROIAlign,sample_lines_grid
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F

from typing import Optional
import logging

def seg_seg_dist(p1, p2, q1, q2, eps=1e-8):
            P1 = p1[:, None]
            P2 = p2[:, None]
            Q1 = q1[None, :]
            Q2 = q2[None, :]
            def proj(X, A, B):
                t = torch.clamp(((X - A) * (B - A)).sum(-1, keepdim=True) /
                                (((B - A)**2).sum(-1, keepdim=True) + eps), 0, 1)
                return A + t * (B - A)
            d1 = ((Q1 - proj(Q1, P1, P2))**2).sum(-1)
            d2 = ((Q2 - proj(Q2, P1, P2))**2).sum(-1)
            d3 = ((P1 - proj(P1, Q1, Q2))**2).sum(-1)
            d4 = ((P2 - proj(P2, Q1, Q2))**2).sum(-1)
            return torch.sqrt(torch.min(torch.min(d1, d2), torch.min(d3, d4)))
        
        
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
    
def line_geometry(line_pts: torch.Tensor, img_size: Tuple[int,int]):
    """
    line_pts : [N, 2, 2]  (x1,y1,x2,y2 per line)
    img_size : (H, W)
    returns   :  ϕ_node   [N, 5]
      [mid_x_norm, mid_y_norm, dir_x, dir_y, length_norm]
    """
    p1, p2   = line_pts[:, 0], line_pts[:, 1]           # [N,2]  [N,2]
    H, W     = img_size
    diag     = (H**2 + W**2)**0.5

    # 1) midpoint, then normalize to [0,1]
    mid      = 0.5 * (p1 + p2)                          # [N,2]
    mid_x    = mid[:,0:1] / W
    mid_y    = mid[:,1:2] / H

    # 2) unit direction
    vec      = p2 - p1
    dir_norm = F.normalize(vec, dim=1)                  # [N,2]

    # 3) length normalized by image diagonal
    length   = vec.norm(dim=1, keepdim=True) / diag     # [N,1]

    return torch.cat([mid_x, mid_y, dir_norm, length], dim=1)  # [N,5]


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
        arr_path = "../" + json_dict[path_field]
        if not os.path.exists(arr_path):
            raise FileNotFoundError(f"{desc} file not found: {arr_path}")
        return np.load(arr_path)
    raise KeyError(f"Neither '{field}' nor '{path_field}' found in JSON.")


import numpy as np
import networkx as nx
from typing import List
import numpy as np
import networkx as nx
from typing import List

def adjacency_to_overlapping_clusters(adj: np.ndarray,
                                      min_size: int = 2
                                     ) -> List[List[int]]:
    """
    Given an N×N symmetric adjacency matrix 'adj' with adj[i,j] = 1
    whenever line i and line j appear in (at least one) common ground-truth
    plane, return a list of all *maximal* cliques of size >= min_size.

    Each clique is returned as a list of line-indices.  Because a line may
    belong to multiple planes, those cliques may overlap in one or more nodes.

    Args:
        adj       : (N,N) numpy array, symmetric, zeros on diag (or ones—either is fine).
        min_size  : ignore any clique smaller than this.
    Returns:
        clusters  : List of List[int], each inner list is the set of lines in one plane.
    """
    N = adj.shape[0]
    # 1) build an undirected graph
    G = nx.Graph()
    G.add_nodes_from(range(N))

    # add an edge (i,j) if adj[i,j] == 1
    # (we only need to add the upper triangle to avoid duplicates)
    rows, cols = np.where(np.triu(adj, k=1) == 1)
    edges = list(zip(rows.tolist(), cols.tolist()))
    G.add_edges_from(edges)

    # 2) enumerate all *maximal* cliques
    #    (NetworkX’s find_cliques returns every clique that is not properly
    #     contained in a larger clique.)
    raw_cliques = list(nx.find_cliques(G))

    # 3) filter by size
    clusters: List[List[int]] = [
        clique for clique in raw_cliques
        if len(clique) >= min_size
    ]
    return clusters


import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List

def cliques_to_flat_labels(
    cliques: List[List[int]],
    N: int,
    tie_break: str = "largest"
) -> np.ndarray:
    """
    Convert overlapping cliques (List[List[int]]) into a flat label array of length N.
    If a line belongs to >1 clique, break ties by selecting:
      - 'first'   → the clique with smallest index
      - 'largest' → the clique whose list has max length
      - 'random'  → pick uniformly at random among its cliques
    Returns a numpy array `labels` of shape (N,), where labels[i] ∈ {0..K-1} or -1.
    """
    sizes = [len(c) for c in cliques]
    memberships = [[] for _ in range(N)]
    for clique_idx, clique in enumerate(cliques):
        for i in clique:
            memberships[i].append(clique_idx)

    labels = -1 * np.ones((N,), dtype=int)
    rng = np.random.default_rng()
    for i in range(N):
        m = memberships[i]
        if not m:
            continue
        if len(m) == 1:
            labels[i] = m[0]
        else:
            if tie_break == "first":
                labels[i] = m[0]
            elif tie_break == "largest":
                best = max(m, key=lambda idx: sizes[idx])
                labels[i] = best
            elif tie_break == "random":
                labels[i] = rng.choice(m)
            else:
                raise ValueError(f"unknown tie_break='{tie_break}'")
    return labels


def plot_line_clusters(
    image: np.ndarray,
    line_coords: np.ndarray,
    cliques: List[List[int]],
    tie_break: str = "largest",
) -> None:
    """
    Overlays each line (given by line_coords) on top of `image`, coloring it
    according to which clique (cluster) it was assigned to. Lines not in any
    clique are drawn in gray.

    Args:
      image      : (H, W, 3) uint8 or float image in [0..255] or [0..1].
      line_coords: (N, 2, 2) array of ints/floats, e.g. [[x1,y1],[x2,y2]] per line.
      cliques    : List of maximal cliques, each a list of line‐indices.
      tie_break  : same as above ("largest"/"first"/"random").
      figsize    : size of the matplotlib figure.
    """
    N = line_coords.shape[0]
    # 1) compute a flat label per line
    flat_labels = cliques_to_flat_labels(cliques, N, tie_break)  # shape (N,)

    # 2) pick a colormap with enough distinct colors
    K = max(flat_labels.max()+1, 1)
    # if no cliques, K=0, so force at least 1
    # use a tab20 (up to 20 colors). If K > 20, repeats will occur.
    cmap = plt.get_cmap("tab20", K)

    # 3) ensure image is in [0..1] floats for matplotlib
    if image.dtype == np.uint8:
        img_plot = image.astype("float32") / 255.0
    else:
        img_plot = image.copy()
        if img_plot.max() > 1.0:
            img_plot = img_plot / 255.0

    fig, ax = plt.subplots()
    ax.imshow(img_plot)
    ax.axis("off")

    # 4) plot each line
    for i in range(N):
        label = int(flat_labels[i])
        (x1, y1), (x2, y2) = line_coords[i]
        if label < 0:
            color = (0.5, 0.5, 0.5)  # gray for “no clique”
            lw = 1.0
            alpha = 0.6
        else:
            color = cmap.colors[label % cmap.N][:3]
            lw = 2.0
            alpha = 1.0

        ax.plot([x1, x2], [y1, y2],
                color=color,
                linewidth=lw,
                alpha=alpha)

    ax.set_title(f"{N} lines → {K} clusters (tie_break='{tie_break}')")
    plt.show()

import os
import json
import orjson
import random
import torchvision.transforms as T


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional

import cv2
import logging
import networkx as nx

from torch.utils.data import Dataset
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors

from line_sampler import LineSampler, EdgeSampler      # <-- your LightningModule
from line_sampler import extract_line_feature_ROIAlign


class GraphDatasetInductive(Dataset):
    def __init__(self,
                 json_dir: str,
                 roi_output_size=(64, 32),
                 method="sample",
                 device=None,
                 edge_sample_size=(32, 24),
                 # ── NEW: how many augmented versions PER JSON file:
                 # ── same augmentation params as before (rot/scale/jitter):
                 rot_range: float = 30.0,      # ± degrees
                 scale_range: float = 0.15,     # scale ∈ [1−0.1, 1+0.1]
                 jitter_sigma: float = 4.0,    # pixels of Gaussian noise on endpoints
                 augment: bool = False,

                ):
        super().__init__()

        # ── 1) find all JSON files on disk ──
        json_files = [
            os.path.join(json_dir, f)
            for f in os.listdir(json_dir)
            if f.endswith(".json")
        ]
        # (You already had this, plus some filtering logic:)
        all_jsons = sorted(json_files)

        # 2) Filter out JSONs that have no image or no lines:
        self.filter_json_files = []
        for jf in all_jsons:
            with open(jf, "r") as f:
                data = orjson.loads(f.read())
            img_path = "../" + data.get("file_path", None)
            # img_path =  data.get("file_path", None)

            lines = data.get("lines", [])
            if img_path and os.path.exists(img_path) and len(lines) > 0:
                self.filter_json_files.append(jf)
            else:
                logging.warning(f"Skipping {jf} because image not found or no lines.")

        # 3) store everything else exactly as before:
        self.roi_output_size = roi_output_size
        self.method = method
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.edge_sample_size = edge_sample_size
        num_samples_edge, width_edge = self.edge_sample_size
        self.edge_sampler = EdgeSampler(num_samples_u=num_samples_edge, num_samples_v=width_edge)

        if self.method == "sample":
            num_samples, width = self.roi_output_size
            self.sampler = LineSampler(num_samples=num_samples, width=width)

        # ── 4) store new “multi-version” & augmentation hyperparameters ──
        self.rot_range    = rot_range
        self.scale_range  = scale_range
        self.jitter_sigma = jitter_sigma
        self.augment      = augment


    def __len__(self):
        
        return len(self.filter_json_files)

    def __getitem__(self, idx):


        jf_path = self.filter_json_files[idx]
        with open(jf_path, "rb") as f:
            graph_data = orjson.loads(f.read())

        # ── 2) load coplanarity matrix & image path ──
        coplanarity_matrix = _load_array_from_json_or_npy("coplanarity_matrix", graph_data)
        file_path_img      = "../" + graph_data.get("file_path", None)
        # file_path_img      = graph_data.get("file_path", None)

        img = _load_image(filepath=file_path_img, color_conversion=cv2.COLOR_BGR2RGB)
        if img is None:
            raise ValueError(f"Could not load image at {file_path_img}")

        # ── 3) load per-line features + raw coordinates + labels ──
        lines = graph_data.get("lines", [])
        if not lines:
            raise ValueError("No lines found in JSON.")

        feats, labels, line_coords = [], [], []
        for ln in lines:
            emb = _load_array_from_json_or_npy("embedding_DeepLSD", ln, desc="Line embedding")
            feats.append(emb)
            labels.append(ln.get("struct_score", 0.0))
            line_coords.append(ln["coordinates"])
        x_emb = torch.tensor(np.vstack(feats), dtype=torch.float)       # (N, D_emb)
        coords_raw = np.array(line_coords, dtype=np.float32)            # (N, 2, 2)
        y       = torch.tensor(labels, dtype=torch.float).unsqueeze(1)   # (N, 1)
        N       = x_emb.size(0)

       # ── 2) If augment=False, skip rotation/scale/jitter entirely ──
        if not self.augment:
            # Directly convert to torch tensors and sample normally, no transform:
            img_np = img
            img_t  = (
                torch.tensor(img_np, dtype=torch.float32)
                     .div(255.0)
                     .permute(2, 0, 1)  # → (C, H, W)
            )
            coords_torch_aug = torch.tensor(coords_raw, dtype=torch.float32)  # (N, 2, 2)
        else:
            # ── 1) Random flips ──
            img_aug = img.copy()
            H, W = img_aug.shape[:2]
            if random.random() < 0.5:
                img_aug = cv2.flip(img_aug, 1)  # horizontal flip
                coords_raw[..., 0] = (W - 1) - coords_raw[..., 0]
            if random.random() < 0.5:
                img_aug = cv2.flip(img_aug, 0)  # vertical flip
                coords_raw[..., 1] = (H - 1) - coords_raw[..., 1]

            # ── 2) Random perspective warp ──
            #  perturb each corner by ±5% of min(H,W)
            delta = 0.05 * min(H, W)
            pts1 = np.float32([[0,0], [W,0], [W,H], [0,H]])
            pts2 = pts1 + np.random.uniform(-delta, delta, size=(4,2)).astype(np.float32)
            M_persp = cv2.getPerspectiveTransform(pts1, pts2)
            img_aug = cv2.warpPerspective(
                img_aug, M_persp, (W, H),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )
            # transform line endpoints
            endpoints = coords_raw.reshape(-1,1,2).astype(np.float32)  # (2N,1,2)
            endpoints = cv2.perspectiveTransform(endpoints, M_persp).reshape(-1,2)

            # ── 3) Random rotation & scale ──
            angle = random.uniform(-30.0, 30.0)  # ±30°
            scale = random.uniform(0.8, 1.2)     # [0.8,1.2]
            cx, cy = W/2.0, H/2.0
            M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
            img_aug = cv2.warpAffine(
                img_aug, M, (W, H),
                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
            )
            # apply to endpoints
            endpoints = cv2.transform(endpoints.reshape(-1,1,2), M).reshape(-1,2)

            # ── 4) Add endpoint jitter ──
            jitter = np.random.normal(0, self.jitter_sigma*2, size=endpoints.shape).astype(np.float32)
            endpoints += jitter

            # reshape back to (N,2,2)
            coords_aug = endpoints.reshape(N,2,2)

            # ── 5) Color jitter + blur + noise ──
            # Convert to PIL for ColorJitter
            img_pil = T.ToPILImage()(img_aug)
            color_aug = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
            img_pil = color_aug(img_pil)
            img_aug = np.array(img_pil)

            if random.random() < 0.3:
                sigma = random.uniform(0.1, 1.0)
                img_aug = cv2.GaussianBlur(img_aug, (5,5), sigma)
            if random.random() < 0.2:
                noise = np.random.normal(0, 10, img_aug.shape).astype(np.uint8)
                img_aug = cv2.add(img_aug, noise)

            # ── 6) Final torch conversions ──
            img_t = (
                torch.tensor(img_aug, dtype=torch.float32)
                    .div(255.0)
                    .permute(2, 0, 1)
            )
            coords_torch_aug = torch.tensor(coords_aug, dtype=torch.float32)
            
        if self.method == "roi":
            roi_features = extract_line_feature_ROIAlign(
                img=img_t,                     # H×W×3  NumPy
                lines=coords_aug.tolist(),       # list of [[x1,y1],[x2,y2]]
                output_size=self.roi_output_size,
                plot_results=False
            )
        else:  # “sample”
            roi_features = self.sampler.sample_lines_grid(
                img=img_t,           # (C,H,W) torch
                lines=coords_torch_aug,  # (N,2,2) torch
                align_corners=True
            )

        if roi_features is None or roi_features.shape[0] != N:
            logging.warning(
                f"ROI feature issue (aug#={idx}) for {jf_path}; "
                f"got {None if roi_features is None else roi_features.shape}, expected ({N}, …)."
            )
            raise ValueError("ROI feature extraction failed (after augmentation).")

     
        img_H, img_W = img_t.shape[-2:]                                  # after your ToTensor
        geo = line_geometry(coords_torch_aug, img_size=(img_H, img_W))


        

        p1, p2 = coords_torch_aug[:, 0], coords_torch_aug[:, 1]
        D = seg_seg_dist(p1, p2, p1, p2)  # (N,N)
        k = min(10, N - 1)
        k_global = min (50, N - 1)
        knn = D.topk(k+1, largest=False).indices[:, 1:]  # (N, k)
        knn_global = D.topk(k_global+1, largest=False).indices[:, 1:]  # (N, k)

        # 11a) local edges:
        src = torch.arange(N, device=coords_torch_aug.device).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = knn.reshape(-1)
        local_edge_index = torch.stack([src, dst], dim=0)  # (2, N*k)
        
        # 11a) local edges:
        src_global = torch.arange(N, device=coords_torch_aug.device).unsqueeze(1).expand(-1, k_global).reshape(-1)
        dst_global = knn_global.reshape(-1)
        global_edge_index = torch.stack([src_global, dst_global], dim=0)  # (2, N*k)

        lines_i = coords_torch_aug[src_global]  # (N*k, 2,2)
        lines_j = coords_torch_aug[dst_global]  # (N*k, 2,2)
        quads = torch.stack([
            lines_i[:, 0],  # start_i
            lines_i[:, 1],  # end_i
            lines_j[:, 1],  # end_j
            lines_j[:, 0],  # start_j
        ], dim=1)          # (N*k, 4,2)

        edge_attr = self.edge_sampler(img_t, quads)  # (N*k, C_edge, H_edge, W_edge)

        # 11b) full edges (all pairs):
        full_edge_index = []
        full_edge_labels = []
        for i2 in range(N):
            for j2 in range(N):
                full_edge_index.append([i2, j2])
                full_edge_labels.append(coplanarity_matrix[i2][j2])
        full_edge_index = torch.tensor(full_edge_index, dtype=torch.long).t().contiguous()
        full_edge_labels = torch.tensor(full_edge_labels, dtype=torch.float).unsqueeze(1)

        # 11c) find cliques (unchanged):
        cliques = adjacency_to_overlapping_clusters(coplanarity_matrix)

        # ── 12) return the Data object ──
        return Data(
            x = x_emb,                       # (N, D_emb)
            y = y,                           # (N, 1)
            plane_id = cliques,              # List[List[int]]
            coordinates = coords_torch_aug,  # (N, 2, 2)
            geo = geo,                       # (N, 5)
            edge_index = local_edge_index,   # (2, N*k)
            global_edge_index = global_edge_index,
            full_edge_index = full_edge_index,
            full_edge_labels = full_edge_labels,
            roi_features = roi_features,     # (N, C, num_samples, width)
            edge_attr = edge_attr,           # (N*k, C_edge, H_edge, W_edge)
            img_path = file_path_img
        )
