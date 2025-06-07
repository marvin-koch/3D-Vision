import os
import json
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset
from torch_geometric.data import Data
import numpy as np
import networkx as nx
from typing import List
import numpy as np
import networkx as nx
from typing import List

from line_descriptor import extract_resized_line_bands, LineSampler, EdgeSampler

# Helper to load images



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



def _load_image(filepath: str, color_conversion: int = None) -> np.ndarray:
    if not filepath or not os.path.exists(filepath):
        raise FileNotFoundError(f"Image file not found: {filepath}")
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cv2.imread failed for: {filepath}")
    if color_conversion is not None:
        img = cv2.cvtColor(img, color_conversion)
    return img


def line_geometry(line_pts: torch.Tensor) -> torch.Tensor:
    p1, p2 = line_pts[:,0], line_pts[:,1]
    mid     = 0.5 * (p1 + p2)                       # (mx,my)

    vec = p2 - p1
    length = vec.norm(dim=1, keepdim=True)
    dir_u = F.normalize(vec, dim=1)
      # orientation
    # orientation (as column vectors)
    theta = torch.atan2(dir_u[:,1], dir_u[:,0]).unsqueeze(1)  # (N,1)
    cos2  = torch.cos(2 * theta)                             # (N,1)
    sin2  = torch.sin(2 * theta)                             # (N,1)
    return torch.cat([mid, length, theta, cos2, sin2, dir_u], dim=1)


# class GraphDatasetInductive(Dataset):
#     def __init__(
#         self,
#         h5_path: str,
#         roi_output_size=(64, 64),
#         method="sample",
#         device=None,
#         edge_sample_size=(32,16)
#     ):
#         import os
#         os.chdir("/mnt/c/Users/shan2/Documents/ETH/MA4/3DV/GitHub/3D-Vision/DeepLSD/notebooks")
#         super().__init__()
#         if not os.path.exists(h5_path):
#             raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        
 
#         # Open HDF5 in read-only mode
#         self.h5 = h5py.File(h5_path, 'r')
#         # List all sample groups
#         all_groups = sorted(self.h5.keys())
#         valid = []
#         for g in all_groups:
#             meta = json.loads(self.h5[g]['metadata'][()].decode() 
#                               if isinstance(self.h5[g]['metadata'][()], bytes) 
#                               else self.h5[g]['metadata'][()])
#             fp = meta.get('file_path')
#             lines_meta = meta.get('lines', [])

#             if fp and os.path.isfile(fp) and len(lines_meta) > 0:
#                 valid.append(g)
#             else:
#                 # you could also log.warning(f"Skipping {g}: missing {fp}")
#                 continue
#         self.groups = valid

#         self.roi_output_size = roi_output_size
#         self.method = method
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         num_edge, width_edge = edge_sample_size
#         self.edge_sampler = EdgeSampler(num_samples_u=num_edge, num_samples_v=width_edge)
#         if self.method == "sample":
#             num_s, width = self.roi_output_size
#             self.sampler = LineSampler(num_samples=num_s, width=width)

#     def __len__(self):
#         return len(self.groups)

#     def __getitem__(self, idx):
#         grp_name = self.groups[idx]
#         grp = self.h5[grp_name]
#         # metadata
#         meta = json.loads(grp['metadata'][()].decode('utf-8') 
#                           if isinstance(grp['metadata'][()], bytes) 
#                           else grp['metadata'][()])
#         # load coplanarity and features
#         copla = torch.from_numpy(grp['coplanarity'][()])      # (N,N)
#         distance_field = torch.from_numpy(grp['distance_field'][()])  
#         angle_field = torch.from_numpy(grp['angle_field'][()])   

#         downsample_h = meta.get('downsample_h', 1)
#         downsample_w = meta.get('downsample_w', 1)

#         downsample_h = 1
#         downsample_w = 1



#         # load image
#         img = _load_image(meta.get('file_path'), color_conversion=cv2.COLOR_BGR2RGB)
#         img_np = img

#         # lines metadata
#         lines_meta = meta['lines']
#         labels = [float(x['struct_score']) for x in lines_meta]
#         coords = [x['coordinates'] for x in lines_meta]



#         coords = torch.tensor(coords, dtype=torch.float)
        
        
#         if coords.dim() == 1 and coords.numel() == 4:
#             # single line case: [4] -> [1,2,2]
#             coords = coords.view(1, 2, 2)
#         elif coords.dim() == 2 and coords.size(1) == 4:
#             # multiple lines flattened: [N,4] -> [N,2,2]
#             coords = coords.view(-1, 2, 2)
#         elif coords.dim() != 3 or coords.size(1) != 2 or coords.size(2) != 2:
#             raise ValueError(f"line_geometry expected tensor of shape (N,2,2), got {tuple(coords.shape)}")


#         y = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
#         N = coords.size(0)



      
#         # extract image strips
#         patches = extract_resized_line_bands(
#             img=img_np,
#             angle_field=angle_field,
#             distance_field=distance_field,
#             lines=coords.tolist(),
#             width=20,
#             target_length=20,
#             downsampling_h=downsample_h,
#             downsampling_w=downsample_w,
#         )

#         rgb_patches        = patches["rgb"]
#         angle_field_patches= patches["angle_field"]
#         distance_patches   = patches["distance_field"]

        
     

#         # geometry and full graph
#         geo = line_geometry(coords)
#         # compute pairwise distances and full edge list
#         p1, p2 = coords[:,0], coords[:,1]
#         # seg-seg distance as in original code
#         def seg_seg_dist(p1,p2,q1,q2,eps=1e-8):
#             P1,P2 = p1[:,None], p2[:,None]
#             Q1,Q2 = q1[None], q2[None]
#             def proj(X,A,B):
#                 t = torch.clamp(((X-A)*(B-A)).sum(-1,keepdim=True) /
#                                 (((B-A)**2).sum(-1,keepdim=True)+eps), 0,1)
#                 return A + t*(B-A)
#             d1 = ((Q1 - proj(Q1,P1,P2))**2).sum(-1)
#             d2 = ((Q2 - proj(Q2,P1,P2))**2).sum(-1)
#             d3 = ((P1 - proj(P1,Q1,Q2))**2).sum(-1)
#             d4 = ((P2 - proj(P2,Q1,Q2))**2).sum(-1)
#             return torch.sqrt(torch.min(torch.min(d1,d2), torch.min(d3,d4)))
#         D = seg_seg_dist(p1,p2,p1,p2)
#         # full edges
#         full_idx, full_lbl = [], []
#         for i in range(N):
#             for j in range(N):
#                 full_idx.append([i,j])
#                 full_lbl.append(copla[i,j].item())
#         full_edge_index = torch.tensor(full_idx, dtype=torch.long).t().contiguous()
#         full_edge_labels = torch.tensor(full_lbl, dtype=torch.float).unsqueeze(1)

#         # k-NN local edges
#         k=30
#         N = D.size(1)                        # number of nodes in this graph
#         k = min(k, N - 1)           # don’t ask for more than N-1 neighbors
#         knn = D.topk(k+1, largest=False).indices[:,1:]
#         src = torch.arange(N).unsqueeze(1).expand(-1,k).reshape(-1)
#         dst = knn.reshape(-1)
#         local_edge_index = torch.stack([src,dst], dim=0)

#         rgb_patches = torch.from_numpy(np.array(rgb_patches))
#         angle_field_patches = torch.from_numpy(np.array(angle_field_patches))
#         distance_patches = torch.from_numpy(np.array(distance_patches))

#         img = cv2.resize(img, self.roi_output_size, interpolation=cv2.INTER_LINEAR)


#         cliques = adjacency_to_overlapping_clusters(copla)


        
#         return Data(
#             y=y,
#             num_nodes = y.size(0),
#             coordinates=coords,
#             plane_id=cliques,

#             geo=geo,
#             edge_index=local_edge_index,
#             full_edge_index=full_edge_index,
#             full_edge_labels=full_edge_labels,
#             img=torch.tensor(img),
#             rgb_patches=rgb_patches,
#             angle_field_patches=angle_field_patches,
#             distance_patches=distance_patches,
#         )

# # Example instantiation:
# ds = GraphDatasetHDF5(h5_path="output/lines_data.h5")




class GraphDatasetInductive(Dataset):
    def __init__(
        self,
        h5_path: str,
        roi_output_size=(64, 64),
        method="sample",
        device=None,
        edge_sample_size=(32, 16),
        # ── NEW ARGUMENTS ──
        augment: bool = False,
        rot_range: float = 15.0,
        scale_range: float = 0.1,
        jitter_sigma: float = 1.0,
    ):
        super().__init__()
        import os
        os.chdir("/mnt/c/Users/shan2/Documents/ETH/MA4/3DV/GitHub/3D-Vision/DeepLSD/notebooks")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

        # Open HDF5 in read-only mode
        self.h5 = h5py.File(h5_path, "r")
        all_groups = sorted(self.h5.keys())

        valid = []
        for g in all_groups:
            meta = json.loads(
                self.h5[g]["metadata"][()].decode()
                if isinstance(self.h5[g]["metadata"][()], bytes)
                else self.h5[g]["metadata"][()]
            )
            fp = meta.get("file_path")
            lines_meta = meta.get("lines", [])
            if fp and os.path.isfile(fp) and len(lines_meta) > 0:
                valid.append(g)

        self.groups = valid

        self.roi_output_size = roi_output_size
        self.method = method
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_e, w_e = edge_sample_size
        self.edge_sampler = EdgeSampler(num_samples_u=num_e, num_samples_v=w_e)
        if self.method == "sample":
            num_s, w = self.roi_output_size
            self.sampler = LineSampler(num_samples=num_s, width=w)

        # ── store augmentation params ──
        self.augment = augment
        self.rot_range = rot_range
        self.scale_range = scale_range
        self.jitter_sigma = jitter_sigma

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        grp_name = self.groups[idx]
        grp = self.h5[grp_name]

        # --- 1) Load metadata, coplanarity, fields, image, and lines ---
        meta = json.loads(
            grp['metadata'][()].decode('utf-8')
            if isinstance(grp['metadata'][()], bytes)
            else grp['metadata'][()]
        )
        copla = torch.from_numpy(grp['coplanarity'][()])       # (N,N)
        distance_field = grp['distance_field'][()]              # (H_full, W_full)
        angle_field = grp['angle_field'][()]                    # (H_full, W_full)

        img = _load_image(meta.get('file_path'), color_conversion=cv2.COLOR_BGR2RGB)
        img_np = img  # (H_full, W_full, 3)

        lines_meta = meta['lines']
        labels = [float(x['struct_score']) for x in lines_meta]
        coords = [x['coordinates'] for x in lines_meta]
        coords = torch.tensor(coords, dtype=torch.float)
        if coords.dim() == 1 and coords.numel() == 4:
            coords = coords.view(1, 2, 2)
        elif coords.dim() == 2 and coords.size(1) == 4:
            coords = coords.view(-1, 2, 2)
        N = coords.size(0)
        y = torch.tensor(labels, dtype=torch.float).unsqueeze(1)  # (N,1)

        # --- 2) AUGMENTATION: full‐image warp (if augment=True) ---
        if self.augment:
            angle = float(np.random.uniform(-self.rot_range, self.rot_range))
            scale = float(np.random.uniform(1.0 - self.scale_range,
                                            1.0 + self.scale_range))
            H_full, W_full = img_np.shape[:2]
            cx, cy = W_full / 2.0, H_full / 2.0
            M = cv2.getRotationMatrix2D(center=(cx, cy), angle=angle, scale=scale)

            # small translation jitter
            tx = float(np.random.normal(0, self.jitter_sigma))
            ty = float(np.random.normal(0, self.jitter_sigma))
            M[0, 2] += tx
            M[1, 2] += ty

            # warp full RGB image
            img_np = cv2.warpAffine(
                img_np,
                M,
                dsize=(W_full, H_full),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )

            # warp per‐pixel angle_field & distance_field
            angle_np = angle_field.astype(np.float32)
            dist_np = distance_field.astype(np.float32)
            angle_field = cv2.warpAffine(
                angle_np,
                M,
                dsize=(W_full, H_full),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )
            distance_field = cv2.warpAffine(
                dist_np,
                M,
                dsize=(W_full, H_full),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )

            # transform each line endpoint
            endpoints = coords.reshape(-1, 2).numpy().astype(np.float32)  # (2N,2)
            endpoints = endpoints.reshape(-1, 1, 2)                        # (2N,1,2)
            endpoints_tf = cv2.transform(endpoints, M).reshape(-1, 2)    # (2N,2)
            coords = torch.from_numpy(endpoints_tf.reshape(N, 2, 2))     # (N,2,2)

        # --- 3) Extract image strips from (possibly‐warped) scene ---
        patches = extract_resized_line_bands(
            img=img_np,
            angle_field=angle_field,
            distance_field=distance_field,
            lines=coords.tolist(),
            width=20,
            target_length=20,
            downsampling_h=1,
            downsampling_w=1,
        )
        rgb_patches        = patches["rgb"]            # list of N (h_p, w_p, 3)
        angle_field_patches= patches["angle_field"]    # list of N (h_p, w_p)
        distance_patches   = patches["distance_field"] # list of N (h_p, w_p)

        # ──────────────── CONVERT TO TORCH with “channel‐last” for single channel ────────────────
        rgb_patches = torch.from_numpy(np.array(rgb_patches))                 # (N, h_p, w_p, 3)
        angle_field_patches = torch.from_numpy(np.array(angle_field_patches))
        distance_patches = torch.from_numpy(np.array(distance_patches))

        # --- 4) Rest of your original code exactly as before ---
        geo = line_geometry(coords)  # (N,8)

        def seg_seg_dist(p1, p2, q1, q2, eps=1e-8):
            P1, P2 = p1[:, None], p2[:, None]
            Q1, Q2 = q1[None], q2[None]
            def proj(X, A, B):
                t = torch.clamp(
                    ((X - A) * (B - A)).sum(-1, keepdim=True)
                    / (((B - A) ** 2).sum(-1, keepdim=True) + eps),
                    0, 1
                )
                return A + t * (B - A)
            d1 = ((Q1 - proj(Q1, P1, P2)) ** 2).sum(-1)
            d2 = ((Q2 - proj(Q2, P1, P2)) ** 2).sum(-1)
            d3 = ((P1 - proj(P1, Q1, Q2)) ** 2).sum(-1)
            d4 = ((P2 - proj(P2, Q1, Q2)) ** 2).sum(-1)
            return torch.sqrt(torch.min(torch.min(d1, d2), torch.min(d3, d4)))

        p1, p2 = coords[:, 0], coords[:, 1]
        D = seg_seg_dist(p1, p2, p1, p2)
        full_idx, full_lbl = [], []
        for i2 in range(N):
            for j2 in range(N):
                full_idx.append([i2, j2])
                full_lbl.append(copla[i2, j2].item())
        full_edge_index = torch.tensor(full_idx, dtype=torch.long).t().contiguous()  # (2, N*N)
        full_edge_labels = torch.tensor(full_lbl, dtype=torch.float).unsqueeze(1)     # (N*N,1)

        k = 30
        k = min(k, N - 1)
        knn = D.topk(k + 1, largest=False).indices[:, 1:]
        src = torch.arange(N).unsqueeze(1).expand(-1, k).reshape(-1)
        dst = knn.reshape(-1)
        local_edge_index = torch.stack([src, dst], dim=0)

        # Convert rgb_patches and single‐channel patches to floats if needed downstream
        rgb_patches = rgb_patches.float() / 255.0              # (N, h_p, w_p, 3)
        angle_field_patches = angle_field_patches.float()     # (N, h_p, w_p, 1)
        distance_patches = distance_patches.float()           # (N, h_p, w_p, 1)

        # resize full image for ROI (as before)
        img = cv2.resize(img_np, self.roi_output_size, interpolation=cv2.INTER_LINEAR)

        cliques = adjacency_to_overlapping_clusters(copla.numpy())


        return Data(
            y=y,
            num_nodes = y.size(0),
            coordinates=coords,
            plane_id=cliques,

            geo=geo,
            edge_index=local_edge_index,
            full_edge_index=full_edge_index,
            full_edge_labels=full_edge_labels,
            img=torch.tensor(img),                 # (H_roi, W_roi, 3) same as before
            rgb_patches=rgb_patches,               # (N, h_p, w_p, 3)
            angle_field_patches=angle_field_patches,# (N, h_p, w_p, 1)
            distance_patches=distance_patches      # (N, h_p, w_p, 1)
        )
