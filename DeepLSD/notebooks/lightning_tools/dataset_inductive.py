import os
import json
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset
from torch_geometric.data import Data
from line_descriptor import extract_resized_line_bands, LineSampler, EdgeSampler

# Helper to load images




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
    return torch.cat([mid, dir_u, length, theta, cos2, sin2], dim=1)


class GraphDatasetInductive(Dataset):
    def __init__(
        self,
        h5_path: str,
        roi_output_size=(64, 64),
        method="sample",
        device=None,
        edge_sample_size=(32,16)
    ):
        import os
        os.chdir("/mnt/c/Users/shan2/Documents/ETH/MA4/3DV/GitHub/3D-Vision/DeepLSD/notebooks")
        super().__init__()
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        
 
        # Open HDF5 in read-only mode
        self.h5 = h5py.File(h5_path, 'r')
        # List all sample groups
        all_groups = sorted(self.h5.keys())
        valid = []
        for g in all_groups:
            meta = json.loads(self.h5[g]['metadata'][()].decode() 
                              if isinstance(self.h5[g]['metadata'][()], bytes) 
                              else self.h5[g]['metadata'][()])
            fp = meta.get('file_path')
            lines_meta = meta.get('lines', [])

            if fp and os.path.isfile(fp) and len(lines_meta) > 0:
                valid.append(g)
            else:
                # you could also log.warning(f"Skipping {g}: missing {fp}")
                continue
        self.groups = valid

        self.roi_output_size = roi_output_size
        self.method = method
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_edge, width_edge = edge_sample_size
        self.edge_sampler = EdgeSampler(num_samples_u=num_edge, num_samples_v=width_edge)
        if self.method == "sample":
            num_s, width = self.roi_output_size
            self.sampler = LineSampler(num_samples=num_s, width=width)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        grp_name = self.groups[idx]
        grp = self.h5[grp_name]
        # metadata
        meta = json.loads(grp['metadata'][()].decode('utf-8') 
                          if isinstance(grp['metadata'][()], bytes) 
                          else grp['metadata'][()])
        # load coplanarity and features
        copla = torch.from_numpy(grp['coplanarity'][()])      # (N,N)
        distance_field = torch.from_numpy(grp['distance_field'][()])  
        angle_field = torch.from_numpy(grp['angle_field'][()])   

        downsample_h = meta.get('downsample_h', 1)
        downsample_w = meta.get('downsample_w', 1)

        downsample_h = 1
        downsample_w = 1



        # load image
        img = _load_image(meta.get('file_path'), color_conversion=cv2.COLOR_BGR2RGB)
        img_np = img

        # lines metadata
        lines_meta = meta['lines']
        labels = [float(x['struct_score']) for x in lines_meta]
        coords = [x['coordinates'] for x in lines_meta]



        coords = torch.tensor(coords, dtype=torch.float)
        
        
        if coords.dim() == 1 and coords.numel() == 4:
            # single line case: [4] -> [1,2,2]
            coords = coords.view(1, 2, 2)
        elif coords.dim() == 2 and coords.size(1) == 4:
            # multiple lines flattened: [N,4] -> [N,2,2]
            coords = coords.view(-1, 2, 2)
        elif coords.dim() != 3 or coords.size(1) != 2 or coords.size(2) != 2:
            raise ValueError(f"line_geometry expected tensor of shape (N,2,2), got {tuple(coords.shape)}")


        y = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
        N = coords.size(0)


      
        # extract image strips
        patches = extract_resized_line_bands(
            img=img_np,
            angle_field=angle_field,
            distance_field=distance_field,
            lines=coords.tolist(),
            width=20,
            target_length=20,
            downsampling_h=downsample_h,
            downsampling_w=downsample_w,
        )

        rgb_patches        = patches["rgb"]
        angle_field_patches= patches["angle_field"]
        distance_patches   = patches["distance_field"]

        
     

        # geometry and full graph
        geo = line_geometry(coords)
        # compute pairwise distances and full edge list
        p1, p2 = coords[:,0], coords[:,1]
        # seg-seg distance as in original code
        def seg_seg_dist(p1,p2,q1,q2,eps=1e-8):
            P1,P2 = p1[:,None], p2[:,None]
            Q1,Q2 = q1[None], q2[None]
            def proj(X,A,B):
                t = torch.clamp(((X-A)*(B-A)).sum(-1,keepdim=True) /
                                (((B-A)**2).sum(-1,keepdim=True)+eps), 0,1)
                return A + t*(B-A)
            d1 = ((Q1 - proj(Q1,P1,P2))**2).sum(-1)
            d2 = ((Q2 - proj(Q2,P1,P2))**2).sum(-1)
            d3 = ((P1 - proj(P1,Q1,Q2))**2).sum(-1)
            d4 = ((P2 - proj(P2,Q1,Q2))**2).sum(-1)
            return torch.sqrt(torch.min(torch.min(d1,d2), torch.min(d3,d4)))
        D = seg_seg_dist(p1,p2,p1,p2)
        # full edges
        full_idx, full_lbl = [], []
        for i in range(N):
            for j in range(N):
                full_idx.append([i,j])
                full_lbl.append(copla[i,j].item())
        full_edge_index = torch.tensor(full_idx, dtype=torch.long).t().contiguous()
        full_edge_labels = torch.tensor(full_lbl, dtype=torch.float).unsqueeze(1)

        # k-NN local edges
        k=7
        N = D.size(1)                        # number of nodes in this graph
        k = min(k, N - 1)           # don’t ask for more than N-1 neighbors
        knn = D.topk(k+1, largest=False).indices[:,1:]
        src = torch.arange(N).unsqueeze(1).expand(-1,k).reshape(-1)
        dst = knn.reshape(-1)
        local_edge_index = torch.stack([src,dst], dim=0)

        rgb_patches = torch.from_numpy(np.array(rgb_patches))
        angle_field_patches = torch.from_numpy(np.array(angle_field_patches))
        distance_patches = torch.from_numpy(np.array(distance_patches))

        
        return Data(
            y=y,
            num_nodes = y.size(0),
            coordinates=coords,
            geo=geo,
            edge_index=local_edge_index,
            full_edge_index=full_edge_index,
            full_edge_labels=full_edge_labels,
            rgb_patches=rgb_patches,
            angle_field_patches=angle_field_patches,
            distance_patches=distance_patches,
        )

# Example instantiation:
# ds = GraphDatasetHDF5(h5_path="output/lines_data.h5")
