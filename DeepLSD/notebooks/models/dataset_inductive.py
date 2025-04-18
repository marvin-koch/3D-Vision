import os
import json
import numpy as np
import torch
from torch_geometric.data import Data, Dataset

def get_line_feature(image_id, coordinates):
    """
    ADD MATINE'S FEATURE EXTRACTION
    """
    
    return np.ones(10)

class GraphDatasetInductive(Dataset):
    def __init__(self, json_dir):
        super().__init__()
        self.json_files = [
            os.path.join(json_dir, f)
            for f in os.listdir(json_dir)
            if f.endswith('.json')
        ]
        #self.json_files = self.json_files[:100]  # Limit to 100 files for testing


    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        # load JSON
        with open(self.json_files[idx], 'r') as f:
            data = json.load(f)

        lines = data.get('lines', [])
        coplanarity_matrix = data.get('coplanarity_matrix', [[]])

        # build node features + labels
        feats, labels = [], []
        for ln in lines:
            emb = np.array(ln['embedding_DeepLSD'])
            score = ln.get('struct_score', 0.5)
            feats.append(emb)
            labels.append(score)

        x = torch.tensor(np.vstack(feats), dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
        N = x.size(0)

        # fully connected graph + coplanarity labels
        edge_list, edge_labels = [], []
        for i in range(N):
            for j in range(N):
                if i == j: continue
                edge_list.append([i, j])
                edge_labels.append(coplanarity_matrix[i][j])

        edge_index  = torch.tensor(edge_list,  dtype=torch.long).t().contiguous()
        edge_labels = torch.tensor(edge_labels, dtype=torch.float).unsqueeze(1)

        data_object = Data(x=x, y=y, edge_index=edge_index, edge_labels=edge_labels)

        return data_object
    
