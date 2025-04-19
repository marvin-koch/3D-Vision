import os
import json
import numpy as np
import torch
from torch_geometric.data import Data, Dataset


class GraphDatasetInference(Dataset):
    def __init__(self, embeddings_list):
        super().__init__()
        self.embeddings_list = embeddings_list    # List of line embeddings

    def __len__(self):
        return len(self.embeddings_list)

    def __getitem__(self, idx):
        # Extract embeddings
        embedding = self.embeddings_list[idx]
        feats = []
        for emb in embedding:
            emb = np.array(emb)
            feats.append(emb)

        x = torch.tensor(np.vstack(feats), dtype=torch.float)
        N = x.size(0)

        # fully connected graph
        edge_list = []
        for i in range(N):
            for j in range(N):
                #if i == j: continue
                edge_list.append([i, j])

        edge_index  = torch.tensor(edge_list,  dtype=torch.long).t().contiguous()

        data_object = Data(x=x, edge_index=edge_index)

        return data_object