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

class GraphDataset(Dataset):
    def __init__(self, json_dir, struct_thresh=0.8, textural_thresh=0.6):

        self.json_files = [
            os.path.join(json_dir, f)
            for f in os.listdir(json_dir)
            if f.endswith('.json')
        ]
        self.struct_thresh = struct_thresh
        self.textural_thresh = textural_thresh

    def __len__(self):

        return len(self.json_files)

    def __getitem__(self, idx):

        json_file = self.json_files[idx]
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        image_id = data.get('image_id')
        lines = data.get('lines', [])
        
        node_features = []
        node_labels = []
        
        for line in lines:
            score = line.get("confidence_score", 0.5)
            if score > self.struct_thresh:
                node_features.append(get_line_feature(image_id, line["coordinates"]))
                node_labels.append(1)
            elif score < self.textural_thresh:
                node_features.append(get_line_feature(image_id, line["coordinates"]))
                node_labels.append(0)
        

        node_features = np.vstack(node_features)
        node_labels = np.array(node_labels)
        num_nodes = node_features.shape[0]


        # fully connected graph
        edge_list = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_list.append([i, j])
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()


        data_object = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            y=torch.tensor(node_labels, dtype=torch.long),
            edge_index=edge_index
        )
        data_object.image_id = image_id  
        return data_object
