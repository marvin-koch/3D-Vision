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
        coplanarity_matrix = data.get('coplanarity_matrix',[[]])
        
        node_features = []
        node_labels = []
        
        for line in lines:
            score = line.get("confidence_score", 0.5)
            if score > self.struct_thresh:
                #print(f"DeepLsd feature of size {np.array(line.get('embedding_DeepLSD')).shape}")
                #print(np.array(line.get('embedding_DeepLSD')))
                node_features.append(np.array(line.get('embedding_DeepLSD')))
                node_labels.append(1)
            elif score < self.textural_thresh:
                node_features.append(np.array(line.get('embedding_DeepLSD')))
                node_labels.append(0)
        

        node_features = np.vstack(node_features)
        node_labels = np.array(node_labels)
        num_nodes = node_features.shape[0]


        # fully connected graph
        edge_index = []
        edge_labels = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_index.append([i, j])
                    edge_labels.append(coplanarity_matrix[i][j])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_labels = torch.tensor(edge_labels, dtype=torch.long).t().contiguous()
 
        # Masks
        indices = np.arange(num_nodes)
        np.random.shuffle(indices)

        train_end = int(0.6 * num_nodes)
        val_end = int(0.8 * num_nodes)

        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[indices[:train_end]] = True
        val_mask[indices[train_end:val_end]] = True
        test_mask[indices[val_end:]] = True


        # the Data object
        data_object = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            y=torch.tensor(node_labels, dtype=torch.long),
            edge_index=edge_index
        )
        data_object.image_id = image_id  
        data_object.train_mask = train_mask
        data_object.val_mask = val_mask
        data_object.test_mask = test_mask
        data_object.edge_labels = edge_labels

        return data_object