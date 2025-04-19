import os
import json
import numpy as np
import torch
import cv2
from torch_geometric.data import Data, Dataset
from ground_truth.dataloader import _load_image
from models.dataset_utils import extract_line_feature_ROIAlign

def get_line_feature(image_path, line_coordinates, output_size = (64,64)):
    """
    ADD MATINE'S FEATURE EXTRACTION
    """
    img = _load_image(filepath = image_path, color_conversion=cv2.COLOR_BGR2RGB)
    roi_results = extract_line_feature_ROIAlign(img=img,lines=line_coordinates,output_size=output_size,plot_results=True)
    return roi_results

class GraphDatasetInductive(Dataset):
    def __init__(self, json_dir, roi_output_size=(64, 64)):
        super().__init__()
        self.json_files = [
            os.path.join(json_dir, f)
            for f in os.listdir(json_dir)
            if f.endswith('.json')
        ]
        self.roi_output_size = roi_output_size
        # self.json_files = self.json_files[:20] # Limit for testing

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        # load JSON
        with open(self.json_files[idx], 'r') as f:
            graph_data = json.load(f)

        lines = graph_data.get('lines', [])
        if not lines: # Handle cases with no lines
             # Return an empty Data object or None, depending on desired behavior
             raise ValueError('lines from the graph_data is empty')
             

        coplanarity_matrix = graph_data.get('coplanarity_matrix', [[]])
        file_path_img = graph_data.get('file_path')

        # build node features (embeddings) + labels
        feats, labels,line_coords = [], [], []
        for ln in lines:
            emb = np.array(ln['embedding_DeepLSD'])
            score = ln.get('struct_score', 0.5)
            line_coords.append(ln.get('coordinates'))
            feats.append(emb)
            labels.append(score)

        x_emb = torch.tensor(np.vstack(feats), dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.float).unsqueeze(1)
        N = x_emb.size(0)

        # Extract ROI Align features
        line_coords_np = np.array(line_coords)
        img = _load_image(filepath=file_path_img, color_conversion=cv2.COLOR_BGR2RGB)
        # roi_features shape: (N, C, H, W)
        roi_features = extract_line_feature_ROIAlign(
            img=img,
            lines=line_coords_np,
            output_size=self.roi_output_size,
            plot_results=False # Typically False during training/loading
        )

        # Handle potential mismatch or failure in ROI extraction
        if roi_features is None or roi_features.shape[0] != N:
             print(f"Warning: ROI feature issue for {self.json_files[idx]}. Features shape: {roi_features.shape if roi_features is not None else 'None'}, Expected N: {N}. Skipping graph.")
             raise ValueError('lines from the graph_data is empty')


        # fully connected graph + coplanarity labels
        edge_list, edge_labels = [], []
        for i in range(N):
            for j in range(N):
                edge_labels.append(coplanarity_matrix[i][j])
                # if i == j: continue
                edge_list.append([i, j])

        edge_index  = torch.tensor(edge_list,  dtype=torch.long).t().contiguous()
        edge_labels = torch.tensor(edge_labels, dtype=torch.float).unsqueeze(1)

        # Create Data object, storing roi_features as a separate attribute
        data_object = Data(x=x_emb, y=y, edge_index=edge_index, edge_labels=edge_labels, roi_features=roi_features)

        return data_object
    
