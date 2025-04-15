import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class GATClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, act):
        super().__init__()
        # GAT to produce node embeddings
        self.gat = pyg_nn.GAT(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            out_channels=out_channels, 
            num_layers=num_layers,
            dropout=dropout,
            act=act
        )
        
        # MLP for node-level (textural/structural) classification
        self.mlp_textural_structural = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )
        
        # MLP for edge-level (coplanarity) classification.
        # This accepts a concatenated pair of node embeddings, so its input dimension is 2 * out_channels.
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )
        
        # Sigmoid for converting logits to probabilities (for both tasks)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index):
        # Compute node embeddings via GAT
        h = self.gat(x, edge_index)
        
        # Node-level predictions: these might be used for textural/structural tasks.
        node_logits = self.mlp_textural_structural(h)
        node_out = self.sigmoid(node_logits)
        
        # Edge-level predictions for coplanarity:
        # First, extract the indices of the source and target nodes of each edge.
        src, dst = edge_index  # edge_index is of shape [2, num_edges]
        # Get the corresponding node embeddings.
        h_src = h[src]
        h_dst = h[dst]
        # Concatenate the source and destination embeddings.
        edge_features = torch.cat([h_src, h_dst], dim=1)  # Shape: [num_edges, 2 * out_channels]
        # Pass through an edge classifier MLP.
        edge_logits = self.edge_classifier(edge_features)
        edge_out = self.sigmoid(edge_logits)
        
        # Return both node-level and edge-level outputs.
        return node_out, edge_out
