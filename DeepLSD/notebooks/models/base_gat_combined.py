import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class GATClassifierCombined(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, act):
        super().__init__()
        self.gat = pyg_nn.GAT(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            out_channels=out_channels, 
            num_layers=num_layers,
            dropout=dropout,
            act=act
        )
        
        # MLP textural/structural classification
        self.mlp_textural_structural = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )
        
        # MLP for coplanarity classification.
        self.edge_classifier = nn.Sequential(
            nn.Linear(2 * out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index):

        h = self.gat(x, edge_index)
        
        # Node-level predictions
        node_logits = self.mlp_textural_structural(h)
        node_out = self.sigmoid(node_logits)
        
        # Edge-level predictions for coplanarity
        src, dst = edge_index
        h_src = h[src]
        h_dst = h[dst]
        # Concatenate the source and destination embeddings.
        edge_features = torch.cat([h_src, h_dst], dim=1)
        # Pass through an edge classifier MLP.
        edge_logits = self.edge_classifier(edge_features)
        edge_out = self.sigmoid(edge_logits)
        
        # Return both node-level and edge-level outputs.
        return node_out, edge_out
