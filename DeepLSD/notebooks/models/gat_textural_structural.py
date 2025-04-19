import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class GATClassifierCombined(torch.nn.Module):
    def __init__(self, in_channels_DeepLSD, in_channels_GAT, hidden_channels, out_channels, roi_align_embedding_shape, num_layers, dropout, act):
        super().__init__()
        self.DeepLSD_gat = pyg_nn.GAT(
            in_channels=in_channels_DeepLSD, 
            hidden_channels=hidden_channels, 
            out_channels=out_channels, 
            num_layers=num_layers,
            dropout=dropout,
            act=act
        )
        self.ROI_align_GAT = pyg_nn.GAT(
            in_channels=in_channels_DeepLSD, 
            hidden_channels=hidden_channels, 
            out_channels=out_channels, 
            num_layers=num_layers,
            dropout=dropout,
            act=act
        )
        # use this to get the roi_embeddings
        conv_roi_embedding = nn.Sequential(
        # Input: (B, 3, 64, 64) where B is batch size
        # Layer 1: Convolution with 2 filters
        # Kernel size 3x3, stride 1, padding 1 preserves size initially
        nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1),
        # Output shape: (B, 2, 64, 64)
        nn.ReLU(),
        # Layer 2: Max Pooling to reduce size
        # Kernel size 2x2, stride 2 halves the height and width
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Output shape: (B, 2, 32, 32)

        # Layer 3: Convolution with 1 filter (to get the single channel output)
        nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
        # Output shape: (B, 1, 32, 32)
        nn.ReLU(),
        nn.Flatten(start_dim=1)
        # Finale Output shape: (B, 1 * 32 * 32) = (B,1024)
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
