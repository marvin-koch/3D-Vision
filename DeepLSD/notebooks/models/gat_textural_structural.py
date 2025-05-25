import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn

class GAT_TEXTURAL_STRUCTURAL(torch.nn.Module):
    def __init__(self, in_channels_DeepLSD, in_channels, hidden_channels, out_channels, roi_align_embedding_shape, num_layers, dropout, act, v2 = True, jk_layer = None, logger = None):
        super().__init__()
        self.gat = pyg_nn.GAT(
            in_channels=in_channels, 
            hidden_channels=hidden_channels, 
            out_channels=out_channels,
            v2 = True,
            num_layers=num_layers,
            dropout=dropout,
            act=act,
            jk = jk_layer
        )
        # output size of embedding:
        channels_conv_roi_embedding = roi_align_embedding_shape[0]//2 * roi_align_embedding_shape[1]//2
        self.conv_roi_embedding = nn.Sequential(
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
        # merge featues in one_layer with output equal to in_channels of GAT
        self.merge_features = nn.Sequential(
            nn.Linear(in_channels_DeepLSD + channels_conv_roi_embedding, in_channels),
            nn.GELU(),
        )
        
        # MLP textural/structural classification
        self.mlp_textural_structural = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, 1)
        )

        self.criterion = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
    def forward(self, batch):
        self.x = batch.x
        self.roi_features = batch.roi_features
        self.edge_index = batch.edge_index
        self.roi_conv_output = self.conv_roi_embedding(self.roi_features)
        self.combined_features = torch.cat([self.x, self.roi_conv_output], dim=1)
        self.h_in = self.merge_features(self.combined_features)
        self.h_out= self.gat(self.h_in, self.edge_index)
        
        # Node-level predictions
        self.node_logits = self.mlp_textural_structural(self.h_out)
        node_out = self.sigmoid(self.node_logits)
        
        # Return both node-level and edge-level outputs.
        return node_out
