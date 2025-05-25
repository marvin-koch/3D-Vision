import torch
import torch_geometric.nn as nn


class GATClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, act):
        super().__init__()
        self.gat = nn.GAT(in_channels=in_channels, 
                       hidden_channels=hidden_channels, 
                       out_channels=out_channels, 
                       num_layers=num_layers,
                       dropout=dropout,
                       act=act)

        self.mlp_textural_structural = torch.nn.Sequential(
            torch.nn.Linear(out_channels, out_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(out_channels, 1)
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x, edge_index):
        h = self.gat(x, edge_index)
        out = self.mlp_textural_structural(h)
        return self.sigmoid(out)