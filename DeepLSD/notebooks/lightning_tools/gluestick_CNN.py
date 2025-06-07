# Combined file: gat_textural_structural_lightning.py
from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch_geometric.nn as pyg_nn
from torch.optim import Adam
from torch_geometric.data import Batch # Import Batch for type hinting if needed
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score, adjusted_rand_score, normalized_mutual_info_score, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from typing import Optional
import logging
import cv2

from sklearn.metrics import confusion_matrix
from typing import List

import os 
import matplotlib.pyplot as plt
import random
def plot_images(images, titles, cmaps=None):
    num = len(images)
    plt.figure(figsize=(15, 5))
    for i, img in enumerate(images):
        plt.subplot(1, num, i + 1)
        if cmaps is not None:
            cmap = cmaps if isinstance(cmaps, str) else cmaps[i]
            plt.imshow(img, cmap=cmap)
        else:
            plt.imshow(img)
        plt.title(titles[i])
        plt.axis('off')
    plt.show()
    
def cliques_to_flat_labels(
    cliques: List[List[int]],
    N: int,
    tie_break: str = "largest"
) -> torch.Tensor:
    """
    Given a list of maximal cliques (each clique is a list of line‐indices)
    and the total number of lines N, produce a 1D tensor of length N where
    each position i holds exactly one integer in [0..K-1] indicating which
    clique that line belongs to.  If a line is in multiple cliques, we break
    ties as specified.

    Args:
      cliques    : List of lists of ints, e.g. [[0,1,5], [1,2,3,5], [4,6]]
      N          : Total number of lines/nodes
      tie_break  : How to choose when a line is in multiple cliques:
                   - "first"   → pick the clique with smallest index
                   - "largest" → pick the clique of maximum size
                   - "random"  → choose uniformly at random among its cliques
    Returns:
      labels     : LongTensor of shape (N,), where labels[i] ∈ {0,…,K-1} or -1.
                   Lines not in any clique get label -1.
    """
    device = torch.device("cpu")
    K = len(cliques)
    # Precompute sizes for tie-breaking
    sizes = [len(c) for c in cliques]

    # For each line, track which clique‐indices it belongs to
    memberships: List[List[int]] = [[] for _ in range(N)]
    for clique_idx, clique in enumerate(cliques):
        for i in clique:
            memberships[i].append(clique_idx)

    # Build the flat labels
    labels = -1 * torch.ones(N, dtype=torch.long, device=device)
    rng = torch.Generator(device=device)
    for i in range(N):
        m = memberships[i]
        if not m:
            # line i is not in any clique → keep -1
            continue
        if len(m) == 1:
            labels[i] = m[0]
        else:
            # tie‐break
            if tie_break == "first":
                labels[i] = m[0]
            elif tie_break == "largest":
                # pick the clique with maximal size
                best = max(m, key=lambda idx: sizes[idx])
                labels[i] = best
            elif tie_break == "random":
                labels[i] = m[rng.randint(len(m))]
            else:
                raise ValueError(f"unknown tie_break='{tie_break}'")
    return labels


def plot_coplanar_lines(ax, lines, labels, image):
    """
    Visualize lines on an image with colors corresponding to their plane labels.
    Outliers (label -1) are drawn in grey. Designed to be used with a subplot axis.
    """
    unique_labels = sorted(set(labels))
    num_clusters = len(unique_labels)

    # Generate random colors for clusters (excluding -1 if present)
    random.seed(42)
    colors = [tuple(random.random() for _ in range(3)) for _ in range(num_clusters)]
    random.shuffle(colors)
    label_to_color = {label: colors[idx] for idx, label in enumerate(unique_labels)}

    ax.imshow(image)
    for idx, line in enumerate(lines):
        label = labels[idx]
        color = 'grey' if label == -1 or label == 0 else label_to_color.get(label, (0, 0, 0))
        ax.plot(
            [line[0, 0], line[1, 0]],
            [line[0, 1], line[1, 1]],
            color=color,
            linewidth=2
        )

    ax.set_title("Coplanar Lines")
    ax.axis('off')
    

def plot_lines_bool(ax, img, lines, is_correct, alpha=1):
    colors = ['red' if not c else 'blue' for c in is_correct]

    for i, l in enumerate(lines):
        arr = np.array(l)     # shape (2,2)
        line = plt.Line2D(
            (arr[0, 0], arr[1, 0]),
            (arr[0, 1], arr[1, 1]),
            linewidth=2,
            color=colors[i],
            alpha=alpha
        )
        ax.add_line(line)

    ax.imshow(img, cmap='gray')
    ax.set_axis_off()


def _load_image(filepath: str, color_conversion: Optional[int] = None) -> Optional[np.ndarray]:
    """Loads an image using OpenCV."""
    if not os.path.exists(filepath):
        logging.error(f"Image file not found: {filepath}")
        return None
    try:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED) # Load as is (handles color, grayscale, alpha)
        if img is None:
            logging.error(f"Failed to load image (cv2.imread returned None): {filepath}")
            return None
        if color_conversion is not None:
            img = cv2.cvtColor(img, color_conversion)
        return img
    except Exception as e:
        logging.error(f"Error loading image {filepath}: {e}")
        return None
# Keep plot_roc_curve as a utility function outside the LightningModule
# It can be called after testing if needed.
def plot_roc_curve(y_true, y_scores, title='ROC Curve', save_path=None):
    """Plots the ROC curve and saves it if a path is provided."""
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve saved to {save_path}")
    else:
        plt.show()
    plt.close() # Close the plot window
import torch
import torch.nn.functional as F

def edge_geometry(node_geo: torch.Tensor,
                  src: torch.Tensor,
                  dst: torch.Tensor,
                  eps: float = 1e-6) -> torch.Tensor:
    """
    node_geo : [N,8]   = [mid_x, mid_y, dir_x, dir_y, length,
                         θ, cos(2θ), sin(2θ)]
    src,dst  : [E]     edge indices

    Returns   : [E,11] same eleven features as before:
        0  d_mid
        1  perp_ij
        2  perp_ji
        3  cosθ
        4  |sinθ|
        5  signed_sinθ
        6  acute angle (rad)
        7  len_i
        8  len_j
        9  log(len_i/len_j)
       10  |cosθ|
    """
    # 1) unpack per-node geometry
    mid   = node_geo[:, 0:2]  # (N,2)
    dvec  = node_geo[:, 6:8]  # (N,2), unit
    leng  = node_geo[:, 2:3]  # (N,1)

    # 2) gather edge endpoints
    mi, mj = mid[src],  mid[dst]    # (E,2)
    di, dj = dvec[src], dvec[dst]   # (E,2)
    li, lj = leng[src], leng[dst]   # (E,1)

    # 3) midpoint‐to‐midpoint distance
    d_mid = (mi - mj).norm(dim=1, keepdim=True)  # (E,1)

    # 4) perpendicular offsets
    cross = lambda a, b: a[:,0:1]*b[:,1:2] - a[:,1:2]*b[:,0:1]
    delta  = mj - mi
    perp_ij = cross(delta, di).abs()  # (E,1)
    perp_ji = cross(delta, dj).abs()  # (E,1)

    # 5) orientation relationships
    cos_th = (di * dj).sum(1, keepdim=True).clamp(-1+eps, 1-eps)  # (E,1)
    sin_th = cross(di, dj)                                       # (E,1)
    abs_sin = sin_th.abs()
    ang_rad = torch.atan2(abs_sin, cos_th)                       # (E,1)
    par_score = cos_th.abs()                                     # (E,1)

    # 6) length ratio
    log_len_ratio = torch.log(li / (lj + eps))                   # (E,1)

    # 7) stack into final (E,11)
    edge_feat = torch.cat([
        d_mid,         # 0
        cos_th,        # 3
        sin_th,        # 5
        log_len_ratio, # 9
    ], dim=1)

    return edge_feat


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def attention(query, key, value):
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim ** .5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob

def MLP(channels, do_bn=True):
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n - 1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()
        assert d_model % h == 0
        self.dim = d_model // h
        self.h = h
        self.merge = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])
        # self.prob = []

    def forward(self, query, key, value):
        b = query.size(0)
        query, key, value = [l(x).view(b, self.dim, self.h, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        x, prob = attention(query, key, value)
        # self.prob.append(prob.mean(dim=1))
        return self.merge(x.contiguous().view(b, self.dim * self.h, -1))


class AttentionalPropagation(nn.Module):
    def __init__(self, num_dim, num_heads, skip_init=False, dropout=0.0):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, num_dim)
        self.mlp = MLP([num_dim * 2, num_dim * 2, num_dim], do_bn=True)
        self.dropout = nn.Dropout(p=dropout)

        nn.init.constant_(self.mlp[-1].bias, 0.0)
        if skip_init:
            self.register_parameter('scaling', nn.Parameter(torch.tensor(0.)))
        else:
            self.scaling = 1.

    def forward(self, x, source):
        message = self.attn(x, source, source)
        out = self.mlp(torch.cat([x, message], dim=1))
        out = self.dropout(out)                         # ← drop after MLP

        return  out * self.scaling

class SelfAttnLayer(nn.Module):
    def __init__(self, feature_dim, skip_init, dropout=0.0):
        super().__init__()
     
        self.update = AttentionalPropagation(feature_dim, 4, skip_init, dropout=dropout)

    def forward(self, desc):
        
        # self.update.attn.prob = []
        delta = self.update(desc, desc)
        return desc + delta


class LocalEdgeLayer(MessagePassing):
    def __init__(self, feature_dim: int):
        # use mean aggregation (you were doing scatter_mean)
        super().__init__(aggr='mean')
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.LongTensor):
        # x:       [N, D]
        # edge_index: [2, E]
        # propagate will handle the loop over edges and aggregation
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # x_i: features of destination nodes, shape [E, D]
        # x_j: features of source nodes,      shape [E, D]
        # return shape [E, D]
        return self.edge_mlp(torch.cat([x_i, x_j], dim=-1))
    
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class EdgeSamplerLayer(MessagePassing):
    def __init__(self,
                 node_dim: int,
                 edge_attr_dim: int,
                 hidden_dim: Optional[int] = None,
                 aggr: str = 'mean'):
        super().__init__(aggr=aggr)
        hidden_dim = hidden_dim or node_dim

        # now we take [x_i || x_j || edge_attr] → hidden → node_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*node_dim + edge_attr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.ln = nn.LayerNorm(node_dim)

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.LongTensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        # x:         [N, node_dim]
        # edge_index:[2, E]
        # edge_attr: [E, edge_attr_dim]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        # x_i:      [E, node_dim]  dest
        # x_j:      [E, node_dim]  src
        # edge_attr:[E, edge_attr_dim]
        m = torch.cat([x_i, x_j, edge_attr], dim=1)  # -> [E, 2*node_dim+edge_attr_dim]
        return self.edge_mlp(m)                       # -> [E, node_dim]



from torchvision.models import mobilenet_v2
from torchvision.models.feature_extraction import create_feature_extractor

class GlobalImageEncoder(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        base = mobilenet_v2(weights="DEFAULT").features

        self.encoder = base

        self.encoder.eval()

        # 2) freeze its weights
        for param in base.parameters():
            param.requires_grad = False

        self.project = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # → [B, 1280, 1, 1]
            nn.Flatten(),             # → [B, 1280]
            nn.Linear(1280, out_dim),
            nn.ReLU()
        )

    def forward(self, x):  # x: [B, 3, H, W]
        features = self.encoder(x)
        return self.project(features)  # → [B, out_dim]


import torch
import torch.nn as nn
import math

class Fourier2DPositionalEncoding(nn.Module):
    def __init__(self, num_frequencies: int = 8, include_input: bool = True, out_dim: int = 32):
        """
        Args:
            num_frequencies: Number of frequency bands (higher = more detail)
            include_input: Whether to include raw (x, y)
            out_dim: Final output dim (after projection)
        """
        super().__init__()
        self.include_input = include_input
        self.num_frequencies = num_frequencies

        # Compute frequency bands: [1, 2, 4, ..., 2^(n-1)]
        self.freq_bands = 2.0 ** torch.arange(num_frequencies).float() * math.pi  # [num_freqs]

        # Output projection
        input_dim = (2 if include_input else 0) + 2 * 2 * num_frequencies  # (x,y) + sin/cos * 2 coords * N
        self.proj = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Tensor of shape [N, 2] with (x, y) positions (normalized to [0, 1])
        Returns:
            Tensor of shape [N, out_dim]
        """
        # coords: [N, 2]
        N, _ = coords.shape
        coords = coords.unsqueeze(-1)                      # → [N, 2, 1]
        freqs = self.freq_bands.to(coords.device)[None]    # → [1, num_freqs]
        angles = coords * freqs                            # → [N, 2, num_freqs]

        sin_enc = torch.sin(angles)                        # [N, 2, num_freqs]
        cos_enc = torch.cos(angles)                        # [N, 2, num_freqs]

        enc = [sin_enc, cos_enc]                           # List of [N, 2, num_freqs]
        enc = torch.cat(enc, dim=1)                        # [N, 4, num_freqs]
        enc = enc.flatten(1)                               # [N, 4 * num_freqs]

        if self.include_input:
            enc = torch.cat([coords.squeeze(-1), enc], dim=1)  # [N, 2 + 4*num_freqs]

        return self.proj(enc)                              # [N, out_dim]


from torchvision.models.resnet import BasicBlock
from torch_geometric.nn import GATConv
from torch_geometric.nn import GATv2Conv
import torchvision.transforms.functional as TF
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, f1_score


    
class AttentionCNN(pl.LightningModule):
    def __init__(self,
        # Model HParams
        in_channels_DeepLSD: int,
        in_channels: int,
        hidden_channels: int,
        hidden_channels_cnn:int,
        out_channels: int,
        geom_channels: int,
        roi_align_embedding_shape: tuple,
        num_layers: int,
        dropout: float = 0.5,
        act: str = 'relu',
        v2: bool = True,
        jk_layer: str = None,
        edge_sample_size = (32,16),
        edge_downsample_dim = 20,
        # Training HParams
        learning_rate: float = 1e-3,
        node_loss_w: float = 1.0,          # Weight for node loss
        edge_loss_w: float = 1.0,          # Weight for edge loss (new)
        threshold_structural: float = 0.5,  # Threshold for accuracy/recall calc
        mlp_dropout: float = 0.0,          # drop out for merge features
        skip_init=False,
        ):
        super().__init__()
        self.save_hyperparameters()
        


  
        # self.bands_cnn = nn.Sequential(
        #     nn.LazyConv2d(self.hparams.hidden_channels_cnn, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),

        #     nn.Conv2d(self.hparams.hidden_channels_cnn, self.hparams.hidden_channels_cnn * 2, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),

        #     nn.AdaptiveAvgPool2d(1),   # Global average pooling → [B, 2h, 1, 1]
        #     nn.Flatten(1),             # → [B, 2h]
        #     nn.Linear(self.hparams.hidden_channels_cnn * 2, self.hparams.hidden_channels_cnn)
        # )

        self.image_encoder = GlobalImageEncoder(out_dim=hidden_channels_cnn)
        self.rgb_image_encoder = GlobalImageEncoder(out_dim=hidden_channels_cnn)


        # self.image_encoder = nn.Sequential(
        #     # conv1: 7×7, stride 2, padding 3 → 64 channels
        #     nn.LazyConv2d(self.hparams.hidden_channels_cnn, kernel_size=7, stride=2, padding=3, bias=False),
        #     nn.InstanceNorm2d(self.hparams.hidden_channels_cnn, affine=True),
        #     nn.ReLU(inplace=True),

        #     # max-pool 3×3, stride 2
        #     nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

        #     # two BasicBlock’s (64→64)

        #     BasicBlock(self.hparams.hidden_channels_cnn, self.hparams.hidden_channels_cnn),
        #     BasicBlock(self.hparams.hidden_channels_cnn, self.hparams.hidden_channels_cnn),

        #     # now downsample to fixed vector
        #     nn.AdaptiveAvgPool2d(1),    # → [B, 64, 1, 1]
        #     nn.Flatten(1),              # → [B, 64]
        #     nn.Linear(self.hparams.hidden_channels_cnn, self.hparams.hidden_channels_cnn)
        # )



        self.bands_cnn = nn.Sequential(
            # conv1: 7×7, stride 2, padding 3 → 64 channels
            nn.LazyConv2d(self.hparams.hidden_channels_cnn, kernel_size=7, stride=2, padding=3, bias=False),
            # nn.InstanceNorm2d(self.hparams.hidden_channels_cnn, affine=True),
            nn.ReLU(inplace=True),

            # max-pool 3×3, stride 2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # two BasicBlock’s (64→64)

            BasicBlock(self.hparams.hidden_channels_cnn, self.hparams.hidden_channels_cnn),
            BasicBlock(self.hparams.hidden_channels_cnn, self.hparams.hidden_channels_cnn),

            # now downsample to fixed vector
            nn.AdaptiveAvgPool2d(1),    # → [B, 64, 1, 1]
            nn.Flatten(1),              # → [B, 64]
            nn.Linear(self.hparams.hidden_channels_cnn, self.hparams.hidden_channels_cnn)
        )


        self.length_mlp = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        self.angle_mlp = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        self.pos_encoder = Fourier2DPositionalEncoding(num_frequencies=8, out_dim=32)



        # self.geo_mlp = nn.Sequential(
        #     nn.Linear(self.hparams_geom_channels, self.hparams.hidden_channels_cnn),
        #     nn.LayerNorm(self.hparams.hidden_channels_cnn),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.hparams.hidden_channels_cnn, self.hparams.hidden_channels_cnn),
        #     nn.ReLU(inplace=True)
        # )
      
        
        self.edge_loss_w = edge_loss_w
  
        self.node_fuse = nn.Sequential(
            nn.Linear(3*self.hparams.hidden_channels_cnn  + 2 * 16 + 32,
                      self.hparams.out_channels),
            nn.LayerNorm(self.hparams.out_channels),

            nn.ReLU(),
            nn.Linear(self.hparams.out_channels,
                      self.hparams.out_channels),
        )


        self.edge_geo_channels = 4


        self.edge_geo_mlp = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )
        # GNN layers
        layers = []
        for i in range(self.hparams.num_layers):
            if i % 2 == 0:
                layers.append(SelfAttnLayer(self.hparams.out_channels, self.hparams.skip_init))
            else:
                layers.append(EdgeSamplerLayer(node_dim=self.hparams.out_channels,
                edge_attr_dim=16,
                hidden_dim=self.hparams.out_channels))
                
                
                # layers.append(LocalEdgeLayer(self.hparams.out_channels))

        self.layers = nn.ModuleList(layers)
        # Node prediction head
        self.mlp_textural_structural = nn.Sequential(
            nn.Linear(self.hparams.out_channels + 3* self.hparams.hidden_channels_cnn + 16, self.hparams.out_channels),
            # nn.LayerNorm(self.hparams.out_channels),

            nn.ReLU(),
            nn.Linear(self.hparams.out_channels, 1)
        )

        self.edge_predictor = nn.Sequential(
            nn.Linear(4*self.hparams.out_channels, self.hparams.out_channels),
            # nn.LayerNorm(self.hparams.out_channels),

            nn.ReLU(),
            nn.Linear(self.hparams.out_channels, 1)
        )

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Containers for metrics
        self.validation_step_outputs = []
        self.test_step_outputs = []
        

                    
    def forward(self, batch):
        
        edge_index, batch_idx = batch.edge_index, batch.batch
        geo = batch.geo


        img = batch.img
        rgb_patches = batch.rgb_patches
        angle_field_patches = batch.angle_field_patches
        distance_patches = batch.distance_patches
   

        # cnn_features = []

        # for idx in range(len(rgb_patches)):
        #     rgb_patches_idx= (rgb_patches[idx]).permute(2,0,1)
        #     angle_field_patches_idx = (angle_field_patches[idx]).permute(2,0,1)
        #     distance_patches_idx =(distance_patches[idx]).permute(2,0,1)


        #     H2, W2 = angle_field_patches.shape[1:3]
        #     img_rs = F.interpolate(
        #         rgb_patches_idx.unsqueeze(0),           # add batch
        #         size=(H2, W2),
        #         mode="bilinear",
        #         align_corners=False
        #     ).squeeze(0)                   # remove batch  → [C1, H2, W2]


        #     full_patch = torch.cat([img_rs, angle_field_patches_idx, distance_patches_idx], dim=0)


        #     # 3) CNN head (expects a batch dim)
        #     out = self.bands_cnn(full_patch.unsqueeze(0))            # [1, output_dim]
        #     cnn_features.append(out.squeeze(0))            # [output_dim]
        # cnn_features = torch.stack(cnn_features, dim=0)          # [N, output_dim]



        img = img.squeeze(0)  # remove batch dim → [H, W, C]
        img = img.permute(2, 0, 1).float()    # float32 C×H×W
        # 5) scale [0,255]→[0,1]
        img = img.div(255.)
        # 6) normalize to ImageNet mean/std
        img = TF.normalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        )          

        img = img.unsqueeze(0)  # add batch dim → [1, C, H, W]
        
        img_feat = self.image_encoder(img)  # [B, hidden_channels_cnn]

        # 1) move channels to dim=1
        rgb   = rgb_patches.permute(0, 3, 1, 2)           # [N, C1, H1, W1]

        rgb = rgb.div(255.)  # scale [0,255]→[0,1]

        angle = angle_field_patches.permute(0, 3, 1, 2)   # [N, C2, H2, W2]
                
        dist  = distance_patches.permute(0, 3, 1, 2)      # [N, C3, H2, W2]

        # 2) resize rgb to (H2, W2)
        #    note: H2, W2 = angle.shape[2], angle.shape[3]
        rgb_rs = F.interpolate(
            rgb,
            size=angle.shape[2:],       # (H2, W2)
            mode="bilinear",
            align_corners=False
        )                              # [N, C1, H2, W2]


        rgb = rgb.div(255.)
        # 6) normalize to ImageNet mean/std
        rgb = TF.normalize(
            rgb,
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        )          

        

        rgb_feat = self.rgb_image_encoder(rgb)  # [B, hidden_channels_cnn]



        # 3) concat along channel axis
        full_patches = torch.cat([angle, dist], dim=1)  
        #                         ——————————————————————^
        #                                 C1 + C2 + C3
        # ⇒ full_patches: [N, C1+C2+C3, H2, W2]


        # 3) CNN head (expects a batch dim)
        cnn_features = self.bands_cnn(full_patches)            # [1, output_dim]
  
        pos_encoding = self.pos_encoder(geo[:, :2])  # [N, out_dim]

        length_feat = self.length_mlp(geo[:, 2:3])  # [N, 16]
        angle_feat = self.angle_mlp(geo[:, 3:6])    # [N, 16]


        n_lines = rgb.size(0)          # 476
        # Method A: expand (no new memory; read-only)
        img_feat_exp = img_feat.expand(n_lines, -1)     # → [476, 64]



        concat_feat = torch.cat([rgb_feat, cnn_features, pos_encoding, length_feat, angle_feat, img_feat_exp], dim=1)

        concat_feat = self.node_fuse(concat_feat)
        
        

        src, dst = batch.edge_index

        edge_geo = edge_geometry(geo, src, dst) 

        edge_geo_feat = self.edge_geo_mlp(edge_geo)  # [E, 16]


        # x = concat_feat
    

        # for i, layer in enumerate(self.layers):
        #     if i % 2 == 0:
        #         delta = layer(x, batch.full_edge_index)
        #         x = F.relu(x + delta)
        #     else:

        #         delta = layer(x, batch.edge_index, edge_attr=edge_geo_feat)
        #         x = F.relu(x + delta)
           

        # # x now holds your final node embeddings [N, D]

        
        # features = x

          
        # Prepare for global attention
        roi_dense, mask = to_dense_batch(concat_feat, batch_idx)
        desc = roi_dense.transpose(1, 2)
        
        
        #local_edge_geo = edge_geo[batch.flat_idx_local] 
        # Alternate layers
        for layer in self.layers:
            if isinstance(layer, SelfAttnLayer):
                desc = layer(desc)
            else:
                flat = desc.transpose(1,2)[mask]
                # flat_norm = layer.ln(flat)   

                delta = layer(flat, edge_index, edge_geo_feat)

                delta_dense, _ = to_dense_batch(delta, batch_idx)
                desc = desc + delta_dense.transpose(1,2)
                

        features = desc.transpose(1,2)[mask]


        struct_feats = torch.cat([rgb_feat, features, cnn_features, img_feat_exp, length_feat], dim=1)  # [N, D+2*D]

        # Node logits
        node_logits = self.mlp_textural_structural(struct_feats)

        # Edge logits



        src, dst = batch.full_edge_index
        h_src, h_dst = features[src], features[dst]
        concat_src, concat_dst = concat_feat[src], concat_feat[dst] # CNN features
        

       
        edge_in = torch.cat([
            h_src, h_dst,   
           concat_src,  concat_dst,  
        ], dim=1)  
                
        edge_in = self.edge_predictor(edge_in)  # overwrite so save memory these are now logits

        return node_logits, edge_in


    def training_step(self, batch, batch_idx):
        node_logits, edge_logits = self(batch)
        node_labels = batch.y.view(-1,1).float()
        full_edge_labels = batch.full_edge_labels
        # sample nodes
        pos_n = (node_labels==1).nonzero(as_tuple=True)[0]
        neg_n = (node_labels==0).nonzero(as_tuple=True)[0]
        
        if pos_n.numel()>0:
            perm = torch.randperm(neg_n.size(0))
            sampled_neg_n = neg_n[perm[:pos_n.size(0)]]
            keep_n = torch.cat([pos_n, sampled_neg_n])
        else:
            k=min(32,neg_n.size(0)); perm=torch.randperm(neg_n.size(0))
            keep_n=neg_n[perm[:k]]
        sampled_node_logits = node_logits[keep_n]
        sampled_node_labels = node_labels[keep_n]
        node_loss = self.criterion(sampled_node_logits, sampled_node_labels)
        # sample edges
        edge_labels_flat = full_edge_labels.view(-1,1)
        pos_e = (edge_labels_flat==1).nonzero(as_tuple=True)[0]
        neg_e = (edge_labels_flat==0).nonzero(as_tuple=True)[0]
        if pos_e.numel()>0:
            perm_e = torch.randperm(neg_e.size(0))
            sampled_neg_e = neg_e[perm_e[:pos_e.size(0)]]
            keep_e = torch.cat([pos_e, sampled_neg_e])
        else:
            k_e=min(32,neg_e.size(0)); perm_e=torch.randperm(neg_e.size(0))
            keep_e=neg_e[perm_e[:k_e]]

        
       

        sampled_edge_logits = edge_logits[keep_e]
        sampled_edge_labels = edge_labels_flat[keep_e]
        edge_loss = self.criterion(sampled_edge_logits, sampled_edge_labels)
        # combined
        
      
     
        # loss = self.hparams.node_loss_w*node_loss + self.hparams.edge_loss_w*edge_loss
        loss = (node_loss + edge_loss) * 0.5
        # metrics
        with torch.no_grad():
            n_probs=torch.sigmoid(sampled_node_logits); n_preds=(n_probs>=self.hparams.threshold_structural).int().detach().cpu().numpy().ravel()
            node_acc=accuracy_score(sampled_node_labels.int().detach().cpu().numpy().ravel(),n_preds)
            e_probs=torch.sigmoid(sampled_edge_logits); e_preds=(e_probs>=self.hparams.threshold_structural).int().detach().cpu().numpy().ravel()
            edge_acc=accuracy_score(sampled_edge_labels.int().detach().cpu().numpy().ravel(), e_preds)
            
            
        self.log('train_loss',loss,on_step=True,on_epoch=False, prog_bar=True, logger=True)
        self.log('train_node_acc',node_acc,on_step=True,on_epoch=False, prog_bar=True, logger=True)
        self.log('train_edge_acc',edge_acc,on_step=True,on_epoch=False, prog_bar=True, logger=True)
        return loss

    # def training_step(self, batch, batch_idx):
    #     # Forward pass
    #     node_logits, edge_logits = self(batch)

    #     # -- NODE LOSS ---------------------------------------------------------------
    #     node_labels = batch.y.view(-1, 1).float()  # [N, 1]
    #     pos_n = (node_labels == 1).nonzero(as_tuple=True)[0]
    #     neg_n = (node_labels == 0).nonzero(as_tuple=True)[0]

    #     if pos_n.numel() > 0:
    #         perm = torch.randperm(len(neg_n), device=self.device)
    #         sampled_neg_n = neg_n[perm[:len(pos_n)]]
    #         keep_n = torch.cat([pos_n, sampled_neg_n])
    #     else:
    #         k = min(32, len(neg_n))
    #         keep_n = neg_n[torch.randperm(len(neg_n), device=self.device)[:k]]

    #     sampled_node_logits = node_logits[keep_n]
    #     sampled_node_labels = node_labels[keep_n]
    #     node_loss = self.criterion(sampled_node_logits, sampled_node_labels)

    #     # -- EDGE LOSS with UNCERTAINTY SAMPLING -------------------------------------
    #     edge_labels = batch.full_edge_labels.view(-1).float()     # [E]
    #     edge_probs = torch.sigmoid(edge_logits.view(-1))          # [E]

    #     pos_e = (edge_labels == 1).nonzero(as_tuple=True)[0]
    #     neg_e = (edge_labels == 0).nonzero(as_tuple=True)[0]

    #     # --- Hard negatives: model is confident but wrong
    #     hard_neg = neg_e[edge_probs[neg_e] > 0.5]

    #     # --- Ambiguous negatives: model is unsure
    #     ambiguous_neg = neg_e[(edge_probs[neg_e] > 0.4) & (edge_probs[neg_e] < 0.6)]
    #     mask = ~torch.tensor([i.item() in set(hard_neg.tolist()) for i in ambiguous_neg], device=ambiguous_neg.device)
    #     ambiguous_only = ambiguous_neg[mask]

    #     # --- Random fallback
    #     rand_neg = neg_e[torch.randperm(len(neg_e), device=self.device)[:len(pos_e)]]

    #     # Combine and trim to match pos_e
    #     combined_neg = torch.cat([
    #         hard_neg,
    #         ambiguous_only,
    #         rand_neg
    #     ])[:len(pos_e)]

    #     keep_e = torch.cat([pos_e, combined_neg])
    #     sampled_edge_logits = edge_logits.view(-1)[keep_e]
    #     sampled_edge_labels = edge_labels[keep_e]
    #     edge_loss = self.criterion(sampled_edge_logits, sampled_edge_labels)

    #     # -- TOTAL LOSS ----------------------------------------------------------------
    #     loss = (node_loss + edge_loss) * 0.5

    #     # -- METRICS -------------------------------------------------------------------
    #     with torch.no_grad():
    #         # Node metrics
    #         n_probs = torch.sigmoid(sampled_node_logits).detach().cpu().numpy().ravel()
    #         n_labels = sampled_node_labels.detach().cpu().numpy().ravel().astype(int)
    #         node_acc = accuracy_score(n_labels, (n_probs >= self.hparams.threshold_structural).astype(int))

    #         # Edge metrics
    #         e_probs = torch.sigmoid(sampled_edge_logits).detach().cpu().numpy().ravel()
    #         e_labels = sampled_edge_labels.detach().cpu().numpy().ravel().astype(int)
    #         edge_acc = accuracy_score(e_labels, (e_probs >= self.hparams.threshold_structural).astype(int))

    #     # -- LOG -----------------------------------------------------------------------
    #     self.log('train_loss', loss, prog_bar=True)
    #     self.log('train_node_acc', node_acc, prog_bar=True)
    #     self.log('train_edge_acc', edge_acc, prog_bar=True)
    #     self.log('train_hard_neg_count', len(hard_neg), prog_bar=False)
    #     self.log('train_ambig_neg_count', len(ambiguous_neg), prog_bar=False)

    #     return loss


    def validation_step(self, batch: Batch, batch_idx: int):
        # forward
        node_logits, edge_logits = self(batch)
        # labels
        node_labels      = batch.y.float()               # [N,1]
        full_edge_labels = batch.full_edge_labels.float()  # [N*N,1]

        # losses
        node_loss = self.criterion(node_logits, node_labels)
        edge_loss = self.criterion(edge_logits, full_edge_labels)
        total_loss = (node_loss + edge_loss) * 0.5



        # log losses
        self.log('val_node_loss', node_loss,  on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('val_edge_loss', edge_loss,  on_step=True,on_epoch=False, prog_bar=True, logger=True)
        self.log('val_loss',      total_loss, on_step=True,on_epoch=False, prog_bar=True,  logger=True)

        # store for epoch-end
        self.validation_step_outputs.append({
            'node_preds': node_logits.sigmoid().detach(),
            'node_labels': node_labels.detach(),
            'edge_preds': edge_logits.sigmoid().detach(),
            'edge_labels': full_edge_labels.detach(),
            'loss': total_loss
        })
        return {'loss': total_loss,  
            'node_preds': node_logits.sigmoid(),
            'node_labels': node_labels,
            'edge_preds': edge_logits.sigmoid(),
            'edge_labels': full_edge_labels}


  
    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        # aggregate outputs
        losses = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        all_node_preds = torch.cat([x['node_preds'] for x in self.validation_step_outputs]).cpu().numpy()
        all_node_labels = torch.cat([x['node_labels'] for x in self.validation_step_outputs]).cpu().numpy()
        all_edge_preds = torch.cat([x['edge_preds'] for x in self.validation_step_outputs]).cpu().numpy()
        all_edge_labels = torch.cat([x['edge_labels'] for x in self.validation_step_outputs]).cpu().numpy()

        # binary decisions
        thresh = self.hparams.threshold_structural
        node_binary = (all_node_preds >= thresh).astype(int)
        edge_binary = (all_edge_preds >= thresh).astype(int)

        # confusion values
        tn_n, fp_n, fn_n, tp_n = confusion_matrix(all_node_labels, node_binary, labels=[0,1]).ravel()
        tn_e, fp_e, fn_e, tp_e = confusion_matrix(all_edge_labels, edge_binary, labels=[0,1]).ravel()

        # compute metrics
        node_acc = (tp_n + tn_n) / (tp_n + tn_n + fp_n + fn_n)
        node_recall = tp_n / (tp_n + fn_n) if (tp_n + fn_n)>0 else 0.0
        node_precision = tp_n / (tp_n + fp_n) if (tp_n + fp_n)>0 else 0.0
        node_specificity = tn_n / (tn_n + fp_n) if (tn_n + fp_n)>0 else 0.0
        node_auc = auc(*roc_curve(all_node_labels, all_node_preds)[:2]) if len(np.unique(all_node_labels))>1 else 0.0

        edge_acc = (tp_e + tn_e) / (tp_e + tn_e + fp_e + fn_e)
        edge_recall = tp_e / (tp_e + fn_e) if (tp_e + fn_e)>0 else 0.0
        edge_precision = tp_e / (tp_e + fp_e) if (tp_e + fp_e)>0 else 0.0
        edge_specificity = tn_e / (tn_e + fp_e) if (tn_e + fp_e)>0 else 0.0
        edge_auc = auc(*roc_curve(all_edge_labels, all_edge_preds)[:2]) if len(np.unique(all_edge_labels))>1 else 0.0

        combined_auc = 0.5 * (node_auc + edge_auc)

           # ---- NEW: PR‐AUC / average precision ----
            # Option A: direct from sklearn
        node_pr_auc = average_precision_score(all_node_labels, all_node_preds) \
                    if len(np.unique(all_node_labels))>1 else 0.0
        edge_pr_auc = average_precision_score(all_edge_labels, all_edge_preds) \
                    if len(np.unique(all_edge_labels))>1 else 0.0
        
        combined_pr_auc = 0.5 * (node_pr_auc + edge_pr_auc)

        # log metrics
        self.log_dict({
            'val_loss_epoch': losses,
            'val_node_acc_epoch': node_acc,
            'val_node_recall_epoch': node_recall,
            'val_node_precision_epoch': node_precision,
            'val_node_specificity_epoch': node_specificity,
            'val_node_auc_epoch': node_auc,
            'val_edge_acc_epoch': edge_acc,
            'val_edge_recall_epoch': edge_recall,
            'val_edge_precision_epoch': edge_precision,
            'val_edge_specificity_epoch': edge_specificity,
            'val_edge_auc_epoch': edge_auc,
            'val_combined_auc_epoch': combined_auc,
            'val_edge_pr_auc_epoch': edge_pr_auc,
            'val_node_pr_auc_epoch': node_pr_auc,
            'val_combined_pr_auc_epoch': combined_pr_auc,

            'tn_n': tn_n,
            'tp_n': tp_n,
            'fp_n': fp_n,
            'fn_n': fn_n,
            'tn_e': tn_e,
            'tp_e': tp_e,
            'fp_e': fp_e,
            'fn_e': fn_e,
        }, on_epoch=True, prog_bar=False)

      

        self.validation_step_outputs.clear()

    


    def test_step(self, batch: Batch, batch_idx: int):
        # forward
        node_logits, edge_logits = self(batch)
        # labels
        node_labels      = batch.y.float()
        full_edge_labels = batch.full_edge_labels.float()

        # losses
        node_loss = self.criterion(node_logits, node_labels)
        edge_loss = self.criterion(edge_logits, full_edge_labels)
       # total_loss = self.hparams.node_loss_w * node_loss + self.hparams.edge_loss_w * edge_loss
        total_loss = (node_loss + edge_loss) * 0.5


        # log losses
        self.log('test_node_loss', node_loss,  on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('test_edge_loss', edge_loss,  on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log('test_loss',      total_loss, on_step=True, on_epoch=False, prog_bar=True,  logger=True)

        # store for epoch-end
        self.test_step_outputs.append({
            'node_preds': node_logits.sigmoid().detach(),
            'node_labels': node_labels.detach(),
            'edge_preds': edge_logits.sigmoid().detach(),
            'edge_labels': full_edge_labels.detach(),
            'loss': total_loss
        })
        return {
            'node_preds': node_logits.sigmoid(),
            'node_labels': node_labels,
            'edge_preds': edge_logits.sigmoid(),
            'edge_labels': full_edge_labels,
            'loss': total_loss
        }


    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            return

        losses = torch.stack([x['loss'] for x in self.test_step_outputs]).mean()
        all_node_preds = torch.cat([x['node_preds'] for x in self.test_step_outputs]).cpu().numpy()
        all_node_labels = torch.cat([x['node_labels'] for x in self.test_step_outputs]).cpu().numpy()
        all_edge_preds = torch.cat([x['edge_preds'] for x in self.test_step_outputs]).cpu().numpy()
        all_edge_labels = torch.cat([x['edge_labels'] for x in self.test_step_outputs]).cpu().numpy()

        thresh = self.hparams.threshold_structural
        node_binary = (all_node_preds >= thresh).astype(int)
        edge_binary = (all_edge_preds >= thresh).astype(int)

        tn_n, fp_n, fn_n, tp_n = confusion_matrix(all_node_labels, node_binary, labels=[0,1]).ravel()
        tn_e, fp_e, fn_e, tp_e = confusion_matrix(all_edge_labels, edge_binary, labels=[0,1]).ravel()

        node_pr_auc = average_precision_score(all_node_labels, all_node_preds) \
                    if len(np.unique(all_node_labels))>1 else 0.0
        edge_pr_auc = average_precision_score(all_edge_labels, all_edge_preds) \
                    if len(np.unique(all_edge_labels))>1 else 0.0
        
        combined_pr_auc = 0.5 * (node_pr_auc + edge_pr_auc)

           
            # ---- NEW: F1‐score (threshold‐dependent) ----
        node_f1 = f1_score(all_node_labels, node_binary) if len(np.unique(all_node_labels))>1 else 0.0
        edge_f1 = f1_score(all_edge_labels, edge_binary) if len(np.unique(all_edge_labels))>1 else 0.0



        self.log_dict({
            'test_loss_epoch': losses,
            'test_node_acc_epoch': (tp_n + tn_n) / (tp_n + tn_n + fp_n + fn_n),
            'test_node_recall_epoch': tp_n / (tp_n + fn_n) if (tp_n + fn_n)>0 else 0.0,
            'test_node_precision_epoch': tp_n / (tp_n + fp_n) if (tp_n + fp_n)>0 else 0.0,
            'test_node_specificity_epoch': tn_n / (tn_n + fp_n) if (tn_n + fp_n)>0 else 0.0,
            'test_node_auc_epoch': auc(*roc_curve(all_node_labels, all_node_preds)[:2]) if len(np.unique(all_node_labels))>1 else 0.0,
            'test_edge_acc_epoch': (tp_e + tn_e) / (tp_e + tn_e + fp_e + fn_e),
            'test_edge_recall_epoch': tp_e / (tp_e + fn_e) if (tp_e + fn_e)>0 else 0.0,
            'test_edge_precision_epoch': tp_e / (tp_e + fp_e) if (tp_e + fp_e)>0 else 0.0,
            'test_edge_specificity_epoch': tn_e / (tn_e + fp_e) if (tn_e + fp_e)>0 else 0.0,
            'test_edge_auc_epoch': auc(*roc_curve(all_edge_labels, all_edge_preds)[:2]) if len(np.unique(all_edge_labels))>1 else 0.0,
            'test_combined_pr_auc_epoch': combined_pr_auc,
            'test_edge_pr_auc_epoch': edge_pr_auc,
            'test_node_pr_auc_epoch': node_pr_auc,
            'test_node_f1_epoch':     node_f1,
            'test_edge_f1_epoch':     edge_f1,


            'tn_n': tn_n,
            'tp_n': tp_n,
            'fp_n': fp_n,
            'fn_n': fn_n,
            'tn_e': tn_e,
            'tp_e': tp_e,
            'fp_e': fp_e,
            'fn_e': fn_e,
        }, on_epoch=True, prog_bar=False)


        self.test_step_outputs.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr            = self.hparams.learning_rate,   # same LR usually works
            betas         = (0.9, 0.999),
            eps           = 1e-8,
            weight_decay  = 1e-4,        # **decoupled** L2, not mixed into the grads
        )

        # warm-up → cosine ↓ to zero (good default for vision & graphs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=0.0
        )

        return [optimizer], [scheduler]







def multi_clique_discriminative_loss(
    emb: torch.Tensor,
    cliques: List[List[int]],
    delta_var: float = 0.5,
    delta_dist: float = 1.5,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 0.001,
) -> torch.Tensor:
    device = emb.device
    K = len(cliques)
    if K == 0:
        # If there are no valid cliques, return three zero‐losses of the correct dtype/device.
        zero = torch.tensor(0.0, device=device)
        return zero, zero, zero


    # 1) Compute centroids for each clique
    centroids = []
    for clique in cliques:
        idx = torch.tensor(clique, device=device, dtype=torch.long)
        if idx.numel() == 0:
            centroids.append(torch.zeros(emb.size(1), device=device))
        else:
            centroids.append(emb[idx].mean(dim=0))
    centroids = torch.stack(centroids, dim=0)  # (K, D)

    # 2) Intra‐cluster (pull) term
    # var_loss = torch.tensor(0.0, device=device)
    # for i, clique in enumerate(cliques):
    #     idx = torch.tensor(clique, device=device, dtype=torch.long)
    #     if idx.numel() == 0:
    #         continue
    #     diff = emb[idx] - centroids[i]       # (|clique|, D)
    #     dist_i = diff.norm(p=2, dim=1)       # (|clique|,)
    #     var_loss = var_loss + torch.mean(torch.clamp(dist_i - delta_var, min=0.0) ** 2)
    # var_loss = var_loss / K
    
    total_points = sum(len(c) for c in cliques)
    var_sum = 0.0
    for i, clique in enumerate(cliques):
        idx = torch.tensor(clique, device=device)
        diff = emb[idx] - centroids[i]
        dist_i = diff.norm(dim=1)
        var_sum += torch.sum(torch.clamp(dist_i - delta_var, min=0.0) ** 2)
    var_loss = var_sum / total_points

    # 3) Inter‐cluster (push) term
    if K > 1:
        # pairwise centroid distances
        c1 = centroids.unsqueeze(0)          # (1, K, D)
        c2 = centroids.unsqueeze(1)          # (K, 1, D)
        dmat = (c1 - c2).norm(p=2, dim=2)     # (K, K)
        margin = 2.0 * delta_dist
        # clamp → squared → average (ignore diagonal)
        md = torch.clamp(margin - dmat, min=0.0)  # (K, K)
        eye = torch.eye(K, device=device)
        md = md * (1.0 - eye)                      # zero out diagonal
        dist_loss = torch.mean(md ** 2)
    else:
        dist_loss = torch.tensor(0.0, device=device)

    # 4) Regularizer
    reg_loss = torch.mean(centroids.norm(p=2, dim=1))

    # 5) Log exactly the clamped‐and‐squared average, not the raw dmat mean
    # print(f"    [CLUSTER] var={var_loss.item():.4f}, dist_loss={dist_loss.item():.4f}, reg={reg_loss.item():.4f}")

    return alpha * var_loss, beta * dist_loss, gamma * reg_loss


def hdbscan_cluster(emb: torch.Tensor, min_cluster_size:int = 6, min_samples: int = 3) -> torch.Tensor:
    if emb.numel() == 0:
        return emb.new_full((0,), -1, dtype=torch.long)
    model = hdbscan.HDBSCAN(
        metric='euclidean',
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        approx_min_span_tree=False,
        # cluster_selection_epsilon=0.05,  # smaller epsilon → fewer merges

    )
    labels = model.fit_predict(emb)


    return torch.as_tensor(labels, device=emb.device, dtype=torch.long)


import hdbscan

import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Any

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    v_measure_score,
    pair_confusion_matrix,
)
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def _pairwise_prf1(labels_true: torch.Tensor | list[int] | "np.ndarray", labels_pred):
    """Return pairwise precision, recall, F1 (see pair_confusion_matrix)."""
    cm = pair_confusion_matrix(labels_true, labels_pred)  # [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def _hungarian_accuracy(labels_true, labels_pred):
    """Cluster accuracy after optimal assignment (a.k.a. purity w/ Hungarian)."""
    cm = contingency_matrix(labels_true, labels_pred)
    if cm.size == 0:
        return 0.0
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximise
    return cm[row_ind, col_ind].sum() / cm.sum()



from torch_geometric.nn import GATv2Conv      # NEW
class EdgeGATv2Layer(nn.Module):
    def __init__(self, node_dim: int, edge_attr_dim: int, heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.gat = GATv2Conv(
            in_channels=node_dim,
            out_channels=node_dim // heads,
            heads=heads,
            concat=True,             # final dim = node_dim
            dropout=dropout,
            edge_dim=edge_attr_dim,   # <<< uses edge_geo_feat
            add_self_loops=False
        )

    def forward(self, x, edge_index, edge_attr):
        out = self.gat(x, edge_index, edge_attr=edge_attr)
        return out - x               # ← delta, to match your residual call-site



# ──────────────────────────────────────────────────────────────────────────────
# Local – Global – Cross attention blocks
# (now both Local and Global accept an optional knn_cache)
# ──────────────────────────────────────────────────────────────────────────────
from torch_geometric.nn import EdgeConv, TransformerConv, knn_graph
import torch, torch.nn as nn
# --------------------------------------------------------------------------- #
class LocalEdgeBlock(nn.Module):
    """k-NN EdgeConv (optionally dilated)."""
    def __init__(self, dim: int, k: int = 8, dilation: int = 1):
        super().__init__()
        self.k, self.dil = k, dilation
        self.edgeconv = EdgeConv(
            nn.Sequential(
                nn.Linear(2 * dim, dim),
                nn.ReLU(inplace=True),
                nn.Linear(dim, dim)
            ),
            aggr='max'
        )

    def forward(self, x, pos, batch, knn_cache=None):
        ei = knn_cache
        if ei is None:                                    # build if absent
            ei = knn_graph(pos, k=self.k * self.dil,
                            batch=batch, loop=False)
        ei = ei[:, :: self.dil]                           # dilation step
        return x + self.edgeconv(x, ei)                   # residual
# --------------------------------------------------------------------------- #
class GlobalBlock(nn.Module):
    """Self-attention to ≤ 100 nearest neighbours (with caching)."""
    def __init__(self, dim: int, heads: int = 4,
                 dropout: float = 0.1, k_max: int = 100):
        super().__init__()
        self.k_max = min(k_max, 100)
        self.attn  = TransformerConv(
            in_channels=dim,
            out_channels=dim // heads,
            heads=heads,
            dropout=dropout,
            beta=True
        )

    def forward(self, x, pos, batch, knn_cache=None):
        ei = knn_cache
        if ei is None:                                    # build if absent
            k  = min(self.k_max, max(1, x.size(0) - 1))
            ei = knn_graph(pos, k=k, batch=batch, loop=False)
            ei = torch.unique(ei, dim=1)                  # drop duplicates
        return self.attn(x, ei)                           # residual already
# --------------------------------------------------------------------------- #
class LinePatchCrossAttn(nn.Module):
    """A node (line) queries *its own* patch tokens."""
    def __init__(self, dim: int, patch_dim: int, heads: int = 4):
        super().__init__()
        self.q_proj  = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(patch_dim, 2 * dim)
        self.mha     = nn.MultiheadAttention(dim, heads, batch_first=True)

    def forward(self, node_x, patch_x):
        q      = self.q_proj(node_x).unsqueeze(1)         # [N, 1, D]
        k, v   = self.kv_proj(patch_x).chunk(2, dim=-1)   # [N, P, D] each
        out, _ = self.mha(q, k, v)
        return node_x + out.squeeze(1)                    # residual add
# --------------------------------------------------------------------------- #
class LGXBlock(nn.Module):
    """One Local-Global-Cross block with k-NN caches for both sub-graphs."""
    def __init__(self, dim: int, patch_dim: int,
                 k: int = 8, dilation: int = 1,
                 heads: int = 4, dropout: float = 0.1,
                 global_k: int = 80):
        super().__init__()
        self.local   = LocalEdgeBlock(dim, k, dilation)
        self.global_ = GlobalBlock(dim, heads, dropout, k_max=min(global_k, 100))
        self.cross   = LinePatchCrossAttn(dim, patch_dim, heads)
        self.ln      = nn.LayerNorm(dim)

    def forward(self, node_x, pos, batch, patch_tok,
                caches: dict | None = None):
        """
        Parameters
        ----------
        caches : dict or None
            Expected keys: {'local', 'global'} holding edge-index tensors.
            Pass None the first time; the function returns the filled dict.
        """
        if caches is None:
            caches = {}
            # build both graphs once
            caches['local']  = knn_graph(
                pos, k=self.local.k * self.local.dil, batch=batch, loop=False)
            caches['global'] = knn_graph(
                pos, k=self.global_.k_max,            batch=batch, loop=False)

        # 1) Local EdgeConv (uses dilated slice of its cache)
        node_x = self.local(node_x, pos, batch, caches['local'])

        # 2) Global TransformerConv (100-NN)
        node_x = self.global_(node_x, pos, batch, caches['global'])

        # 3) Node ↔ patch cross-attention
        node_x = self.cross(node_x, patch_tok)

        return self.ln(node_x), caches



def GN(c: int) -> nn.GroupNorm:
    """
    Return a GroupNorm layer with num_groups chosen so that
    - it divides `c`
    - it is <= 16 (common rule-of-thumb for vision backbones)
    """
    for g in (16, 8, 4, 2, 1):
        if c % g == 0:
            return nn.GroupNorm(g, c, affine=True)
    # should never hit here, but be safe
    return nn.GroupNorm(1, c, affine=True)

class AttentionCNNCluster(pl.LightningModule):
    def __init__(self,
        # Model HParams
        in_channels_DeepLSD: int,
        in_channels: int,
        hidden_channels: int,
        hidden_channels_cnn:int,
        out_channels: int,
        geom_channels: int,
        roi_align_embedding_shape: tuple,
        num_layers: int,
        dropout: float = 0.5,
        act: str = 'relu',
        v2: bool = True,
        jk_layer: str = None,
        edge_sample_size = (32,16),
        edge_downsample_dim = 20,
        # Training HParams
        learning_rate: float = 1e-3,
        node_loss_w: float = 1.0,          # Weight for node loss
        edge_loss_w: float = 1.0,          # Weight for edge loss (new)
        threshold_structural: float = 0.5,  # Threshold for accuracy/recall calc
        delta_var: float = 0.5,
        delta_dist: float = 1.5,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 0.001,
        embed_dim: int = 128,
        mlp_dropout: float = 0.0,          # drop out for merge features
        skip_init=False,
        ):
        super().__init__()
        self.save_hyperparameters()
        


  
        # self.bands_cnn = nn.Sequential(
        #     nn.LazyConv2d(self.hparams.hidden_channels_cnn, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),

        #     nn.Conv2d(self.hparams.hidden_channels_cnn, self.hparams.hidden_channels_cnn * 2, kernel_size=3, padding=1),
        #     nn.ReLU(inplace=True),

        #     nn.AdaptiveAvgPool2d(1),   # Global average pooling → [B, 2h, 1, 1]
        #     nn.Flatten(1),             # → [B, 2h]
        #     nn.Linear(self.hparams.hidden_channels_cnn * 2, self.hparams.hidden_channels_cnn)
        # )

        # self.image_encoder = GlobalImageEncoder(out_dim=hidden_channels_cnn)
        # self.rgb_image_encoder = GlobalImageEncoder(out_dim=hidden_channels_cnn)

        D = out_channels                     # shorthand

        self.image_encoder = nn.Sequential(
            # conv1: 7×7, stride 2, padding 3 → 64 channels
            nn.LazyConv2d(self.hparams.hidden_channels_cnn, kernel_size=7, stride=2, padding=3, bias=False),
            GN(self.hparams.hidden_channels_cnn),          # ← swapped in
            nn.ReLU(inplace=True),

            # max-pool 3×3, stride 2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # two BasicBlock’s (64→64)

            # two BasicBlocks (make sure they also use GroupNorm – see step 3)
            BasicBlock(self.hparams.hidden_channels_cnn,
                    self.hparams.hidden_channels_cnn, norm_layer=GN),
            BasicBlock(self.hparams.hidden_channels_cnn,
                    self.hparams.hidden_channels_cnn, norm_layer=GN),


            # now downsample to fixed vector
            nn.AdaptiveAvgPool2d(1),    # → [B, 64, 1, 1]
            nn.Flatten(1),              # → [B, 64]
            nn.Linear(self.hparams.hidden_channels_cnn, self.hparams.hidden_channels_cnn),
            nn.Dropout(p=self.hparams.mlp_dropout),   # drop here after the final linear

        )

        # —————————————————  2.  Per-patch CNN  ————————————————————
        # keep spatial map; *no* AdaptivePool at the end
        self.patch_cnn_feat = nn.Sequential(
            nn.LazyConv2d(hidden_channels_cnn, 7, 2, 3, bias=False),
            GN(hidden_channels_cnn),          # ← swapped in
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            # two “basic” residual blocks (64 → 64)
            BasicBlock(hidden_channels_cnn, hidden_channels_cnn, norm_layer=GN),
            BasicBlock(hidden_channels_cnn, hidden_channels_cnn, norm_layer=GN),
        )
        self.patch_proj = nn.Linear(hidden_channels_cnn, D)   # token projector
        self.patch_pool = nn.AdaptiveAvgPool2d(1)             # global pool

        # —————————————————  3.  Geo-extra MLPs  ————————————————
        self.length_mlp = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 16))
        self.angle_mlp  = nn.Sequential(
            nn.Linear(3, 16), nn.ReLU(),
            nn.Linear(16, 16))
        self.pos_encoder = Fourier2DPositionalEncoding(8, 32)

        # —————————————————  4.  Node fuse  ————————————————
        fuse_in = 2*hidden_channels_cnn + 2*16 + 32
        self.node_fuse = nn.Sequential(
            nn.Linear(fuse_in, D), nn.LayerNorm(D), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(D, D))

        # —————————————————  5.  LGX stack  ————————————————
        lgx = []
        for i in range(num_layers):          # e.g. 6
            lgx.append(LGXBlock(
                dim=D,
                patch_dim=D,
                k=9,
                dilation=2**(i % 3),        # 1,2,4,1,2,4…
                heads=4,
                dropout=dropout))
        self.lgx_layers = nn.ModuleList(lgx)

        # —————————————————  6.  Heads  ————————————————
        self.mlp_textural_structural = nn.Sequential(
            nn.Linear(D + 2*hidden_channels_cnn + 16, D),
            nn.LayerNorm(D), nn.ReLU(),
            nn.Linear(D, 1), nn.Dropout(dropout))

        self.plane_embed_head = nn.Sequential(
            nn.Linear(D, 2*128), nn.LayerNorm(2*128), nn.ReLU(inplace=True),
            nn.Linear(2*128, 128))

        self.criterion = nn.BCEWithLogitsLoss()
        self.delta_var, self.delta_dist = delta_var, delta_dist
        self.alpha, self.beta, self.gamma = alpha, beta, gamma

        # Containers for metrics
        self.validation_step_outputs = []
        self.test_step_outputs = []

    # ─────────────────────────────────────────────────────────────────
    #   F O R W A R D
    # ─────────────────────────────────────────────────────────────────
    def forward(self, batch):
        # 0. unpack ---------------------------------------------------
        edge_index, batch_idx = batch.edge_index, batch.batch
        geo      = batch.geo                       # [N,6]
        img      = batch.img                       # [1,H,W,C]
        rgb      = batch.rgb_patches               # [N,h,w,3]
        angle_pt = batch.angle_field_patches       # [N,h2,w2,3]
        dist_pt  = batch.distance_patches          # [N,h2,w2,1]

        # 1. global-image feature ------------------------------------
        # 1) keep the batch dim
        img = batch.img
        if img.dim() == 3:                     # [H,W,C]  – single graph
            img = img.permute(2, 0, 1).unsqueeze(0)        # → [1,3,H,W]
        elif img.dim() == 4:                   # [B,H,W,C] – mini-batch of graphs
            img = img.permute(0, 3, 1, 2)                    # → [B,3,H,W]
        else:
            raise ValueError(f"Unexpected img shape {img.shape}")
        img = img.float()                      # cast once

        # 2) run the image-level CNN once per image in the mini-batch
        img_feat = self.image_encoder(img)              # [B, Cʹ]

        # 3) scatter each image-feature to *its own* nodes
        #    batch.batch is a vector of length N telling you which graph
        #    (0 … B-1) every node came from.
        img_feat_exp = img_feat[batch.batch]            # [N, Cʹ]

        # 2. build full patch tensor ---------------------------------
        rgb = rgb.permute(0,3,1,2).float()           # [N,3,H1,W1]
        angle_pt = angle_pt.permute(0,3,1,2)
        dist_pt  = dist_pt.permute(0,3,1,2)
        full_patches = torch.cat([rgb, angle_pt, dist_pt], dim=1)

        # 2.a spatial fmap & tokens ----------------------------------
        fmap = self.patch_cnn_feat(full_patches)     # [N,C,H',W']
        N, C, Hf, Wf = fmap.shape
        P = Hf * Wf
        patch_tok = fmap.flatten(2).permute(0,2,1)   # [N,P,C]
        patch_tok = self.patch_proj(patch_tok)       # [N,P,D]

        # 2.b global pooled patch vector -----------------------------
        cnn_global = self.patch_pool(fmap).flatten(1)  # [N,C]

        # 3. extra geo features --------------------------------------
        pos_enc    = self.pos_encoder(geo[:,:2])
        len_feat   = self.length_mlp(geo[:,2:3])
        ang_feat   = self.angle_mlp (geo[:,3:6])

        # 4. initial node vector -------------------------------------
        #img_feat_exp = img_feat.expand(N, -1)         # [N,512]
        concat = torch.cat([cnn_global, pos_enc, len_feat,
                            ang_feat, img_feat_exp], dim=1)
        node_x = self.node_fuse(concat)               # [N,D]

        # 5. LGX propagation -----------------------------------------
        pos = geo[:,:2]
        knn_ei = None

        for blk in self.lgx_layers:
            node_x, knn_ei = blk(node_x, pos, batch_idx, patch_tok, knn_ei)

        features = node_x                             # final [N,D]

        # 6. heads ---------------------------------------------------
        struct_feats = torch.cat([features, cnn_global,
                                  img_feat_exp, len_feat], dim=1)
        node_logits  = self.mlp_textural_structural(struct_feats)

        plane_emb = F.normalize(self.plane_embed_head(features), p=2, dim=1)

        return node_logits, plane_emb

    def training_step(self, batch, batch_idx):
            node_logits, plane_emb = self(batch)

            # 1) binary structural line loss ----------------------------------
            node_labels = batch.y.view(-1,1).float()
            
            node_logits = node_logits.view(-1,1).float()

    # (N,)
            node_loss = self.criterion(node_logits, node_labels)
            
            

            # 2) discriminative clustering loss -------------------------------
            # plane_labels = batch.plane_id.long()                       # (N,)

            # Get raw list of overlapping‐plane cliques
            raw_cliques = batch.plane_id[0]

            # Remove any clique with fewer than 3 lines
            filtered_cliques = [c for c in raw_cliques if len(c) >= 3]

            N = plane_emb.size(0)

            gt_flat = cliques_to_flat_labels(filtered_cliques, N, tie_break="largest").numpy()
            gt_flat = torch.tensor(gt_flat).long().to(plane_emb.device)

            var_loss, dist_loss, reg_loss = multi_clique_discriminative_loss(
                plane_emb,
                filtered_cliques,
                self.delta_var,
                self.delta_dist,
                self.alpha,
                self.beta,
                self.gamma,
            )
            


            loss = 0.4 * node_loss + 0.6 * (var_loss + dist_loss + reg_loss) 

            self.log_dict({
                "train/struct_bce": node_loss,
                "train/var_loss":    var_loss,
                "train/dist_loss":   dist_loss,
                "train/reg_loss":    reg_loss,
                "train/total_loss":  (var_loss + dist_loss + reg_loss),
                "train/loss":       loss,
            }, on_step=True, prog_bar=True, logger=True)
            return loss

            
        # Validation
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        node_logits, plane_emb = self(batch)
        node_labels = batch.y.float()

        # --- structural (link‑prediction) metrics ----------------------------
        preds = node_logits.sigmoid().detach().cpu().numpy()
        gt_bin = node_labels.cpu().numpy()
        if torch.unique(node_labels).numel() > 1:
            roc = roc_auc_score(gt_bin, preds)
            ap = average_precision_score(gt_bin, preds)
        else:
            roc, ap = 0.0, 0.0

        # --- clustering labels ----------------------------------------------
        cliques: List[List[int]] = batch.plane_id[0]
        N = batch.num_nodes
        gt_flat = cliques_to_flat_labels(cliques, N, tie_break="largest").numpy()
        pred_plane = hdbscan_cluster(plane_emb.cpu())
        pred_np = pred_plane.detach().cpu().numpy()

        valid_mask = gt_flat != -1
        if valid_mask.sum():
            gt_filt, pred_filt = gt_flat[valid_mask], pred_np[valid_mask]

            # core clustering scores
            ari = adjusted_rand_score(gt_filt, pred_filt)
            nmi = normalized_mutual_info_score(gt_filt, pred_filt)

            # newly added scores
            p_pair, r_pair, f1_pair = _pairwise_prf1(gt_filt, pred_filt)
            fmi = fowlkes_mallows_score(gt_filt, pred_filt)
            vme = v_measure_score(gt_filt, pred_filt)
            hung_acc = _hungarian_accuracy(gt_filt, pred_filt)
        else:
            ari = nmi = p_pair = r_pair = f1_pair = fmi = vme = hung_acc = 0.0

        # store for epoch‑end aggregation
        self.validation_step_outputs.append({
            "roc": roc, "ap": ap,
            "ari": ari, "nmi": nmi,
            "p_pair": p_pair, "r_pair": r_pair, "f1_pair": f1_pair,
            "fmi": fmi, "vme": vme, "hung": hung_acc,
        })

        # per‑batch logging ---------------------------------------------------
        self.log_dict({
            "val/roc": roc, "val/ap": ap,
            "val/ari": ari, "val/nmi": nmi,
            "val/pair_P": p_pair, "val/pair_R": r_pair, "val/pair_F1": f1_pair,
            "val/FMI": fmi, "val/V‑measure": vme, "val/HungAcc": hung_acc,
        }, on_step=True, on_epoch=False, prog_bar=False, logger=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Test
    # ──────────────────────────────────────────────────────────────────────────
    def test_step(self, batch, batch_idx):
        node_logits, plane_emb = self(batch)
        node_labels = batch.y.float()

        preds = node_logits.sigmoid().detach().cpu().numpy()
        gt_bin = node_labels.cpu().numpy()
        if torch.unique(node_labels).numel() > 1:
            roc = roc_auc_score(gt_bin, preds)
            ap = average_precision_score(gt_bin, preds)
        else:
            roc, ap = 0.0, 0.0

        cliques: List[List[int]] = batch.plane_id[0]
        N = batch.num_nodes
        gt_flat = cliques_to_flat_labels(cliques, N, tie_break="largest").numpy()
        pred_plane = hdbscan_cluster(plane_emb)
        pred_np = pred_plane.detach().cpu().numpy()

        valid_mask = gt_flat != -1
        if not valid_mask.sum():
            # all noise → return zeros
            metrics = {k: 0.0 for k in ["roc", "ap", "ari", "nmi",
                                         "p_pair", "r_pair", "f1_pair",
                                         "fmi", "vme", "hung"]}
        else:
            gt_filt, pred_filt = gt_flat[valid_mask], pred_np[valid_mask]
            ari = adjusted_rand_score(gt_filt, pred_filt)
            nmi = normalized_mutual_info_score(gt_filt, pred_filt)
            p_pair, r_pair, f1_pair = _pairwise_prf1(gt_filt, pred_filt)
            fmi = fowlkes_mallows_score(gt_filt, pred_filt)
            vme = v_measure_score(gt_filt, pred_filt)
            hung_acc = _hungarian_accuracy(gt_filt, pred_filt)
            metrics = {"roc": roc, "ap": ap, "ari": ari, "nmi": nmi,
                       "p_pair": p_pair, "r_pair": r_pair, "f1_pair": f1_pair,
                       "fmi": fmi, "vme": vme, "hung": hung_acc}

        # store for epoch‑end aggregation
        self.test_step_outputs.append(metrics)

        # log per batch -------------------------------------------------------
        self.log_dict({f"test/{k}": v for k, v in metrics.items()},
                      on_step=True, on_epoch=False, prog_bar=False, logger=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Epoch‑end aggregation (now handles all metrics) ------------------------
    # ──────────────────────────────────────────────────────────────────────────
    def _aggregate_and_log(self, outputs: list[Dict[str, float]], prefix: str):
        if not outputs:
            return

        # stack into tensors for convenience
        keys = outputs[0].keys()
        stacked = {k: torch.tensor([o[k] for o in outputs]) for k in keys}

        stats = {}
        for k, v in stacked.items():
            stats[f"{prefix}/{k}_mean"] = v.mean()
            stats[f"{prefix}/{k}_std"] = v.std(unbiased=False)

        self.log_dict(stats, prog_bar=True, logger=True)

       
        outputs.clear()  # free memory


    def on_validation_epoch_end(self) -> None:
        self._aggregate_and_log(self.validation_step_outputs, "val")


    def on_test_epoch_end(self) -> None:
        self._aggregate_and_log(self.test_step_outputs, "test")

    # ------------------------------------------------------------------
    # Prediction (inference helper)
    # ------------------------------------------------------------------
    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> Dict[str, Any]:
        node_logits, plane_emb = self(batch)
        node_prob = node_logits.sigmoid()
        return {"node_prob": node_prob, "plane_emb": plane_emb}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr            = self.hparams.learning_rate,   # same LR usually works
            betas         = (0.9, 0.999),
            eps           = 1e-8,
            weight_decay  = 1e-4,        # **decoupled** L2, not mixed into the grads
        )

        # warm-up → cosine ↓ to zero (good default for vision & graphs)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=0.0
        )

        return [optimizer], [scheduler]

