# Combined file: gat_textural_structural_lightning.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch_geometric.nn as pyg_nn
from torch.optim import Adam
from torch_geometric.data import Batch # Import Batch for type hinting if needed
from sklearn.metrics import roc_curve, auc, accuracy_score, recall_score
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
    dvec  = node_geo[:, 2:4]  # (N,2), unit
    leng  = node_geo[:, 4:5]  # (N,1)

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
        perp_ij,       # 1
        perp_ji,       # 2
        cos_th,        # 3
        abs_sin,       # 4
        sin_th,        # 5
        ang_rad,       # 6
        li,            # 7
        lj,            # 8
        log_len_ratio, # 9
        par_score      #10
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
    def __init__(self, num_dim, num_heads, skip_init=False):
        super().__init__()
        self.attn = MultiHeadedAttention(num_heads, num_dim)
        self.mlp = MLP([num_dim * 2, num_dim * 2, num_dim], do_bn=True)
        nn.init.constant_(self.mlp[-1].bias, 0.0)
        if skip_init:
            self.register_parameter('scaling', nn.Parameter(torch.tensor(0.)))
        else:
            self.scaling = 1.

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1)) * self.scaling

class SelfAttnLayer(nn.Module):
    def __init__(self, feature_dim, skip_init):
        super().__init__()
     
        self.update = AttentionalPropagation(feature_dim, 4, skip_init)

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

from torchvision.models.resnet import BasicBlock

    
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

        self.bands_cnn = nn.Sequential(
            # conv1: 7×7, stride 2, padding 3 → 64 channels
            nn.LazyConv2d(self.hparams.hidden_channels_cnn, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.hparams.hidden_channels_cnn),
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
      
        
        self.edge_loss_w = edge_loss_w
  
        self.node_fuse = nn.Sequential(
            nn.Linear(self.hparams.hidden_channels_cnn +  self.hparams.geom_channels,
                      self.hparams.out_channels),
            nn.BatchNorm1d(self.hparams.out_channels),

            nn.ReLU(),
            nn.Linear(self.hparams.out_channels,
                      self.hparams.out_channels),
        )


        self.edge_geo_channels = 11
        # GNN layers
        layers = []
        for i in range(self.hparams.num_layers):
            if i % 2 == 0:
                layers.append(SelfAttnLayer(self.hparams.out_channels, self.hparams.skip_init))
            else:
                layers.append(EdgeSamplerLayer(self.hparams.out_channels, self.edge_geo_channels, hidden_dim=self.hparams.out_channels))
                # layers.append(LocalEdgeLayer(self.hparams.out_channels))

        self.layers = nn.ModuleList(layers)

        # Node prediction head
        self.mlp_textural_structural = nn.Sequential(
            nn.Linear(self.hparams.out_channels + self.hparams.hidden_channels_cnn, self.hparams.out_channels),
            nn.BatchNorm1d(self.hparams.out_channels),

            nn.ReLU(),
            nn.Linear(self.hparams.out_channels, 1)
        )

        self.edge_predictor = nn.Sequential(
            nn.Linear(2*self.hparams.out_channels + self.hparams.out_channels * 2 + self.edge_geo_channels, self.hparams.out_channels),
            nn.BatchNorm1d(self.hparams.out_channels),

            nn.ReLU(),
            nn.Linear(self.hparams.out_channels, 1)
        )

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Containers for metrics
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        self.log_sigma_node = nn.Parameter(torch.zeros(()))
        self.log_sigma_edge = nn.Parameter(torch.zeros(()))
                    
    def forward(self, batch):
        
        edge_index, batch_idx = batch.edge_index, batch.batch
        geo = batch.geo



        rgb_patches = batch.rgb_patches
        angle_field_patches = batch.angle_field_patches
        distance_patches = batch.distance_patches
   

        cnn_features = []

        for idx in range(len(rgb_patches)):
            rgb_patches_idx= (rgb_patches[idx]).permute(2,0,1)
            angle_field_patches_idx = (angle_field_patches[idx]).permute(2,0,1)
            distance_patches_idx =(distance_patches[idx]).permute(2,0,1)


            H2, W2 = angle_field_patches.shape[1:3]
            img_rs = F.interpolate(
                rgb_patches_idx.unsqueeze(0),           # add batch
                size=(H2, W2),
                mode="bilinear",
                align_corners=False
            ).squeeze(0)                   # remove batch  → [C1, H2, W2]


            full_patch = torch.cat([img_rs, angle_field_patches_idx, distance_patches_idx], dim=0)


            # 3) CNN head (expects a batch dim)
            out = self.bands_cnn(full_patch.unsqueeze(0))            # [1, output_dim]
            cnn_features.append(out.squeeze(0))            # [output_dim]

        cnn_features = torch.stack(cnn_features, dim=0)          # [N, output_dim]
     




        concat_feat = torch.cat([cnn_features, geo], dim=1)

        concat_feat = self.node_fuse(concat_feat)
        
        # Prepare for global attention
        roi_dense, mask = to_dense_batch(concat_feat, batch_idx)
        desc = roi_dense.transpose(1, 2)
        
        
     
        
        
        src, dst = batch.edge_index
        edge_geo = edge_geometry(geo, src, dst) 

        #local_edge_geo = edge_geo[batch.flat_idx_local] 
        # Alternate layers
        for layer in self.layers:
            if isinstance(layer, SelfAttnLayer):
                desc = layer(desc)
            else:
                flat = desc.transpose(1,2)[mask]
                flat_norm = layer.ln(flat)    # assuming you add ln = nn.LayerNorm(node_dim) to EdgeSamplerLayer

                delta = layer(flat_norm, edge_index, edge_geo)

                delta_dense, _ = to_dense_batch(delta, batch_idx)
                desc = desc + delta_dense.transpose(1,2)

        # Collapse to node features
        features = desc.transpose(1,2)[mask]
        
        features = F.dropout(features, p=self.hparams.mlp_dropout, training=self.training)

        struct_feats = torch.cat([features, cnn_features], dim=1)  # [N, D+2*D]

        # Node logits
        node_logits = self.mlp_textural_structural(struct_feats)

        # Edge logits


        src, dst = batch.full_edge_index

        edge_geo_full = edge_geometry(geo, src, dst) 

        h_src, h_dst = features[src], features[dst]
        concat_src, concat_dst = concat_feat[src], concat_feat[dst] # CNN features
        

       
        edge_in = torch.cat([
            0.5 * (h_src + h_dst),        # symmetric mean      [E, D]
            (h_src - h_dst).abs(),        # symmetric distance  [E, D]
            0.5 * (concat_src + concat_dst),        # symmetric mean      [E, D]
            (concat_src - concat_dst).abs(),        # symmetric distance  [E, D]
            edge_geo_full,
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
        loss = (node_loss * torch.exp(-2*self.log_sigma_node) +
                edge_loss * torch.exp(-2*self.log_sigma_edge) +
                self.log_sigma_node + self.log_sigma_edge) * 0.5
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

    def validation_step(self, batch: Batch, batch_idx: int):
        # forward
        node_logits, edge_logits = self(batch)
        # labels
        node_labels      = batch.y.float()               # [N,1]
        full_edge_labels = batch.full_edge_labels.float()  # [N*N,1]

        # losses
        node_loss = self.criterion(node_logits, node_labels)
        edge_loss = self.criterion(edge_logits, full_edge_labels)
        total_loss = (node_loss * torch.exp(-2*self.log_sigma_node) +
                edge_loss * torch.exp(-2*self.log_sigma_edge) +
                self.log_sigma_node + self.log_sigma_edge) * 0.5


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
        total_loss = (node_loss * torch.exp(-2*self.log_sigma_node) +
                edge_loss * torch.exp(-2*self.log_sigma_edge) +
                self.log_sigma_node + self.log_sigma_edge) * 0.5

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
