# Combined file: gat_textural_structural_lightning.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch_geometric.nn as pyg_nn
from torch.optim import Adam
from torch_geometric.data import Batch # Import Batch for type hinting if needed
from sklearn.metrics import accuracy_score, recall_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from typing import Optional
import logging
import cv2
import os 
import matplotlib.pyplot as plt
import random
from torch_geometric.utils import dropout_edge
from dataset_inductive import seg_seg_dist

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


def edge_geometry(node_geo: torch.Tensor,
                  src: torch.Tensor,
                  dst: torch.Tensor) -> torch.Tensor:
    """
    node_geo : [N,5]  = [mid_x, mid_y, dir_x, dir_y, length]
    src,dst  : [E]    edge_index rows (already on the right device)
    returns  : [E,5]  =  [d_mid, cosθ, |sinθ|, len_i, len_j]
    """
    mid      = node_geo[:, 0:2]             # [N,2]
    dir_norm = node_geo[:, 2:4]             # [N,2]
    length   = node_geo[:, 4:5]             # [N,1]

    # -- distance between mid-points -----------------------------------------
    d_mid = (mid[src] - mid[dst]).norm(dim=1, keepdim=True)      # [E,1]

    # -- orientation relationship --------------------------------------------
    dir_i  = dir_norm[src]                                        # [E,2]
    dir_j  = dir_norm[dst]                                        # [E,2]
    cos_th = (dir_i * dir_j).sum(1, keepdim=True)                 # [E,1]
    sin_th = (dir_i[:,0]*dir_j[:,1] - dir_i[:,1]*dir_j[:,0]).abs().unsqueeze(1)

    # -- individual lengths ---------------------------------------------------
    len_i  = length[src]                                           # [E,1]
    len_j  = length[dst]                                           # [E,1]

    return torch.cat([d_mid, cos_th, sin_th, len_i, len_j], dim=1) # [E,5]

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
                 aggr: str = 'mean', dropout=0.0):
        super().__init__(aggr=aggr)
        hidden_dim = hidden_dim or node_dim

        # now we take [x_i || x_j || edge_attr] → hidden → node_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*node_dim + edge_attr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),            # ← drop here
            nn.Linear(hidden_dim, node_dim),
            nn.Dropout(p=dropout)             # ← and maybe here, too
        )
        # self.ln = nn.LayerNorm(node_dim)


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
        m = torch.cat([x_i, x_j, edge_attr], dim=-1)  # -> [E, 2*node_dim+edge_attr_dim]
        return self.edge_mlp(m)                       # -> [E, node_dim]

class EdgeSamplerLayerNoPatch(MessagePassing):
    def __init__(self,
                 node_dim: int,
                 hidden_dim: Optional[int] = None,
                 aggr: str = 'mean', dropout=0.0):
        super().__init__(aggr=aggr)
        hidden_dim = hidden_dim or node_dim

        # now we take [x_i || x_j || edge_attr] → hidden → node_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(2*node_dim + edge_attr_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),            # ← drop here
            nn.Linear(hidden_dim, node_dim),
            nn.Dropout(p=dropout)             # ← and maybe here, too
        )

    def forward(self,
                x: torch.Tensor,
                edge_index: torch.LongTensor,
                ) -> torch.Tensor:
        # x:         [N, node_dim]
        # edge_index:[2, E]
        # edge_attr: [E, edge_attr_dim]
        return self.propagate(edge_index, x=x)

    def message(self,
                x_i: torch.Tensor,
                x_j: torch.Tensor,
              ) -> torch.Tensor:
        # x_i:      [E, node_dim]  dest
        # x_j:      [E, node_dim]  src
        # edge_attr:[E, edge_attr_dim]
        m = torch.cat([x_i, x_j], dim=-1)  # -> [E, 2*node_dim+edge_attr_dim]
        return self.edge_mlp(m)                       # -> [E, node_dim]

    
from torch_geometric.nn import GATv2Conv

class AttentionEdgeSampleLinear(pl.LightningModule):
    def __init__(self,
        # Model HParams
        in_channels_DeepLSD: int,
        in_channels: int,
        hidden_channels: int,
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
        

        node_height, node_width =  self.hparams.roi_align_embedding_shape
        edge_height, edge_width =  self.hparams.edge_sample_size
        self.edge_patch_enc = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(3* edge_height * edge_width, self.hparams.edge_downsample_dim),
        # nn.LayerNorm(self.hparams.edge_downsample_dim),         # ← swapped out

        nn.Dropout(p=self.hparams.mlp_dropout),
        nn.ReLU(),
        )
        self.node_linear = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(3*node_height*node_width, self.hparams.hidden_channels),
        # nn.LayerNorm(self.hparams.hidden_channels),         # ← swapped out

        nn.Dropout(p=self.hparams.mlp_dropout),
        nn.ReLU(),
        )
        
        
        self.edge_loss_w = edge_loss_w
        self.edge_predictor = nn.Sequential(
            nn.Linear(2*self.hparams.out_channels + self.hparams.geom_channels, self.hparams.out_channels),
            # nn.LayerNorm(self.hparams.out_channels),         # ← swapped out

            nn.ReLU(),
            nn.Linear(self.hparams.out_channels, 1)
        )

        self.node_fuse = nn.Sequential(
            nn.Linear(self.hparams.hidden_channels + self.hparams.in_channels_DeepLSD +  self.hparams.geom_channels,
                      self.hparams.out_channels),
            # nn.LayerNorm(self.hparams.out_channels),         # ← swapped out

            nn.ReLU(),
            nn.Linear(self.hparams.out_channels,
                      self.hparams.out_channels),
        )

        # GNN layers
        layers = []
        for i in range(self.hparams.num_layers):
            if i % 2 == 0:
                layers.append(SelfAttnLayer(self.hparams.out_channels, self.hparams.skip_init))
            else:
                layers.append(EdgeSamplerLayer(node_dim=self.hparams.out_channels,
                edge_attr_dim=self.hparams.edge_downsample_dim,
                hidden_dim=self.hparams.out_channels))
                
                # layers.append(

                #     GATv2Conv(
                #         in_channels  = self.hparams.out_channels,
                #         out_channels = self.hparams.out_channels,
                #         heads        = 2,
                #         concat       = False,             # keep dims = out_channels
                #         dropout      = self.hparams.dropout,
                #         edge_dim     = self.hparams.edge_downsample_dim
                #     )
                # )
                # layers.append(LocalEdgeLayer(self.hparams.out_channels))

        self.layers = nn.ModuleList(layers)

        # Node prediction head
        self.mlp_textural_structural = nn.Sequential(
            nn.Linear(self.hparams.out_channels, self.hparams.out_channels),
            # nn.LayerNorm(self.hparams.out_channels),         # ← swapped out

            nn.ReLU(),
            nn.Linear(self.hparams.out_channels, 1)
        )

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Containers for metrics
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
      
       
                    
    def forward(self, batch):
        
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        roi_features = batch.roi_features 
        geo = batch.geo
        # ROI conv and fuse
        roi_feats = self.node_linear(roi_features)
        concat_feat = torch.cat([roi_feats, x, geo], dim=1)
        concat_feat = self.node_fuse(concat_feat)
        
        # Prepare for global attention
        roi_dense, mask = to_dense_batch(concat_feat, batch_idx)
        desc = roi_dense.transpose(1, 2)
        
        
     
        
        edge_patch = self.edge_patch_enc(batch.edge_attr)  

        
        src, dst = batch.full_edge_index
        edge_geo = edge_geometry(geo, src, dst)             # [E,5]
        #local_edge_geo = edge_geo[batch.flat_idx_local] 
        # Alternate layers
        for layer in self.layers:
            if isinstance(layer, SelfAttnLayer):
                desc = layer(desc)
            else:
                flat = desc.transpose(1,2)[mask]
                flat_norm = layer.ln(flat)   

                delta = layer(flat_norm, edge_index, edge_patch)

                delta_dense, _ = to_dense_batch(delta, batch_idx)
                desc = desc + delta_dense.transpose(1,2)
                
                # delta = layer(flat, batch.edge_index, edge_attr=edge_patch)
                # x = F.relu(flat + delta)
                # desc, mask = to_dense_batch(x, batch.batch)
                # desc        = desc.transpose(1,2)


        # Collapse to node features
        features = desc.transpose(1,2)[mask]
        
        features = F.dropout(features, p=self.hparams.mlp_dropout, training=self.training)

        # Node logits
        node_logits = self.mlp_textural_structural(features)

        # Edge logits

        h_src, h_dst = features[src], features[dst]
       
        edge_in = torch.cat([
            0.5 * (h_src + h_dst),        # symmetric mean      [E, D]t
            (h_src - h_dst).abs(),        # symmetric distance  [E, D]
            edge_geo # geometric extras    [E, 5]
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
        
      
       
        loss = (node_loss + edge_loss) * 0.5

        # loss = self.hparams.node_loss_w*node_loss + self.hparams.edge_loss_w*edge_loss
        
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
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200),
            'interval': 'epoch'
        }
        return [opt], [scheduler]
    
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, f1_score

# put this near your imports
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import math

# class SmoothedAsymmetricFocalLoss(nn.Module):
#     """
#     γ_neg   –> focusing on easy negatives  (2 is standard)
#     γ_pos   –> focusing on easy positives  (0 keeps recall high)
#     eps     –> label-smoothing factor      (0.05 for ~5 % noise)
#     """
#     def __init__(self, gamma_neg: float = 2.0,
#                        gamma_pos: float = 0.0,
#                        eps: float = 0.05,
#                        reduction: str = "mean"):
#         super().__init__()
#         self.gn, self.gp, self.eps, self.reduction = gamma_neg, gamma_pos, eps, reduction

#     def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         # ----- 1. label smoothing -------------------------------------------
#         targets = targets * (1.0 - self.eps) + 0.5 * self.eps

#         # ----- 2. standard BCE components -----------------------------------
#         prob   = torch.sigmoid(logits)
#         loss_p = targets * F.binary_cross_entropy_with_logits(
#                     logits, torch.ones_like(targets), reduction="none")
#         loss_n = (1.0 - targets) * F.binary_cross_entropy_with_logits(
#                     logits, torch.zeros_like(targets), reduction="none")

#         # ----- 3. asymmetric focal weighting --------------------------------
#         focal_p = (1.0 - prob).pow(self.gp)   # usually gp = 0
#         focal_n = prob.pow(self.gn)           # gn > 0 down-weights easy negs
#         loss    = focal_p * loss_p + focal_n * loss_n

#         if self.reduction == "mean":
#             return loss.mean()
#         elif self.reduction == "sum":
#             return loss.sum()
#         return loss

from sklearn.metrics import roc_auc_score, average_precision_score

class AttentionEdgeSampleLinearNoWeight(pl.LightningModule):
    def __init__(self,
        # Model HParams
        in_channels_DeepLSD: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        geom_channels: int,
        roi_align_embedding_shape: tuple,
        num_layers: int,
        # node_pos_weight,
        # edge_pos_weight,
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
        
         
        node_height, node_width =  self.hparams.roi_align_embedding_shape
        edge_height, edge_width =  self.hparams.edge_sample_size
        self.edge_patch_enc = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(3* edge_height * edge_width, self.hparams.edge_downsample_dim),
        #nn.LayerNorm(self.hparams.edge_downsample_dim),         # ← swapped out

        nn.Dropout(p=self.hparams.mlp_dropout),
        nn.ReLU(),
        )
        self.node_linear = nn.Sequential(
        nn.Flatten(start_dim=1),
        nn.Linear(3*node_height*node_width, self.hparams.hidden_channels),
        #nn.LayerNorm(self.hparams.hidden_channels),         # ← swapped out

        nn.Dropout(p=self.hparams.mlp_dropout),
        nn.ReLU(),
        )
        
        
        self.edge_loss_w = edge_loss_w
        self.edge_predictor = nn.Sequential(
            nn.Linear(2*self.hparams.out_channels + self.hparams.geom_channels, self.hparams.out_channels),
            #nn.LayerNorm(self.hparams.out_channels),         # ← swapped out

            nn.ReLU(),
            nn.Linear(self.hparams.out_channels, 1)
        )
        



        self.node_fuse = nn.Sequential(
            nn.Linear(self.hparams.hidden_channels + self.hparams.in_channels_DeepLSD +  self.hparams.geom_channels,
                      self.hparams.out_channels),
            #nn.LayerNorm(self.hparams.out_channels),         # ← swapped out

            nn.ReLU(),
            nn.Linear(self.hparams.out_channels,
                      self.hparams.out_channels),
        )

        # GNN layers
        layers = []
        for i in range(self.hparams.num_layers):
            if i % 2 == 0:
                layers.append(SelfAttnLayer(self.hparams.out_channels, self.hparams.skip_init))
            else:
                layers.append(EdgeSamplerLayer(node_dim=self.hparams.out_channels,
                edge_attr_dim=self.hparams.edge_downsample_dim,
                hidden_dim=self.hparams.out_channels))
                
                
                # layers.append(LocalEdgeLayer(self.hparams.out_channels))

        self.layers = nn.ModuleList(layers)

        # Node prediction head
        self.mlp_textural_structural = nn.Sequential(
            nn.Linear(self.hparams.out_channels, self.hparams.out_channels),
            # nn.LayerNorm(self.hparams.out_channels),         # ← swapped out

            nn.ReLU(),
            nn.Linear(self.hparams.out_channels, 1)
        )

        # # Loss
        self.criterion = nn.BCEWithLogitsLoss()
        
        # self.node_criterion = nn.BCEWithLogitsLoss(
        #     pos_weight=torch.tensor(self.hparams.node_pos_weight, dtype=torch.float)
        # )
        # self.edge_criterion = SmoothedAsymmetricFocalLoss(
        #     gamma_neg=2.0,   # same hard-negative focus as before
        #     gamma_pos=0.0,   # keep positives recall-friendly
        #     eps=0.05)        # assume ≈5 % label noise

        # self.edge_criterion = nn.BCEWithLogitsLoss(
        #     pos_weight=torch.tensor(self.hparams.edge_pos_weight, dtype=torch.float)
        # )
        # Containers for metrics
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
      
       
                    
    def forward(self, batch):
        
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        roi_features = batch.roi_features 
        geo = batch.geo
        # ROI conv and fuse
        roi_feats = self.node_linear(roi_features)
        concat_feat = torch.cat([roi_feats, x, geo], dim=1)
        concat_feat = self.node_fuse(concat_feat)
        
        # Prepare for global attention
        roi_dense, mask = to_dense_batch(concat_feat, batch_idx)
        desc = roi_dense.transpose(1, 2)
        
        
     
        
        edge_patch = self.edge_patch_enc(batch.edge_attr)  

        
        src, dst = batch.full_edge_index
        edge_geo = edge_geometry(geo, src, dst)             # [E,5]
        #local_edge_geo = edge_geo[batch.flat_idx_local] 
        # Alternate layers
        for layer in self.layers:
            if isinstance(layer, SelfAttnLayer):
                desc = layer(desc)
            else:
                flat = desc.transpose(1,2)[mask]
                # flat_norm = layer.ln(flat)   

                delta = layer(flat, edge_index, edge_patch)

                delta_dense, _ = to_dense_batch(delta, batch_idx)
                desc = desc + delta_dense.transpose(1,2)
                



        # Collapse to node features
        features = desc.transpose(1,2)[mask]
        
        #features = F.dropout(features, p=self.hparams.mlp_dropout, training=self.training)

        # Node logits
        node_logits = self.mlp_textural_structural(features)

        # Edge logits

        h_src, h_dst = features[src], features[dst]
       
        edge_in = torch.cat([
            0.5 * (h_src + h_dst),        # symmetric mean      [E, D]t
            (h_src - h_dst).abs(),        # symmetric distance  [E, D]
            edge_geo # geometric extras    [E, 5]
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
        node_loss = self.criterion(node_logits, node_labels)

        
        # node_loss = self.node_criterion(node_logits, node_labels)
        # edge_loss = self.edge_criterion(edge_logits, full_edge_labels)

       
        loss = (node_loss + edge_loss) * 0.5

        # loss = self.hparams.node_loss_w*node_loss + self.hparams.edge_loss_w*edge_loss
        
        # # # metrics
        # # with torch.no_grad():
        # #     n_probs=torch.sigmoid(sampled_node_logits); n_preds=(n_probs>=self.hparams.threshold_structural).int().detach().cpu().numpy().ravel()
        # #     node_acc=accuracy_score(sampled_node_labels.int().detach().cpu().numpy().ravel(),n_preds)
        # #     e_probs=torch.sigmoid(sampled_edge_logits); e_preds=(e_probs>=self.hparams.threshold_structural).int().detach().cpu().numpy().ravel()
        # #     edge_acc=accuracy_score(sampled_edge_labels.int().detach().cpu().numpy().ravel(), e_preds)
            
            
        # self.log('train_loss',loss,on_step=True,on_epoch=False, prog_bar=True, logger=True)
        # # self.log('train_node_acc',node_acc,on_step=True,on_epoch=False, prog_bar=True, logger=True)
        # # self.log('train_edge_acc',edge_acc,on_step=True,on_epoch=False, prog_bar=True, logger=True)
        
        
        # return loss
        
        # with torch.no_grad():
            # edge_probs = edge_logits.sigmoid().flatten()
            # labels     = full_edge_labels.flatten()

        # 1) collect positive indices (and subsample if too many)
        # pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
        # max_pos = 1024                        # cap positives to 1024
        # if pos_idx.numel() > max_pos:
        #     pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)[:max_pos]]

        # # 2) build negative pool around your boundary
        # neg_mask  = (labels == 0)
        # best_thr  = 0.85; margin = 0.10
        #     low, high = best_thr - margin, best_thr + margin
        #     hard_neg  = neg_mask & (edge_probs >= low) & (edge_probs <= high)
        #     hard_idx  = hard_neg.nonzero(as_tuple=True)[0]

        #     # 3) supplement with top-scoring negatives if needed
        #     neg_idx     = neg_mask.nonzero(as_tuple=True)[0]
        #     target_neg  = pos_idx.numel()          # want 1× as many negatives as positives
        #     # take from hard first, then from top-scoring
        #     neg_sel = hard_idx
        #     idx_rest = torch.tensor([])
        #     if neg_sel.numel() < target_neg:
        #         remaining = target_neg - neg_sel.numel()
        #         # exclude already chosen
        #         rest = neg_idx[~torch.isin(neg_idx, neg_sel)]
        #         # pick highest-prob among the rest
        #         probs_rest, idx_rest = edge_probs[rest].topk(min(remaining, rest.numel()), largest=True)
        #         neg_sel = torch.cat([neg_sel, rest[idx_rest]])
        #     # if too many hard, just truncate
        #     if neg_sel.numel() > target_neg:
        #         neg_sel = neg_sel[torch.randperm(neg_sel.numel(), device=neg_sel.device)[:target_neg]]

        #     # 4) combine positives + sampled negatives
        #     keep_edges = torch.cat([pos_idx, neg_sel])

        # # guard & cap as before…
        # if keep_edges.numel() == 0:
        #     keep_edges = torch.randperm(len(edge_probs), device=edge_probs.device)[:2048]
        
                
        # # 2) positives (cap if needed)
        # pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
        # if pos_idx.numel() > max_pos:
        #     pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)[:max_pos]]

        # # 3) negatives: hard window + random
        # neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
        # low, high = best_thr - margin, best_thr + margin
        # hard_idx = neg_idx[(edge_probs[neg_idx] >= low) & (edge_probs[neg_idx] <= high)]

        # n_pos  = pos_idx.numel()
        # n_hard = min(len(hard_idx), int(0.5 * n_pos))
        # n_rand = n_pos - n_hard

        # hard_sel = hard_idx[:n_hard]
        # rand_sel = neg_idx[torch.randperm(len(neg_idx), device=neg_idx.device)[:n_rand]]

        # keep = torch.cat([pos_idx, hard_sel, rand_sel])
        # if keep.numel() > 2048:
        #     keep = keep[torch.randperm(keep.numel(), device=keep.device)[:2048]]


        # edge_loss = self.edge_criterion(
        #     edge_logits.flatten()[keep_edges],
        #     full_edge_labels.flatten()[keep_edges]
        # )


        # edge_loss = self.edge_criterion(edge_logits.flatten()[keep_edges],
        #                                 full_edge_labels.flatten()[keep_edges])
        
        
        # edge_loss = self.edge_criterion(edge_logits,
        #                         full_edge_labels)

        # node_loss = self.node_criterion(node_logits, node_labels)
        
        # loss      = 0.5 * node_loss + 0.5 * edge_loss
        
        
         # 1) Compute probabilities
        edge_labels_flat  = batch.full_edge_labels.view(-1,1).float()

        node_probs = node_logits.sigmoid().detach().cpu().numpy().ravel()
        edge_probs = edge_logits.sigmoid().detach().cpu().numpy().ravel()
        node_trues = node_labels.detach().cpu().numpy().ravel()
        edge_trues = edge_labels_flat.detach().cpu().numpy().ravel()

        # 2) Only compute if both classes present
        if len(np.unique(node_trues)) > 1:
            node_roc  = roc_auc_score(node_trues, node_probs)
            node_pr   = average_precision_score(node_trues, node_probs)
        else:
            node_roc, node_pr = 0.0, 0.0

        if len(np.unique(edge_trues)) > 1:
            edge_roc  = roc_auc_score(edge_trues, edge_probs)
            edge_pr   = average_precision_score(edge_trues, edge_probs)
        else:
            edge_roc, edge_pr = 0.0, 0.0

        # 3) Log them to TensorBoard / progress bar
        self.log_dict({
            "train_node_roc_auc":  node_roc,
            "train_node_pr_auc":   node_pr,
            "train_edge_roc_auc":  edge_roc,
            "train_edge_pr_auc":   edge_pr,
            "train_loss": loss,
            
        }, on_step=True, on_epoch=False, prog_bar=True)

        # self.log("num_hard",   hard_idx.numel(),   on_step=True, prog_bar=False)
        # self.log("num_topk",   idx_rest.numel(),   on_step=True, prog_bar=False)
        # self.log("num_kept",   keep_edges.numel(), on_step=True, prog_bar=True)

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


        # node_loss = self.node_criterion(node_logits, node_labels)
        # edge_loss = self.edge_criterion(edge_logits, full_edge_labels)
        
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


        # ---- NEW: PR‐AUC / average precision ----
            # Option A: direct from sklearn
        node_pr_auc = average_precision_score(all_node_labels, all_node_preds) \
                    if len(np.unique(all_node_labels))>1 else 0.0
        edge_pr_auc = average_precision_score(all_edge_labels, all_edge_preds) \
                    if len(np.unique(all_edge_labels))>1 else 0.0
        combined_auc = 0.5 * (node_auc + edge_auc)
        
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
        
        # node_loss = self.node_criterion(node_logits, node_labels)
        # edge_loss = self.edge_criterion(edge_logits, full_edge_labels)

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
        edge_binary = (all_edge_preds >= 0.95).astype(int)

        tn_n, fp_n, fn_n, tp_n = confusion_matrix(all_node_labels, node_binary, labels=[0,1]).ravel()
        tn_e, fp_e, fn_e, tp_e = confusion_matrix(all_edge_labels, edge_binary, labels=[0,1]).ravel()
        
               # ---- NEW: PR‐AUC / average precision ----
            # Option A: direct from sklearn
        node_pr_auc = average_precision_score(all_node_labels, all_node_preds) \
                    if len(np.unique(all_node_labels))>1 else 0.0
        edge_pr_auc = average_precision_score(all_edge_labels, all_edge_preds) \
                    if len(np.unique(all_edge_labels))>1 else 0.0
        
            # ---- NEW: F1‐score (threshold‐dependent) ----
        node_f1 = f1_score(all_node_labels, node_binary) if len(np.unique(all_node_labels))>1 else 0.0
        edge_f1 = f1_score(all_edge_labels, edge_binary) if len(np.unique(all_edge_labels))>1 else 0.0


        # ------------------------------------------------
        # 2.  NODE-wise threshold sweep with sklearn
        # ------------------------------------------------
        p_n, r_n, t_n = precision_recall_curve(all_node_labels, all_node_preds)
        # precision_recall_curve returns an extra point at t = inf; drop it
        f1_n          = 2 * p_n[:-1] * r_n[:-1] / (p_n[:-1] + r_n[:-1] + 1e-12)
        best_idx_n    = np.argmax(f1_n)
        best_t_n      = float(t_n[best_idx_n])      # best threshold
        best_f1_n     = float(f1_n[best_idx_n])

        # ------------------------------------------------
        # 3.  EDGE-wise sweep (identical pattern)
        # ------------------------------------------------
        p_e, r_e, t_e = precision_recall_curve(all_edge_labels, all_edge_preds)
        f1_e          = 2 * p_e[:-1] * r_e[:-1] / (p_e[:-1] + r_e[:-1] + 1e-12)
        best_idx_e    = np.argmax(f1_e)
        best_t_e      = float(t_e[best_idx_e])
        best_f1_e     = float(f1_e[best_idx_e])

        # 1.  Build PR data
        # ------------------------------------------------------------------
        p_n, r_n, _ = precision_recall_curve(all_node_labels,  all_node_preds)
        ap_n        = average_precision_score(all_node_labels, all_node_preds)

        p_e, r_e, _ = precision_recall_curve(all_edge_labels,  all_edge_preds)
        ap_e        = average_precision_score(all_edge_labels, all_edge_preds)

        # ------------------------------------------------------------------
        # 2.  Plot
        # ------------------------------------------------------------------
        plt.figure(figsize=(5, 5))
        plt.plot(r_n, p_n, label=f"Node – AP={ap_n:.3f}")
        plt.plot(r_e, p_e, label=f"Edge – AP={ap_e:.3f}", linestyle="--")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)
        plt.title("Precision–Recall curve (test set)")
        plt.grid(ls=":")
        plt.legend(loc="lower left")
        plt.tight_layout()

        plt.show()        # <— pops up the window / renders inline

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
            'test_edge_pr_auc_epoch': edge_pr_auc,
            'test_node_pr_auc_epoch': node_pr_auc,
            'test_node_f1_epoch':     node_f1,
            'test_edge_f1_epoch':     edge_f1,
            "test_node_best_thr": best_t_n,
            "test_node_best_f1":  best_f1_n,
            "test_edge_best_thr": best_t_e,
            "test_edge_best_f1":  best_f1_e,

            # 'tn_n': tn_n,
            # 'tp_n': tp_n,
            # 'fp_n': fp_n,
            # 'fn_n': fn_n,
            # 'tn_e': tn_e,
            # 'tp_e': tp_e,
            # 'fp_e': fp_e,
            # 'fn_e': fn_e,
        }, on_epoch=True, prog_bar=False)


        self.test_step_outputs.clear()
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=1e-4)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200),
            'interval': 'epoch'
        }
        return [opt], [scheduler]





class AttentionEdgeSampleLinearAverage(pl.LightningModule):
    def __init__(self,
        # Model HParams
        in_channels_DeepLSD: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        geom_channels: int,
        roi_align_embedding_shape: tuple,
        num_layers: int,
        # node_pos_weight,
        # edge_pos_weight,
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
        
        
        node_height, node_width =  self.hparams.roi_align_embedding_shape
        edge_height, edge_width =  self.hparams.edge_sample_size
        # self.edge_patch_enc = nn.Sequential(
        #     nn.Flatten(start_dim=1),
        #     nn.Linear(3* edge_height * edge_width, self.hparams.edge_downsample_dim),
        #     nn.LayerNorm(self.hparams.edge_downsample_dim),         # ← swapped out

        #     nn.Dropout(p=self.hparams.mlp_dropout),
        #     nn.ReLU(),
        # )
        
        # self.edge_patch_enc = nn.Sequential(
        #     nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=2),  # → (16, 32, 16)
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),                 # → (16, 16, 8)

        #     nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), # → (32, 16, 8)
        #     nn.ReLU(),
        #     nn.AdaptiveAvgPool2d((1,1)),                            # → (32, 1, 1)

        #     nn.Flatten(start_dim=1),   # → shape (E, 32)
        #     nn.Linear(32, edge_downsample_dim), # → shape (E, D_edge)
        #     nn.ReLU(),
        # )
        num_groups = 4
        self.edge_patch_enc = nn.Sequential(
            # conv block 1
            nn.Conv2d(3, 16, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(num_groups, 16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.hparams.mlp_dropout),               # spatial dropout
            
            # # conv block 2
            # nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            # nn.GroupNorm(num_groups, 64),
            # nn.ReLU(inplace=True),
            # nn.Dropout2d(p=self.hparams.mlp_dropout),
           
            # global pooling  
            nn.AdaptiveAvgPool2d((1, 1)),         # → (64, 1, 1)
            nn.Flatten(start_dim=1),              # → (E, 64)
            
            # final projection
            nn.Dropout(p=self.hparams.mlp_dropout),
            nn.Linear(16, self.hparams.edge_downsample_dim, bias=False),
            nn.ReLU(inplace=True),
        )
        
        self.node_linear = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups, 32),       # ← group‐norm instead of batch‐norm
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.hparams.mlp_dropout),

            # # Block 2
            # nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            # nn.GroupNorm(num_groups, 64),
            # nn.ReLU(inplace=True),
            # nn.Dropout2d(p=self.hparams.mlp_dropout),

            # collapse spatial dims
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),

            # final MLP head
            nn.Dropout(p=self.hparams.mlp_dropout),
            nn.Linear(32, self.hparams.in_channels)
        )
        # self.node_linear = nn.Sequential(
        #     nn.Flatten(start_dim=1),
        #     nn.Linear(3*node_height*node_width, self.hparams.hidden_channels),
        #     nn.LayerNorm(self.hparams.hidden_channels),         # ← swapped out

        #     nn.Dropout(p=self.hparams.mlp_dropout),
        #     nn.ReLU(),
        #     nn.Dropout(p=self.hparams.mlp_dropout),   # drop again before output

        # )
        
        
       


        self.node_fuse = nn.Sequential(
            nn.Linear(self.hparams.in_channels + self.hparams.in_channels_DeepLSD +  self.hparams.geom_channels,
                      self.hparams.out_channels),
            nn.LayerNorm(self.hparams.out_channels),         # ← swapped out

            nn.ReLU(),
            nn.Linear(self.hparams.out_channels,
                      self.hparams.out_channels),
            nn.Dropout(p=self.hparams.mlp_dropout),   # drop again before output

        )

        # GNN layers
        layers = []
        for i in range(self.hparams.num_layers):
            if i % 2 == 0:
                # layers.append(SelfAttnLayer(self.hparams.out_channels, self.hparams.skip_init, dropout=self.hparams.mlp_dropout))
                layers.append(
                    GATv2Conv(
                        in_channels  = self.hparams.out_channels,
                        out_channels = self.hparams.out_channels,
                        heads        = 2,
                        concat       = False,           # keep dimensionality = out_channels
                        dropout      = self.hparams.mlp_dropout,
                        edge_dim     = None             # or set to edge‐feature dim if you want to pass edge_attr
                    )
                )
            else:
                # layers.append(EdgeSamplerLayer(node_dim=self.hparams.out_channels,
                # edge_attr_dim=self.hparams.edge_downsample_dim,
                # hidden_dim=self.hparams.out_channels, dropout=self.hparams.mlp_dropout))
                
                layers.append(
                    GATv2Conv(
                        in_channels  = self.hparams.out_channels,
                        out_channels = self.hparams.out_channels,
                        heads        = 4,
                        concat       = False,           # keep dimensionality = out_channels
                        dropout      = self.hparams.mlp_dropout,
                        edge_dim     = self.hparams.edge_downsample_dim             # or set to edge‐feature dim if you want to pass edge_attr
                    )
                )
                
                

                
                
                # layers.append(LocalEdgeLayer(self.hparams.out_channels))

        self.layers = nn.ModuleList(layers)

        # Node prediction head
        self.mlp_textural_structural = nn.Sequential(
            nn.Linear(self.hparams.out_channels, self.hparams.out_channels),
            nn.LayerNorm(self.hparams.out_channels),         # ← swapped out

            nn.ReLU(),
            nn.Linear(self.hparams.out_channels, 1)
        )

        self.edge_loss_w = edge_loss_w
        
        self.geom_mlp_dim = 16
        self.edge_geom_mlp = nn.Sequential(
            nn.Linear(self.hparams.geom_channels, self.geom_mlp_dim, bias=False),
            nn.LayerNorm(self.geom_mlp_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.hparams.mlp_dropout),
            nn.Linear(self.geom_mlp_dim,self.geom_mlp_dim)
        )
        
        # assume `D` = self.hparams.out_channels, `D_geo` = self.hparams.geom_channels
        total_dim = 2 * self.hparams.out_channels + self.geom_mlp_dim
        self.edge_norm = nn.LayerNorm(total_dim)

        self.edge_predictor = nn.Sequential(
            nn.Linear(total_dim, self.hparams.out_channels),
            nn.LayerNorm(self.hparams.out_channels),         # ← swapped out

            nn.ReLU(),
            nn.Linear(self.hparams.out_channels, 1)
        )
        

        # # Loss
        self.criterion = nn.BCEWithLogitsLoss()
        
        # self.node_criterion = nn.BCEWithLogitsLoss(
        #     pos_weight=torch.tensor(self.hparams.node_pos_weight, dtype=torch.float)
        # )
        # self.edge_criterion = SmoothedAsymmetricFocalLoss(
        #     gamma_neg=2.0,   # same hard-negative focus as before
        #     gamma_pos=0.0,   # keep positives recall-friendly
        #     eps=0.05)        # assume ≈5 % label noise

        # self.edge_pos_weight = 2.17
        # # self.edge_pos_weight = 5.67

        # self.edge_criterion = nn.BCEWithLogitsLoss(
        #     pos_weight=torch.tensor(self.edge_pos_weight, dtype=torch.float)
        # )

        # Containers for metrics
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
      
       
                    
    def forward(self, batch):
        
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        roi_features = batch.roi_features 
        geo = batch.geo
        # ROI conv and fuse
        roi_feats = self.node_linear(roi_features)
        concat_feat = torch.cat([roi_feats, x, geo], dim=1)
        N = concat_feat.size(0)   # number of nodes

        
        
        if self.training and self.hparams.mlp_dropout > 0:
            mask = (torch.rand(N) > self.hparams.mlp_dropout).float().unsqueeze(1)
            concat_feat = concat_feat * mask

        concat_feat = self.node_fuse(concat_feat)
        
        # Prepare for global attention
        # roi_dense, mask = to_dense_batch(concat_feat, batch_idx)
        # desc = roi_dense.transpose(1, 2)[mask]
        
        
     
        
        edge_patch = self.edge_patch_enc(batch.edge_attr)  

        
        src, dst = batch.full_edge_index
        edge_geo = edge_geometry(geo, src, dst)             # [E,5]
        #local_edge_geo = edge_geo[batch.flat_idx_local] 
        
        # Alternate layers
        
        # layer_node_features = []
        # for layer in self.layers:
        #     if isinstance(layer, SelfAttnLayer):
        #         desc = layer(desc)

        #     else:
        #         flat = desc.transpose(1,2)[mask]
        #         # flat_norm = layer.ln(flat)   
                
        #         if self.training:
        #             edge_index_dropped, edge_mask= dropout_edge(
        #                 edge_index,       # [2, E]
        #                 p=self.hparams.mlp_dropout,            # drop probability; tune to 0.3–0.5 if needed
        #             )
        #             edge_patch_dropped = edge_patch[edge_mask]
        #         else:
        #         # Keep all edges in eval
        #             edge_index_dropped, edge_patch_dropped = edge_index, edge_patch

        #         delta = layer(flat, edge_index_dropped, edge_patch_dropped)

        #         delta_dense, _ = to_dense_batch(delta, batch_idx)
        #         desc = desc + delta_dense.transpose(1,2)
            
        #     desc = F.dropout(desc, p=self.hparams.mlp_dropout, training=self.training)

        #     cur_feat = desc.transpose(1,2)[mask]      # [total_nodes, out_channels]
        #     layer_node_features.append(cur_feat)
                

        layer_node_features = []
        x = concat_feat  # start from your flat node features

        for i, conv in enumerate(self.layers):
            if i % 2 == 0:
                # local
                if self.training:
                    edge_index_dropped, edge_mask= dropout_edge(
                        edge_index,       # [2, E]
                        p=self.hparams.mlp_dropout,            # drop probability; tune to 0.3–0.5 if needed
                    )
                else:
                # Keep all edges in eval
                    edge_index_dropped = edge_index
                x = conv(x, edge_index_dropped)
            else:
                # global, with edge features
                if self.training:
                    edge_index_dropped, edge_mask= dropout_edge(
                        batch.global_edge_index,       # [2, E]
                        p=self.hparams.mlp_dropout,            # drop probability; tune to 0.3–0.5 if needed
                    )
                    edge_patch_dropped = edge_patch[edge_mask]
                else:
                # Keep all edges in eval
                    edge_index_dropped, edge_patch_dropped = batch.global_edge_index, edge_patch
                    
                x = conv(x, edge_index_dropped, edge_patch_dropped)
            x = F.relu(x)
            x = F.dropout(x, p=self.hparams.mlp_dropout, training=self.training)
            layer_node_features.append(x)

        # stack across layers and average
        features = torch.stack(layer_node_features, dim=0).mean(dim=0)  # [N, D]


        # Collapse to node features
      
        stacked = torch.stack(layer_node_features, dim=0)  # [num_layers, N_total, D]
        features = stacked.mean(dim=0)
        
        #features = F.dropout(features, p=self.hparams.mlp_dropout, training=self.training)

        # Node logits
        node_logits = self.mlp_textural_structural(features)
        


        # Edge logits


        geo_feat = self.edge_geom_mlp(edge_geo) 
        h_src, h_dst = features[src], features[dst]
       
        edge_in = torch.cat([
            0.5 * (h_src + h_dst),        # symmetric mean      [E, D]t
            (h_src - h_dst).abs(),        # symmetric distance  [E, D]
            geo_feat # geometric extras    [E, 5]
        ], dim=1)  
        
        edge_in = self.edge_norm(edge_in)

                
        edge_in = self.edge_predictor(edge_in)  # overwrite so save memory these are now logits



        return node_logits, edge_in


    def training_step(self, batch, batch_idx):
        node_logits, edge_logits = self(batch)
        
        
        node_labels = batch.y.view(-1,1).float()
        
        node_loss = self.criterion(node_logits, node_labels)

        full_edge_labels = batch.full_edge_labels
        ## sample nodes
        # pos_n = (node_labels==1).nonzero(as_tuple=True)[0]
        # neg_n = (node_labels==0).nonzero(as_tuple=True)[0]
        # if pos_n.numel()>0:
        #     perm = torch.randperm(neg_n.size(0))
        #     sampled_neg_n = neg_n[perm[:pos_n.size(0)]]
        #     keep_n = torch.cat([pos_n, sampled_neg_n])
        # else:
        #     k=min(32,neg_n.size(0)); perm=torch.randperm(neg_n.size(0))
        #     keep_n=neg_n[perm[:k]]
        # sampled_node_logits = node_logits[keep_n]
        # sampled_node_labels = node_labels[keep_n]
        # node_loss = self.criterion(sampled_node_logits, sampled_node_labels)
        # # sample edges
        # edge_labels_flat = full_edge_labels.view(-1,1)
        # pos_e = (edge_labels_flat==1).nonzero(as_tuple=True)[0]
        # neg_e = (edge_labels_flat==0).nonzero(as_tuple=True)[0]
        # if pos_e.numel()>0:
        #     perm_e = torch.randperm(neg_e.size(0))
        #     sampled_neg_e = neg_e[perm_e[:pos_e.size(0)]]
        #     keep_e = torch.cat([pos_e, sampled_neg_e])
        # else:
        #     k_e=min(32,neg_e.size(0)); perm_e=torch.randperm(neg_e.size(0))
        #     keep_e=neg_e[perm_e[:k_e]]
        # sampled_edge_logits = edge_logits[keep_e]
        # sampled_edge_labels = edge_labels_flat[keep_e]
        
        
        # edge_loss = self.criterion(sampled_edge_logits, sampled_edge_labels)
        
        

        
        # N = batch.coordinates.shape[0]

 
        # # Reshape flattened edge_logits & labels → (N, N)
        # edge_logits_matrix = edge_logits.view(N, N)
        # edge_labels_matrix = full_edge_labels.view(N, N)  # still a torch.Tensor

        # # 1) Compute row-sums of the ground-truth adjacency (torch)
        # #    (this is equivalent to `edge_labels_array.sum(axis=1)` but in torch)
        # row_sums = edge_labels_matrix.sum(dim=1)  # shape: [N]

        # # 2) Keep only rows i with row_sums[i] > 0
        # keep_mask = row_sums > 0                  # shape: [N] (bool tensor)
        # kept_idx  = torch.nonzero(keep_mask, as_tuple=False).view(-1)  # [num_kept]

        # # 3) Index into the (N×N) matrices using the kept indices
        # #    This is pure‐PyTorch boolean/integer indexing:
        # kept_logits = edge_logits_matrix[kept_idx][:, kept_idx]  # [num_kept, num_kept]
        # kept_labels = edge_labels_matrix[kept_idx][:, kept_idx]  # [num_kept, num_kept]

        # # 4) Flatten back to vectors
        # edge_logits = kept_logits.reshape(-1)  # shape: [num_kept * num_kept]
        # full_edge_labels = kept_labels.reshape(-1)  # shape: [num_kept * num_kept]
        
        #  # 6) compute seg–seg distances and flatten
        # coords = batch.coordinates                      # [N,2,2]
        # p1, p2 = coords[:,0], coords[:,1]               # each [N,2]
        # D = seg_seg_dist(p1, p2, p1, p2)                # [N,N]
        # # if you applied the keep_nodes filtering above, also filter D:
        # D = D[kept_idx][:, kept_idx]
        # D_flat = D.reshape(-1)                          # [M]
        
        # # 7) apply distance cutoff mask
        # cutoff = 200
        # dist_mask = D_flat <= cutoff             # boolean mask [M]
        # idx = dist_mask.nonzero().view(-1)
        
        # # 8) index into logits & labels
        # sel_logits = edge_logits[idx]
        # sel_labels = full_edge_labels[idx]
        src, dst = batch.full_edge_index               # each [N*N]
        labels = batch.full_edge_labels.view(-1)       # [N*N]
        logits = edge_logits.view(-1)                  # [N*N]

        # 0) Remove all self‐pairs i==j
        nonself = (src != dst)
        src = src[nonself]
        dst = dst[nonself]
        labels = labels[nonself]
        logits = logits[nonself]
            
        # Now reshape into matrix form over the remaining indices if you want:
        N = batch.coordinates.size(0)
        # … or just continue treating them as flat…

        # # 1) (Optional) Filter out rows with no positives
        # #    First, find which nodes have at least one positive edge:
        # #    build a mask per node i: has_pos[i] = any(labels[src==i] == 1)
        # has_pos = torch.zeros(N, dtype=torch.bool, device=src.device)
        # pos_edges = (labels == 1).nonzero(as_tuple=True)[0]
        # has_pos[src[pos_edges]] = True
        # keep_nodes = has_pos.nonzero(as_tuple=True)[0]

        # # 2) Build a node‐mask over your flat lists:
        # node_mask = torch.isin(src, keep_nodes) & torch.isin(dst, keep_nodes)
        # src = src[node_mask]
        # dst = dst[node_mask]
        # labels = labels[node_mask]
        # logits = logits[node_mask]

        # 3) (Optional) Distance cutoff
        #    Compute your seg–seg distances for only the kept src/dst pairs:
        p1, p2 = batch.coordinates[:, 0], batch.coordinates[:, 1]      
        D = seg_seg_dist(p1, p2, p1, p2)       # (N, N)
        D_pairs = D[src, dst]                  # flatten to [#edges]
        cutoff = 200
        dist_mask = D_pairs <= cutoff
        labels = labels[dist_mask]
        logits = logits[dist_mask]
        
        
         # 7) Balance positives vs negatives by down‐sampling the larger class
        pos_idx = torch.nonzero(labels == 1.0, as_tuple=True)[0]
        neg_idx = torch.nonzero(labels == 0.0, as_tuple=True)[0]
        n_pos, n_neg = pos_idx.numel(), neg_idx.numel()

        if n_pos > 0 and n_neg > 0:
            k = min(n_pos, n_neg)
            sel_pos = pos_idx[torch.randperm(n_pos, device=pos_idx.device)[:k]]
            sel_neg = neg_idx[torch.randperm(n_neg, device=neg_idx.device)[:k]]
            sel_idx = torch.cat([sel_pos, sel_neg], dim=0)
        elif n_pos > 0:
            k = min(n_pos, 64)
            sel_idx = pos_idx[torch.randperm(n_pos, device=pos_idx.device)[:k]]
        elif n_neg > 0:
            k = min(n_neg, 64)
            sel_idx = neg_idx[torch.randperm(n_neg, device=neg_idx.device)[:k]]
        else:
            # no edges left after filtering—skip edge loss
            edge_loss = torch.tensor(0.0, device=node_logits.device)
            loss = 0.5 * (node_loss + edge_loss)
            self.log('train_loss', loss, on_step=True, prog_bar=True, on_epoch=False)
            return loss

        sampled_logits = logits[sel_idx]
        sampled_labels = labels[sel_idx]

    
        
        
        
        


        # in your training_step, before edge_loss
        eps = 0.05
        edge_labels = sampled_labels.clone()
        edge_labels = edge_labels * (1.0 - 2*eps) + eps  # 0→eps, 1→1-eps

        
        # node_loss = self.node_criterion(node_logits, node_labels)
        edge_loss = self.criterion(sampled_logits, edge_labels)

       
        loss = (node_loss + edge_loss) * 0.5

        # loss = self.hparams.node_loss_w*node_loss + self.hparams.edge_loss_w*edge_loss
        
        # # # metrics
        # # with torch.no_grad():
        # #     n_probs=torch.sigmoid(sampled_node_logits); n_preds=(n_probs>=self.hparams.threshold_structural).int().detach().cpu().numpy().ravel()
        # #     node_acc=accuracy_score(sampled_node_labels.int().detach().cpu().numpy().ravel(),n_preds)
        # #     e_probs=torch.sigmoid(sampled_edge_logits); e_preds=(e_probs>=self.hparams.threshold_structural).int().detach().cpu().numpy().ravel()
        # #     edge_acc=accuracy_score(sampled_edge_labels.int().detach().cpu().numpy().ravel(), e_preds)
            
            
        # self.log('train_loss',loss,on_step=True,on_epoch=False, prog_bar=True, logger=True)
        # # self.log('train_node_acc',node_acc,on_step=True,on_epoch=False, prog_bar=True, logger=True)
        # # self.log('train_edge_acc',edge_acc,on_step=True,on_epoch=False, prog_bar=True, logger=True)
        
        
        # return loss
        
        # with torch.no_grad():
            # edge_probs = edge_logits.sigmoid().flatten()
            # labels     = full_edge_labels.flatten()

        # 1) collect positive indices (and subsample if too many)
        # pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
        # max_pos = 1024                        # cap positives to 1024
        # if pos_idx.numel() > max_pos:
        #     pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)[:max_pos]]

        # # 2) build negative pool around your boundary
        # neg_mask  = (labels == 0)
        # best_thr  = 0.85; margin = 0.10
        #     low, high = best_thr - margin, best_thr + margin
        #     hard_neg  = neg_mask & (edge_probs >= low) & (edge_probs <= high)
        #     hard_idx  = hard_neg.nonzero(as_tuple=True)[0]

        #     # 3) supplement with top-scoring negatives if needed
        #     neg_idx     = neg_mask.nonzero(as_tuple=True)[0]
        #     target_neg  = pos_idx.numel()          # want 1× as many negatives as positives
        #     # take from hard first, then from top-scoring
        #     neg_sel = hard_idx
        #     idx_rest = torch.tensor([])
        #     if neg_sel.numel() < target_neg:
        #         remaining = target_neg - neg_sel.numel()
        #         # exclude already chosen
        #         rest = neg_idx[~torch.isin(neg_idx, neg_sel)]
        #         # pick highest-prob among the rest
        #         probs_rest, idx_rest = edge_probs[rest].topk(min(remaining, rest.numel()), largest=True)
        #         neg_sel = torch.cat([neg_sel, rest[idx_rest]])
        #     # if too many hard, just truncate
        #     if neg_sel.numel() > target_neg:
        #         neg_sel = neg_sel[torch.randperm(neg_sel.numel(), device=neg_sel.device)[:target_neg]]

        #     # 4) combine positives + sampled negatives
        #     keep_edges = torch.cat([pos_idx, neg_sel])

        # # guard & cap as before…
        # if keep_edges.numel() == 0:
        #     keep_edges = torch.randperm(len(edge_probs), device=edge_probs.device)[:2048]
        
                
        # # 2) positives (cap if needed)
        # pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
        # if pos_idx.numel() > max_pos:
        #     pos_idx = pos_idx[torch.randperm(pos_idx.numel(), device=pos_idx.device)[:max_pos]]

        # # 3) negatives: hard window + random
        # neg_idx = (labels == 0).nonzero(as_tuple=True)[0]
        # low, high = best_thr - margin, best_thr + margin
        # hard_idx = neg_idx[(edge_probs[neg_idx] >= low) & (edge_probs[neg_idx] <= high)]

        # n_pos  = pos_idx.numel()
        # n_hard = min(len(hard_idx), int(0.5 * n_pos))
        # n_rand = n_pos - n_hard

        # hard_sel = hard_idx[:n_hard]
        # rand_sel = neg_idx[torch.randperm(len(neg_idx), device=neg_idx.device)[:n_rand]]

        # keep = torch.cat([pos_idx, hard_sel, rand_sel])
        # if keep.numel() > 2048:
        #     keep = keep[torch.randperm(keep.numel(), device=keep.device)[:2048]]


        # edge_loss = self.edge_criterion(
        #     edge_logits.flatten()[keep_edges],
        #     full_edge_labels.flatten()[keep_edges]
        # )


        # edge_loss = self.edge_criterion(edge_logits.flatten()[keep_edges],
        #                                 full_edge_labels.flatten()[keep_edges])
        
        
        # edge_loss = self.edge_criterion(edge_logits,
        #                         full_edge_labels)

        # node_loss = self.node_criterion(node_logits, node_labels)
        
        # loss      = 0.5 * node_loss + 0.5 * edge_loss
        
        
        
        
        #  # 1) Compute probabilities
        # edge_labels_flat  = full_edge_labels.view(-1,1).float()

        # node_probs = node_logits.sigmoid().detach().cpu().numpy().ravel()
        # edge_probs = edge_logits.sigmoid().detach().cpu().numpy().ravel()
        # node_trues = node_labels.detach().cpu().numpy().ravel()
        # edge_trues = edge_labels_flat.detach().cpu().numpy().ravel()

        # # 2) Only compute if both classes present
        # if len(np.unique(node_trues)) > 1:
        #     node_roc  = roc_auc_score(node_trues, node_probs)
        #     node_pr   = average_precision_score(node_trues, node_probs)
        # else:
        #     node_roc, node_pr = 0.0, 0.0

        # if len(np.unique(edge_trues)) > 1:
        #     edge_roc  = roc_auc_score(edge_trues, edge_probs)
        #     edge_pr   = average_precision_score(edge_trues, edge_probs)
        # else:
        #     edge_roc, edge_pr = 0.0, 0.0

        # 3) Log them to TensorBoard / progress bar
        self.log_dict({
            # "train_node_roc_auc":  node_roc,
            # "train_node_pr_auc":   node_pr,
            # "train_edge_roc_auc":  edge_roc,
            # "train_edge_pr_auc":   edge_pr,
            "train_loss": loss,
            "node_loss": node_loss,
            "edge_loss": edge_loss
            
        }, on_step=True, on_epoch=False, prog_bar=True)

        # self.log("num_hard",   hard_idx.numel(),   on_step=True, prog_bar=False)
        # self.log("num_topk",   idx_rest.numel(),   on_step=True, prog_bar=False)
        # self.log("num_kept",   keep_edges.numel(), on_step=True, prog_bar=True)

        return loss

       

    def validation_step(self, batch: Batch, batch_idx: int):
        # forward
        node_logits, edge_logits = self(batch)
        # labels
        node_labels      = batch.y.float()               # [N,1]
        full_edge_labels = batch.full_edge_labels.float()  # [N*N,1]

        # losses
        
        # N = batch.coordinates.shape[0]

        
        # # Reshape flattened edge_logits & labels → (N, N)
        # edge_logits_matrix = edge_logits.view(N, N)
        # edge_labels_matrix = full_edge_labels.view(N, N)  # still a torch.Tensor

        # # 1) Compute row-sums of the ground-truth adjacency (torch)
        # #    (this is equivalent to `edge_labels_array.sum(axis=1)` but in torch)
        # row_sums = edge_labels_matrix.sum(dim=1)  # shape: [N]

        # # 2) Keep only rows i with row_sums[i] > 0
        # keep_mask = row_sums > 0                  # shape: [N] (bool tensor)
        # kept_idx  = torch.nonzero(keep_mask, as_tuple=False).view(-1)  # [num_kept]

        # # 3) Index into the (N×N) matrices using the kept indices
        # #    This is pure‐PyTorch boolean/integer indexing:
        # kept_logits = edge_logits_matrix[kept_idx][:, kept_idx]  # [num_kept, num_kept]
        # kept_labels = edge_labels_matrix[kept_idx][:, kept_idx]  # [num_kept, num_kept]

        # # 4) Flatten back to vectors
        # edge_logits = kept_logits.reshape(-1)  # shape: [num_kept * num_kept]
        # full_edge_labels = kept_labels.reshape(-1)  # shape: [num_kept * num_kept]
        
        node_loss = self.criterion(node_logits, node_labels)
        # edge_loss = self.criterion(edge_logits, full_edge_labels)


        # node_loss = self.node_criterion(node_logits, node_labels)
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


        # ---- NEW: PR‐AUC / average precision ----
            # Option A: direct from sklearn
        node_pr_auc = average_precision_score(all_node_labels, all_node_preds) \
                    if len(np.unique(all_node_labels))>1 else 0.0
        edge_pr_auc = average_precision_score(all_edge_labels, all_edge_preds) \
                    if len(np.unique(all_edge_labels))>1 else 0.0
        combined_auc = 0.5 * (node_auc + edge_auc)
        
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
        
        # N = batch.coordinates.shape[0]

      
        # # Reshape flattened edge_logits & labels → (N, N)
        # edge_logits_matrix = edge_logits.view(N, N)
        # edge_labels_matrix = full_edge_labels.view(N, N)  # still a torch.Tensor

        # # 1) Compute row-sums of the ground-truth adjacency (torch)
        # #    (this is equivalent to `edge_labels_array.sum(axis=1)` but in torch)
        # row_sums = edge_labels_matrix.sum(dim=1)  # shape: [N]

        # # 2) Keep only rows i with row_sums[i] > 0
        # keep_mask = row_sums > 0                  # shape: [N] (bool tensor)
        # kept_idx  = torch.nonzero(keep_mask, as_tuple=False).view(-1)  # [num_kept]

        # # 3) Index into the (N×N) matrices using the kept indices
        # #    This is pure‐PyTorch boolean/integer indexing:
        # kept_logits = edge_logits_matrix[kept_idx][:, kept_idx]  # [num_kept, num_kept]
        # kept_labels = edge_labels_matrix[kept_idx][:, kept_idx]  # [num_kept, num_kept]

        # # 4) Flatten back to vectors
        # edge_logits = kept_logits.reshape(-1)  # shape: [num_kept * num_kept]
        # full_edge_labels = kept_labels.reshape(-1)  # shape: [num_kept * num_kept]
        
        node_loss = self.criterion(node_logits, node_labels)
        # edge_loss = self.criterion(edge_logits, full_edge_labels)
        
        # node_loss = self.node_criterion(node_logits, node_labels)
        edge_loss = self.criterion(edge_logits, full_edge_labels)

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
        edge_binary = (all_edge_preds >= 0.95).astype(int)

        tn_n, fp_n, fn_n, tp_n = confusion_matrix(all_node_labels, node_binary, labels=[0,1]).ravel()
        tn_e, fp_e, fn_e, tp_e = confusion_matrix(all_edge_labels, edge_binary, labels=[0,1]).ravel()
        
               # ---- NEW: PR‐AUC / average precision ----
            # Option A: direct from sklearn
        node_pr_auc = average_precision_score(all_node_labels, all_node_preds) \
                    if len(np.unique(all_node_labels))>1 else 0.0
        edge_pr_auc = average_precision_score(all_edge_labels, all_edge_preds) \
                    if len(np.unique(all_edge_labels))>1 else 0.0
        
            # ---- NEW: F1‐score (threshold‐dependent) ----
        node_f1 = f1_score(all_node_labels, node_binary) if len(np.unique(all_node_labels))>1 else 0.0
        edge_f1 = f1_score(all_edge_labels, edge_binary) if len(np.unique(all_edge_labels))>1 else 0.0


        # ------------------------------------------------
        # 2.  NODE-wise threshold sweep with sklearn
        # ------------------------------------------------
        p_n, r_n, t_n = precision_recall_curve(all_node_labels, all_node_preds)
        # precision_recall_curve returns an extra point at t = inf; drop it
        f1_n          = 2 * p_n[:-1] * r_n[:-1] / (p_n[:-1] + r_n[:-1] + 1e-12)
        best_idx_n    = np.argmax(f1_n)
        best_t_n      = float(t_n[best_idx_n])      # best threshold
        best_f1_n     = float(f1_n[best_idx_n])

        # ------------------------------------------------
        # 3.  EDGE-wise sweep (identical pattern)
        # ------------------------------------------------
        p_e, r_e, t_e = precision_recall_curve(all_edge_labels, all_edge_preds)
        f1_e          = 2 * p_e[:-1] * r_e[:-1] / (p_e[:-1] + r_e[:-1] + 1e-12)
        best_idx_e    = np.argmax(f1_e)
        best_t_e      = float(t_e[best_idx_e])
        best_f1_e     = float(f1_e[best_idx_e])

        # 1.  Build PR data
        # ------------------------------------------------------------------
        p_n, r_n, _ = precision_recall_curve(all_node_labels,  all_node_preds)
        ap_n        = average_precision_score(all_node_labels, all_node_preds)

        p_e, r_e, _ = precision_recall_curve(all_edge_labels,  all_edge_preds)
        ap_e        = average_precision_score(all_edge_labels, all_edge_preds)

        # ------------------------------------------------------------------
        # 2.  Plot
        # ------------------------------------------------------------------
        plt.figure(figsize=(5, 5))
        plt.plot(r_n, p_n, label=f"Node – AP={ap_n:.3f}")
        plt.plot(r_e, p_e, label=f"Edge – AP={ap_e:.3f}", linestyle="--")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.xlim(0, 1.0)
        plt.ylim(0, 1.0)
        plt.title("Precision–Recall curve (test set)")
        plt.grid(ls=":")
        plt.legend(loc="lower left")
        plt.tight_layout()

        plt.show()        # <— pops up the window / renders inline

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
            'test_edge_pr_auc_epoch': edge_pr_auc,
            'test_node_pr_auc_epoch': node_pr_auc,
            'test_node_f1_epoch':     node_f1,
            'test_edge_f1_epoch':     edge_f1,
            "test_node_best_thr": best_t_n,
            "test_node_best_f1":  best_f1_n,
            "test_edge_best_thr": best_t_e,
            "test_edge_best_f1":  best_f1_e,

            # 'tn_n': tn_n,
            # 'tp_n': tp_n,
            # 'fp_n': fp_n,
            # 'fn_n': fn_n,
            # 'tn_e': tn_e,
            # 'tp_e': tp_e,
            # 'fp_e': fp_e,
            # 'fn_e': fn_e,
        }, on_epoch=True, prog_bar=False)


        self.test_step_outputs.clear()
    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW(
    #         self.parameters(),
    #         lr=self.hparams.learning_rate,           # base LR
    #         weight_decay=1e-4  # keep strong regularization
    #     )
    #     scheduler = {
    #         'scheduler': torch.optim.lr_scheduler.OneCycleLR(
    #             optimizer,
    #             max_lr=1.5e-3,                       # ~3× base LR
    #             total_steps=self.trainer.estimated_stepping_batches,
    #             pct_start=0.3,                      # 30% of steps warming up
    #             anneal_strategy='cos',
    #             div_factor=10.0,                    # start LR = max_lr/10 = 1.5e-4
    #             final_div_factor=1e4                # end LR = max_lr/1e4 = 1.5e-7
    #         ),
    #         'interval': 'step'
    #     }
    #     return [optimizer], [scheduler]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,   # e.g. 1e-4
            weight_decay=1e-4
        )
        # Cosine anneal *over epochs* so LR slowly decays from lr → η_min
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.max_epochs,  # span full training run
                eta_min=1e-6                    # floor LR at 1e-6
            ),
            "interval": "epoch",
        }
        return [optimizer], [scheduler]
