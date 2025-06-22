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
# from dataset_inductive import seg_seg_dist
from lightning_tools.dataset_inductive import seg_seg_dist


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



from sklearn.metrics import precision_recall_curve, auc, average_precision_score, f1_score
from torch_geometric.nn import GATv2Conv



class GNN(pl.LightningModule):
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
   
        num_groups = 4
        self.edge_patch_enc = nn.Sequential(
            # conv block 1
            nn.Conv2d(3, 16, kernel_size=5, padding=2, bias=False),
            nn.GroupNorm(num_groups, 16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=self.hparams.mlp_dropout),               # spatial dropout
            

           
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

  

            # collapse spatial dims
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),

            # final MLP head
            nn.Dropout(p=self.hparams.mlp_dropout),
            nn.Linear(32, self.hparams.in_channels)
        )
 
        
       


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
                layers.append(
                    GATv2Conv(
                        in_channels  = self.hparams.out_channels,
                        out_channels = self.hparams.out_channels,
                        heads        = 2,
                        concat       = False,        
                        dropout      = self.hparams.mlp_dropout,
                        edge_dim     = None          
                    )
                )
            else:
 
                
                layers.append(
                    GATv2Conv(
                        in_channels  = self.hparams.out_channels,
                        out_channels = self.hparams.out_channels,
                        heads        = 4,
                        concat       = False,           # keep dimensionality = out_channels
                        dropout      = self.hparams.mlp_dropout,
                        edge_dim     = self.hparams.edge_downsample_dim             
                    )
                )
                
                

                
                

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
   
     
        
        edge_patch = self.edge_patch_enc(batch.edge_attr)  

        
        src, dst = batch.full_edge_index
        edge_geo = edge_geometry(geo, src, dst)             # [E,5]
     

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

    
        eps = 0.05
        edge_labels = sampled_labels.clone()
        edge_labels = edge_labels * (1.0 - 2*eps) + eps  # 0→eps, 1→1-eps

        
        # node_loss = self.node_criterion(node_logits, node_labels)
        edge_loss = self.criterion(sampled_logits, edge_labels)

       
        loss = (node_loss + edge_loss) * 0.5

    

        self.log_dict({
      
            "train_loss": loss,
            "node_loss": node_loss,
            "edge_loss": edge_loss
            
        }, on_step=True, on_epoch=False, prog_bar=True)



        return loss

       

    def validation_step(self, batch: Batch, batch_idx: int):
        # forward
        node_logits, edge_logits = self(batch)
        # labels
        node_labels      = batch.y.float()               # [N,1]
        full_edge_labels = batch.full_edge_labels.float()  # [N*N,1]

      
        
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

        }, on_epoch=True, prog_bar=False)


        self.test_step_outputs.clear()
   

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
