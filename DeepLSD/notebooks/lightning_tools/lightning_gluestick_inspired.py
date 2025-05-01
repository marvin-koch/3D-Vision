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

class AttentionBothCoplanar(pl.LightningModule):
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
        # Training HParams
        learning_rate: float = 1e-3,
        node_loss_w: float = 1.0,          # Weight for node loss
        edge_loss_w: float = 1.0,          # Weight for edge loss (new)
        threshold_structural: float = 0.5,  # Threshold for accuracy/recall calc
        mlp_dropout: float = 0.0,          # drop out for merge features
        skip_init=False
        ):
        super().__init__()
        self.save_hyperparameters()

        # Edge prediction head
        self.edge_loss_w = edge_loss_w
        self.edge_predictor = nn.Sequential(
            nn.Linear(2*self.hparams.out_channels + self.hparams.geom_channels, self.hparams.out_channels),
            nn.ReLU(),
            nn.Linear(self.hparams.out_channels, 1)
        )

        # Convolutional ROI embedding
        self.conv_roi_embedding = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=8, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=8, kernel_size=6, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(start_dim=1)
        )
        with torch.no_grad():
            x = torch.randn(1, 3, *self.hparams.roi_align_embedding_shape)
            self.hparams.channels_conv_roi_embedding = self.conv_roi_embedding(x).shape[1]

        self.node_fuse = nn.Sequential(
            nn.Linear(self.hparams.channels_conv_roi_embedding + self.hparams.in_channels_DeepLSD +  self.hparams.geom_channels,
                      self.hparams.out_channels),
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
                layers.append(LocalEdgeLayer(self.hparams.out_channels))
        self.layers = nn.ModuleList(layers)

        # Node prediction head
        self.mlp_textural_structural = nn.Sequential(
            nn.Linear(self.hparams.out_channels, self.hparams.out_channels),
            nn.ReLU(),
            nn.Linear(self.hparams.out_channels, 1)
        )

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Containers for metrics
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        #Learned weights for loss
        self.log_sigma_node = nn.Parameter(torch.zeros(()))
        self.log_sigma_edge = nn.Parameter(torch.zeros(()))

    def forward(self, batch):
        
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        roi_features = batch.roi_features 
        geo = batch.geo
        # ROI conv and fuse
        roi_feats = self.conv_roi_embedding(roi_features)
        concat_feat = torch.cat([roi_feats, x, geo], dim=1)
        concat_feat = self.node_fuse(concat_feat)

        # Prepare for global attention
        roi_dense, mask = to_dense_batch(concat_feat, batch_idx)
        desc = roi_dense.transpose(1, 2)

        # Alternate layers
        for layer in self.layers:
            if isinstance(layer, SelfAttnLayer):
                desc = layer(desc)
            else:
                flat = desc.transpose(1,2)[mask]
                delta = layer(flat, edge_index)
                delta_dense, _ = to_dense_batch(delta, batch_idx)
                desc = desc + delta_dense.transpose(1,2)

        # Collapse to node features
        features = desc.transpose(1,2)[mask]
        
        features = F.dropout(features, p=self.hparams.mlp_dropout, training=self.training)

        # Node logits
        node_logits = self.mlp_textural_structural(features)

        # Edge logits

        src, dst = batch.full_edge_index
        edge_geo  = edge_geometry(geo, src, dst)             # [E,5]

        h_src, h_dst = features[src], features[dst]
       
        edge_in = torch.cat([
            0.5 * (h_src + h_dst),        # symmetric mean      [E, D]
            (h_src - h_dst).abs(),        # symmetric distance  [E, D]
            edge_geo                      # geometric extras    [E, 5]
        ], dim=1)  
                
        edge_logits = self.edge_predictor(edge_in)

        return node_logits, edge_logits

    def training_step(self, batch, batch_idx):
        node_logits, edge_logits = self(batch)
        node_labels = batch.y.view(-1,1).float()
        full_edge_labels = batch.full_edge_labels
        # sample nodes
        pos_n = (node_labels==1).nonzero(as_tuple=True)[0]
        neg_n = (node_labels==0).nonzero(as_tuple=True)[0]
        if pos_n.numel()>0:
            perm = torch.randperm(neg_n.size(0), device=neg_n.device)
            sampled_neg_n = neg_n[perm[:pos_n.size(0)]]
            keep_n = torch.cat([pos_n, sampled_neg_n])
        else:
            k=min(32,neg_n.size(0)); perm=torch.randperm(neg_n.size(0),device=neg_n.device)
            keep_n=neg_n[perm[:k]]
        sampled_node_logits = node_logits[keep_n]
        sampled_node_labels = node_labels[keep_n]
        node_loss = self.criterion(sampled_node_logits, sampled_node_labels)
        # sample edges
        edge_labels_flat = full_edge_labels.view(-1,1)
        pos_e = (edge_labels_flat==1).nonzero(as_tuple=True)[0]
        neg_e = (edge_labels_flat==0).nonzero(as_tuple=True)[0]
        if pos_e.numel()>0:
            perm_e = torch.randperm(neg_e.size(0),device=neg_e.device)
            sampled_neg_e = neg_e[perm_e[:pos_e.size(0)]]
            keep_e = torch.cat([pos_e, sampled_neg_e])
        else:
            k_e=min(32,neg_e.size(0)); perm_e=torch.randperm(neg_e.size(0),device=neg_e.device)
            keep_e=neg_e[perm_e[:k_e]]
        sampled_edge_logits = edge_logits[keep_e]
        sampled_edge_labels = edge_labels_flat[keep_e]
        edge_loss = self.criterion(sampled_edge_logits, sampled_edge_labels)
        # combined
        
      
        loss = (node_loss * torch.exp(-2*self.log_sigma_node) +
                edge_loss * torch.exp(-2*self.log_sigma_edge) +
                self.log_sigma_node + self.log_sigma_edge) * 0.5

        # loss = self.hparams.node_loss_w*node_loss + self.hparams.edge_loss_w*edge_loss
        
        # metrics
        with torch.no_grad():
            n_probs=torch.sigmoid(sampled_node_logits); n_preds=(n_probs>=self.hparams.threshold_structural).int()
            node_acc=accuracy_score(sampled_node_labels.int().cpu(),n_preds.cpu())
            e_probs=torch.sigmoid(sampled_edge_logits); e_preds=(e_probs>=self.hparams.threshold_structural).int()
            edge_acc=accuracy_score(sampled_edge_labels.int().cpu(), e_preds.cpu())
            
            
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
        total_loss = self.hparams.node_loss_w * node_loss + self.hparams.edge_loss_w * edge_loss

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
        losses = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        all_node_preds = torch.cat([x['node_preds'] for x in self.validation_step_outputs]).cpu().numpy()
        all_node_labels = torch.cat([x['node_labels'] for x in self.validation_step_outputs]).cpu().numpy()
        all_edge_preds = torch.cat([x['edge_preds'] for x in self.validation_step_outputs]).cpu().numpy()
        all_edge_labels = torch.cat([x['edge_labels'] for x in self.validation_step_outputs]).cpu().numpy()

        # Node metrics
        node_binary = (all_node_preds >= self.hparams.threshold_structural).astype(int)
        node_acc = accuracy_score(all_node_labels, node_binary)
        node_recall = recall_score(all_node_labels, node_binary, zero_division=0)
        node_auc = auc(*roc_curve(all_node_labels, all_node_preds)[:2]) if len(np.unique(all_node_labels))>1 else 0.0

        # Edge metrics
        edge_binary = (all_edge_preds >= self.hparams.threshold_structural).astype(int)
        edge_acc = accuracy_score(all_edge_labels, edge_binary)
        edge_recall = recall_score(all_edge_labels, edge_binary, zero_division=0)
        edge_auc = auc(*roc_curve(all_edge_labels, all_edge_preds)[:2]) if len(np.unique(all_edge_labels))>1 else 0.0

        self.log('val_loss_epoch', losses, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_node_acc_epoch', node_acc, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_node_recall_epoch', node_recall, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_node_auc_epoch', node_auc, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_edge_acc_epoch', edge_acc, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_edge_recall_epoch', edge_recall, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_edge_auc_epoch', edge_auc, on_epoch=True, prog_bar=False, logger=True)
        combined_auc = 0.5 * (edge_auc + node_auc)
        self.log('val_combined_auc_epoch', combined_auc, on_epoch=True, prog_bar=False, logger=True)
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
        total_loss = self.hparams.node_loss_w * node_loss + self.hparams.edge_loss_w * edge_loss

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

        node_binary = (all_node_preds >= self.hparams.threshold_structural).astype(int)
        node_acc = accuracy_score(all_node_labels, node_binary)
        node_recall = recall_score(all_node_labels, node_binary, zero_division=0)
        node_auc = auc(*roc_curve(all_node_labels, all_node_preds)[:2]) if len(np.unique(all_node_labels))>1 else 0.0

        edge_binary = (all_edge_preds >= self.hparams.threshold_structural).astype(int)
        edge_acc = accuracy_score(all_edge_labels, edge_binary)
        edge_recall = recall_score(all_edge_labels, edge_binary, zero_division=0)
        edge_auc = auc(*roc_curve(all_edge_labels, all_edge_preds)[:2]) if len(np.unique(all_edge_labels))>1 else 0.0

        self.log('test_loss_epoch', losses, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_node_acc_epoch', node_acc, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_node_recall_epoch', node_recall, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_node_auc_epoch', node_auc, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_edge_acc_epoch', edge_acc, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_edge_recall_epoch', edge_recall, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_edge_auc_epoch', edge_auc, on_epoch=True, prog_bar=False, logger=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    
    
class AttentionEdgeSample(pl.LightningModule):
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
        # Training HParams
        learning_rate: float = 1e-3,
        node_loss_w: float = 1.0,          # Weight for node loss
        edge_loss_w: float = 1.0,          # Weight for edge loss (new)
        threshold_structural: float = 0.5,  # Threshold for accuracy/recall calc
        mlp_dropout: float = 0.0,          # drop out for merge features
        skip_init=False
        ):
        super().__init__()
        self.save_hyperparameters()

      
        # Convolutional ROI embedding
        self.conv_roi_embedding = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=8, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=8, kernel_size=6, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(start_dim=1)
        )
        
        self.edge_patch_enc = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),         # 50×5 → 25×5
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),      # → 16×1×1
            nn.Flatten(),                 # → [E, 16]
        )
        self.patch_dim      = 8          # ← output of edge_patch_enc


        # Edge prediction head
        self.edge_loss_w = edge_loss_w
        self.edge_predictor = nn.Sequential(
            nn.Linear(2*self.hparams.out_channels + self.hparams.geom_channels + self.patch_dim, self.hparams.out_channels),
            nn.ReLU(),
            nn.Linear(self.hparams.out_channels, 1)
        )

        with torch.no_grad():
            x = torch.randn(1, 3, *self.hparams.roi_align_embedding_shape)
            self.hparams.channels_conv_roi_embedding = self.conv_roi_embedding(x).shape[1]

        self.node_fuse = nn.Sequential(
            nn.Linear(self.hparams.channels_conv_roi_embedding + self.hparams.in_channels_DeepLSD +  self.hparams.geom_channels,
                      self.hparams.out_channels),
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
                layers.append(LocalEdgeLayer(self.hparams.out_channels))
        self.layers = nn.ModuleList(layers)

        # Node prediction head
        self.mlp_textural_structural = nn.Sequential(
            nn.Linear(self.hparams.out_channels, self.hparams.out_channels),
            nn.ReLU(),
            nn.Linear(self.hparams.out_channels, 1)
        )

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Containers for metrics
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        #Learned weights for loss
        self.log_sigma_node = nn.Parameter(torch.zeros(()))
        self.log_sigma_edge = nn.Parameter(torch.zeros(()))

    def forward(self, batch):
        
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        roi_features = batch.roi_features 
        geo = batch.geo
        # ROI conv and fuse
        roi_feats = self.conv_roi_embedding(roi_features)
        concat_feat = torch.cat([roi_feats, x, geo], dim=1)
        concat_feat = self.node_fuse(concat_feat)

        # Prepare for global attention
        roi_dense, mask = to_dense_batch(concat_feat, batch_idx)
        desc = roi_dense.transpose(1, 2)
        edge_patch = self.edge_patch_enc(batch.edge_attr)   # [E, 16]

        # Alternate layers
        for layer in self.layers:
            if isinstance(layer, SelfAttnLayer):
                desc = layer(desc)
            else:
                flat = desc.transpose(1,2)[mask]
                delta = layer(flat, edge_index, edge_patch)
                delta_dense, _ = to_dense_batch(delta, batch_idx)
                desc = desc + delta_dense.transpose(1,2)

        # Collapse to node features
        features = desc.transpose(1,2)[mask]
        
        features = F.dropout(features, p=self.hparams.mlp_dropout, training=self.training)

        # Node logits
        node_logits = self.mlp_textural_structural(features)

        # Edge logits

        src, dst = batch.full_edge_index
        edge_geo  = edge_geometry(geo, src, dst)             # [E,5]

        h_src, h_dst = features[src], features[dst]
       
        edge_in = torch.cat([
            0.5 * (h_src + h_dst),        # symmetric mean      [E, D]
            (h_src - h_dst).abs(),        # symmetric distance  [E, D]
            edge_geo, # geometric extras    [E, 5]
            edge_patch
        ], dim=1)  
                
        edge_logits = self.edge_predictor(edge_in)

        return node_logits, edge_logits

    def training_step(self, batch, batch_idx):
        node_logits, edge_logits = self(batch)
        node_labels = batch.y.view(-1,1).float()
        full_edge_labels = batch.full_edge_labels
        # sample nodes
        pos_n = (node_labels==1).nonzero(as_tuple=True)[0]
        neg_n = (node_labels==0).nonzero(as_tuple=True)[0]
        if pos_n.numel()>0:
            perm = torch.randperm(neg_n.size(0), device=neg_n.device)
            sampled_neg_n = neg_n[perm[:pos_n.size(0)]]
            keep_n = torch.cat([pos_n, sampled_neg_n])
        else:
            k=min(32,neg_n.size(0)); perm=torch.randperm(neg_n.size(0),device=neg_n.device)
            keep_n=neg_n[perm[:k]]
        sampled_node_logits = node_logits[keep_n]
        sampled_node_labels = node_labels[keep_n]
        node_loss = self.criterion(sampled_node_logits, sampled_node_labels)
        # sample edges
        edge_labels_flat = full_edge_labels.view(-1,1)
        pos_e = (edge_labels_flat==1).nonzero(as_tuple=True)[0]
        neg_e = (edge_labels_flat==0).nonzero(as_tuple=True)[0]
        if pos_e.numel()>0:
            perm_e = torch.randperm(neg_e.size(0),device=neg_e.device)
            sampled_neg_e = neg_e[perm_e[:pos_e.size(0)]]
            keep_e = torch.cat([pos_e, sampled_neg_e])
        else:
            k_e=min(32,neg_e.size(0)); perm_e=torch.randperm(neg_e.size(0),device=neg_e.device)
            keep_e=neg_e[perm_e[:k_e]]
        sampled_edge_logits = edge_logits[keep_e]
        sampled_edge_labels = edge_labels_flat[keep_e]
        edge_loss = self.criterion(sampled_edge_logits, sampled_edge_labels)
        # combined
        
      
        loss = (node_loss * torch.exp(-2*self.log_sigma_node) +
                edge_loss * torch.exp(-2*self.log_sigma_edge) +
                self.log_sigma_node + self.log_sigma_edge) * 0.5

        # loss = self.hparams.node_loss_w*node_loss + self.hparams.edge_loss_w*edge_loss
        
        # metrics
        with torch.no_grad():
            n_probs=torch.sigmoid(sampled_node_logits); n_preds=(n_probs>=self.hparams.threshold_structural).int()
            node_acc=accuracy_score(sampled_node_labels.int().cpu(),n_preds.cpu())
            e_probs=torch.sigmoid(sampled_edge_logits); e_preds=(e_probs>=self.hparams.threshold_structural).int()
            edge_acc=accuracy_score(sampled_edge_labels.int().cpu(), e_preds.cpu())
            
            
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
        total_loss = self.hparams.node_loss_w * node_loss + self.hparams.edge_loss_w * edge_loss

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
        
        


        losses = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        all_node_preds = torch.cat([x['node_preds'] for x in self.validation_step_outputs]).cpu().numpy()
        all_node_labels = torch.cat([x['node_labels'] for x in self.validation_step_outputs]).cpu().numpy()
        all_edge_preds = torch.cat([x['edge_preds'] for x in self.validation_step_outputs]).cpu().numpy()
        all_edge_labels = torch.cat([x['edge_labels'] for x in self.validation_step_outputs]).cpu().numpy()

        # Node metrics
        node_binary = (all_node_preds >= self.hparams.threshold_structural).astype(int)
        node_acc = accuracy_score(all_node_labels, node_binary)
        node_recall = recall_score(all_node_labels, node_binary, zero_division=0)
        node_auc = auc(*roc_curve(all_node_labels, all_node_preds)[:2]) if len(np.unique(all_node_labels))>1 else 0.0

        # Edge metrics
        edge_binary = (all_edge_preds >= self.hparams.threshold_structural).astype(int)
        edge_acc = accuracy_score(all_edge_labels, edge_binary)
        edge_recall = recall_score(all_edge_labels, edge_binary, zero_division=0)
        edge_auc = auc(*roc_curve(all_edge_labels, all_edge_preds)[:2]) if len(np.unique(all_edge_labels))>1 else 0.0

        self.log('val_loss_epoch', losses, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_node_acc_epoch', node_acc, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_node_recall_epoch', node_recall, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_node_auc_epoch', node_auc, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_edge_acc_epoch', edge_acc, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_edge_recall_epoch', edge_recall, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_edge_auc_epoch', edge_auc, on_epoch=True, prog_bar=False, logger=True)
        combined_auc = 0.5 * (edge_auc + node_auc)
        self.log('val_combined_auc_epoch', combined_auc, on_epoch=True, prog_bar=False, logger=True)
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
        total_loss = self.hparams.node_loss_w * node_loss + self.hparams.edge_loss_w * edge_loss

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

        node_binary = (all_node_preds >= self.hparams.threshold_structural).astype(int)
        node_acc = accuracy_score(all_node_labels, node_binary)
        node_recall = recall_score(all_node_labels, node_binary, zero_division=0)
        node_auc = auc(*roc_curve(all_node_labels, all_node_preds)[:2]) if len(np.unique(all_node_labels))>1 else 0.0

        edge_binary = (all_edge_preds >= self.hparams.threshold_structural).astype(int)
        edge_acc = accuracy_score(all_edge_labels, edge_binary)
        edge_recall = recall_score(all_edge_labels, edge_binary, zero_division=0)
        edge_auc = auc(*roc_curve(all_edge_labels, all_edge_preds)[:2]) if len(np.unique(all_edge_labels))>1 else 0.0

        self.log('test_loss_epoch', losses, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_node_acc_epoch', node_acc, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_node_recall_epoch', node_recall, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_node_auc_epoch', node_auc, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_edge_acc_epoch', edge_acc, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_edge_recall_epoch', edge_recall, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_edge_auc_epoch', edge_auc, on_epoch=True, prog_bar=False, logger=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    
    
class AttentionEdgeSampleFull(pl.LightningModule):
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
        # Training HParams
        learning_rate: float = 1e-3,
        node_loss_w: float = 1.0,          # Weight for node loss
        edge_loss_w: float = 1.0,          # Weight for edge loss (new)
        threshold_structural: float = 0.5,  # Threshold for accuracy/recall calc
        mlp_dropout: float = 0.0,          # drop out for merge features
        skip_init=False
        ):
        super().__init__()
        self.save_hyperparameters()

        self.patch_dim      = 8          # ← output of edge_patch_enc

        # Convolutional ROI embedding
        self.conv_roi_embedding = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=8, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=8, kernel_size=6, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(start_dim=1)
        )
        

        # Edge prediction head
        self.edge_loss_w = edge_loss_w
        self.edge_predictor = nn.Sequential(
            nn.Linear(2*self.hparams.out_channels + self.hparams.geom_channels + self.patch_dim, self.hparams.out_channels),
            nn.ReLU(),
            nn.Linear(self.hparams.out_channels, 1)
        )

        with torch.no_grad():
            x = torch.randn(1, 3, *self.hparams.roi_align_embedding_shape)
            self.hparams.channels_conv_roi_embedding = self.conv_roi_embedding(x).shape[1]

        self.node_fuse = nn.Sequential(
            nn.Linear(self.hparams.channels_conv_roi_embedding + self.hparams.in_channels_DeepLSD +  self.hparams.geom_channels,
                      self.hparams.out_channels),
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
                edge_attr_dim=self.patch_dim,
                hidden_dim=self.hparams.out_channels))
        self.layers = nn.ModuleList(layers)

        # Node prediction head
        self.mlp_textural_structural = nn.Sequential(
            nn.Linear(self.hparams.out_channels, self.hparams.out_channels),
            nn.ReLU(),
            nn.Linear(self.hparams.out_channels, 1)
        )

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Containers for metrics
        self.validation_step_outputs = []
        self.test_step_outputs = []
        
        #Learned weights for loss
        self.log_sigma_node = nn.Parameter(torch.zeros(()))
        self.log_sigma_edge = nn.Parameter(torch.zeros(()))
        self._viz_image = None  # H×W×3 float
        self._viz_lines = None       # list of [(x1,y1),(x2,y2)]
                    
    def forward(self, batch):
        
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        roi_features = batch.roi_features 
        geo = batch.geo
        # ROI conv and fuse
        roi_feats = self.conv_roi_embedding(roi_features)
        concat_feat = torch.cat([roi_feats, x, geo], dim=1)
        concat_feat = self.node_fuse(concat_feat)

        # Prepare for global attention
        roi_dense, mask = to_dense_batch(concat_feat, batch_idx)
        desc = roi_dense.transpose(1, 2)
        
        
        

        # Alternate layers
        for layer in self.layers:
            if isinstance(layer, SelfAttnLayer):
                desc = layer(desc)
            else:
                flat = desc.transpose(1,2)[mask]
                delta = layer(flat, edge_index, batch.local_edge_patch)
                delta_dense, _ = to_dense_batch(delta, batch_idx)
                desc = desc + delta_dense.transpose(1,2)

        # Collapse to node features
        features = desc.transpose(1,2)[mask]
        
        features = F.dropout(features, p=self.hparams.mlp_dropout, training=self.training)

        # Node logits
        node_logits = self.mlp_textural_structural(features)

        # Edge logits

        src, dst = batch.full_edge_index
        edge_geo  = edge_geometry(geo, src, dst)             # [E,5]

        h_src, h_dst = features[src], features[dst]
       
        edge_in = torch.cat([
            0.5 * (h_src + h_dst),        # symmetric mean      [E, D]
            (h_src - h_dst).abs(),        # symmetric distance  [E, D]
            edge_geo, # geometric extras    [E, 5]
            batch.edge_patch
        ], dim=1)  
                
        edge_logits = self.edge_predictor(edge_in)

        return node_logits, edge_logits

    def training_step(self, batch, batch_idx):
        node_logits, edge_logits = self(batch)
        node_labels = batch.y.view(-1,1).float()
        full_edge_labels = batch.full_edge_labels
        # sample nodes
        pos_n = (node_labels==1).nonzero(as_tuple=True)[0]
        neg_n = (node_labels==0).nonzero(as_tuple=True)[0]
        if pos_n.numel()>0:
            perm = torch.randperm(neg_n.size(0), device=neg_n.device)
            sampled_neg_n = neg_n[perm[:pos_n.size(0)]]
            keep_n = torch.cat([pos_n, sampled_neg_n])
        else:
            k=min(32,neg_n.size(0)); perm=torch.randperm(neg_n.size(0),device=neg_n.device)
            keep_n=neg_n[perm[:k]]
        sampled_node_logits = node_logits[keep_n]
        sampled_node_labels = node_labels[keep_n]
        node_loss = self.criterion(sampled_node_logits, sampled_node_labels)
        # sample edges
        edge_labels_flat = full_edge_labels.view(-1,1)
        pos_e = (edge_labels_flat==1).nonzero(as_tuple=True)[0]
        neg_e = (edge_labels_flat==0).nonzero(as_tuple=True)[0]
        if pos_e.numel()>0:
            perm_e = torch.randperm(neg_e.size(0),device=neg_e.device)
            sampled_neg_e = neg_e[perm_e[:pos_e.size(0)]]
            keep_e = torch.cat([pos_e, sampled_neg_e])
        else:
            k_e=min(32,neg_e.size(0)); perm_e=torch.randperm(neg_e.size(0),device=neg_e.device)
            keep_e=neg_e[perm_e[:k_e]]
        sampled_edge_logits = edge_logits[keep_e]
        sampled_edge_labels = edge_labels_flat[keep_e]
        edge_loss = self.criterion(sampled_edge_logits, sampled_edge_labels)
        # combined
        
      
        loss = (node_loss * torch.exp(-2*self.log_sigma_node) +
                edge_loss * torch.exp(-2*self.log_sigma_edge) +
                self.log_sigma_node + self.log_sigma_edge) * 0.5

        # loss = self.hparams.node_loss_w*node_loss + self.hparams.edge_loss_w*edge_loss
        
        # metrics
        with torch.no_grad():
            n_probs=torch.sigmoid(sampled_node_logits); n_preds=(n_probs>=self.hparams.threshold_structural).int()
            node_acc=accuracy_score(sampled_node_labels.int().cpu(),n_preds.cpu())
            e_probs=torch.sigmoid(sampled_edge_logits); e_preds=(e_probs>=self.hparams.threshold_structural).int()
            edge_acc=accuracy_score(sampled_edge_labels.int().cpu(), e_preds.cpu())
            
            
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
        total_loss = self.hparams.node_loss_w * node_loss + self.hparams.edge_loss_w * edge_loss

        # log losses
        self.log('val_node_loss', node_loss,  on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('val_edge_loss', edge_loss,  on_step=True,on_epoch=False, prog_bar=True, logger=True)
        self.log('val_loss',      total_loss, on_step=True,on_epoch=False, prog_bar=True,  logger=True)
        if batch_idx == 0 and self._viz_batch is None:
            # (you may want to .detach() or move to cpu if you’re worried
            #  about GPU memory)
            img_tensor = batch.image[0]                     # [3,H,W]
            coords_t   = batch.coordinates[0]               # [N,2,2]
            # move to CPU, convert to simple types
            self._viz_image = img_tensor.permute(1,2,0).cpu().numpy()  # H×W×3 float
            self._viz_lines = coords_t.cpu().numpy().tolist()         # list of [(x1,y1),(x2,y2)]
                    
        # store for epoch-end
        self.validation_step_outputs.append({
            'node_preds': node_logits.sigmoid().detach(),
            'node_labels': node_labels.detach(),
            'edge_preds': edge_logits.sigmoid().detach(),
            'edge_labels': full_edge_labels.detach(),
            'loss': total_loss,
        })
        return {'loss': total_loss,  
            'node_preds': node_logits.sigmoid(),
            'node_labels': node_labels,
            'edge_preds': edge_logits.sigmoid(),
            'edge_labels': full_edge_labels}

    def on_validation_epoch_end(self):
        import matplotlib.pyplot as plt
        import wandb
        import torch
        import numpy as np
        from sklearn.metrics import accuracy_score, recall_score, roc_curve, auc
        from ground_truth.visualization import plot_images, plot_coplanar_lines, plot_lines_bool

        # 1) Do nothing if no outputs
        if not self.validation_step_outputs:
            return

        # 2) Aggregate losses & preds
        losses = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        all_node_preds = torch.cat([x['node_preds'] for x in self.validation_step_outputs]).cpu().numpy()
        all_node_labels = torch.cat([x['node_labels'] for x in self.validation_step_outputs]).cpu().numpy()
        all_edge_preds = torch.cat([x['edge_preds'] for x in self.validation_step_outputs]).cpu().numpy()
        all_edge_labels = torch.cat([x['edge_labels'] for x in self.validation_step_outputs]).cpu().numpy()

        # 3) Compute epoch‐level metrics
        node_binary = (all_node_preds >= self.hparams.threshold_structural).astype(int)
        node_acc    = accuracy_score(all_node_labels, node_binary)
        node_recall = recall_score(all_node_labels, node_binary, zero_division=0)
        node_auc    = auc(*roc_curve(all_node_labels, all_node_preds)[:2]) if len(np.unique(all_node_labels))>1 else 0.0

        edge_binary = (all_edge_preds >= self.hparams.threshold_structural).astype(int)
        edge_acc    = accuracy_score(all_edge_labels, edge_binary)
        edge_recall = recall_score(all_edge_labels, edge_binary, zero_division=0)
        edge_auc    = auc(*roc_curve(all_edge_labels, all_edge_preds)[:2]) if len(np.unique(all_edge_labels))>1 else 0.0

        combined_auc = 0.5 * (node_auc + edge_auc)

        # 4) Log scalars
        self.log('val_loss_epoch',      losses,    on_epoch=True)
        self.log('val_node_acc_epoch',  node_acc,  on_epoch=True)
        self.log('val_node_recall_epoch', node_recall, on_epoch=True)
        self.log('val_node_auc_epoch',  node_auc,  on_epoch=True)
        self.log('val_edge_acc_epoch',  edge_acc,  on_epoch=True)
        self.log('val_edge_recall_epoch', edge_recall, on_epoch=True)
        self.log('val_edge_auc_epoch',  edge_auc,  on_epoch=True)
        self.log('val_combined_auc_epoch', combined_auc, on_epoch=True)

        # 5) —— Visualization —— 
        # Pull one batch so we have the raw image + coordinates
    
     
        color_img   = self._viz_image      # H×W×3 numpy
        pred_lines = self._viz_lines      # list of [(x1,y1),(x2,y2)]

        node_preds_viz = self.validation_step_outputs[0]['node_preds'][0].cpu().flatten().tolist()
        edge_preds_viz  = self.validation_step_outputs[0]['edge_preds'][0].cpu().numpy()

        # 5a) Structural plots
        figs_struct = []
        flat_node_scores = node_preds_viz
        for i in range(min(4, len(pred_lines))):
            fig, ax = plt.subplots(figsize=(4,4))
            plot_lines_bool(ax, color_img, pred_lines, flat_node_scores)
            figs_struct.append(fig)

        # 5b) Coplanarity plots
        num_lines = len(pred_lines)
        edge_array = edge_preds_viz.reshape((num_lines, -1))
        figs_copl = []
        for i in range(min(4, num_lines)):
            fig, ax = plt.subplots(figsize=(4,4))
            plot_coplanar_lines(ax, pred_lines, edge_array[i], color_img)
            figs_copl.append(fig)

        # 6) Log all to W&B
        self.logger.experiment.log({
            "Validation/Structural": [
                wandb.Image(fig, caption=f"Line {i} structural")
                for i, fig in enumerate(figs_struct)
            ],
            "Validation/Coplanarity": [
                wandb.Image(fig, caption=f"Line {i} coplanarity")
                for i, fig in enumerate(figs_copl)
            ],
            "epoch": self.current_epoch
        })

        # 7) Clean up
        for fig in figs_struct + figs_copl:
            plt.close(fig)
        self._viz_batch = None

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
        total_loss = self.hparams.node_loss_w * node_loss + self.hparams.edge_loss_w * edge_loss

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

        node_binary = (all_node_preds >= self.hparams.threshold_structural).astype(int)
        node_acc = accuracy_score(all_node_labels, node_binary)
        node_recall = recall_score(all_node_labels, node_binary, zero_division=0)
        node_auc = auc(*roc_curve(all_node_labels, all_node_preds)[:2]) if len(np.unique(all_node_labels))>1 else 0.0

        edge_binary = (all_edge_preds >= self.hparams.threshold_structural).astype(int)
        edge_acc = accuracy_score(all_edge_labels, edge_binary)
        edge_recall = recall_score(all_edge_labels, edge_binary, zero_division=0)
        edge_auc = auc(*roc_curve(all_edge_labels, all_edge_preds)[:2]) if len(np.unique(all_edge_labels))>1 else 0.0

        self.log('test_loss_epoch', losses, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_node_acc_epoch', node_acc, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_node_recall_epoch', node_recall, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_node_auc_epoch', node_auc, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_edge_acc_epoch', edge_acc, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_edge_recall_epoch', edge_recall, on_epoch=True, prog_bar=False, logger=True)
        self.log('test_edge_auc_epoch', edge_auc, on_epoch=True, prog_bar=False, logger=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)