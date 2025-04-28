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

@torch.amp.custom_fwd(cast_inputs=torch.float32)
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


class AttentionTexturalStructural(pl.LightningModule):
    
    def __init__(self,
        # Model HParams
        in_channels_DeepLSD: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        roi_align_embedding_shape: tuple,
        num_layers: int,
        dropout: float = 0.5,
        act: str = 'relu',
        v2: bool = True,
        jk_layer: str = None,
        # Training HParams
        learning_rate: float = 1e-3,
        node_loss_w: float = 1.0, # Weight for node loss
        threshold_structural: float = 0.5, # Threshold for accuracy/recall calc
        mlp_dropout: float = 0.0, # drop out for merge features
        skip_init=False
        ):
        super().__init__()
        self.save_hyperparameters()
        
        self.conv_roi_embedding = nn.Sequential(
            # Layer 1: 3 -> 8 channels, k=3, s=1, p=1 (preserves H, W before pooling)
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=8, stride=1, padding=1),
            nn.ReLU(),
            # First reduction: HxW -> H/2 x W/2
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 2: 8 -> 12 channels, k=3, s=1, p=1 (preserves H/2, W/2 before pooling)
            nn.Conv2d(in_channels=6, out_channels=8, kernel_size=6, stride=1, padding=1),
            nn.ReLU(),
            # Second reduction: H/2 x W/2 -> H/4 x W/4
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Flatten
            # Output shape: (B, 12 * (H/4) * (W/4))
            nn.Flatten(start_dim=1)
        )
        
       
        with torch.no_grad():
            # make a dummy batch of size 1
            x = torch.randn(1, 3, *self.hparams.roi_align_embedding_shape)
            self.hparams.channels_conv_roi_embedding = self.conv_roi_embedding(x).shape[1]
        
        
        self.node_fuse = nn.Sequential(
            nn.Linear(self.hparams.channels_conv_roi_embedding + self.hparams.in_channels_DeepLSD,
                      self.hparams.out_channels),
            nn.ReLU(),
            nn.Linear(self.hparams.out_channels,
                      self.hparams.out_channels),
        )
        # --- Model Architecture ---
        
        layers = []
        for i in range(self.hparams.num_layers):
            if i % 2 == 0:
                layers.append(SelfAttnLayer(self.hparams.out_channels, self.hparams.skip_init))
            else:
                layers.append(LocalEdgeLayer(self.hparams.out_channels))
        self.layers = nn.ModuleList(layers)
        
        self.mlp_textural_structural = nn.Sequential(
            nn.Linear(self.hparams.out_channels, self.hparams.out_channels),
            nn.ReLU(),
            nn.Linear(self.hparams.out_channels, 1) # Output single logit per node
        )
    

        # Loss function and activation
        # Using BCEWithLogitsLoss is generally more numerically stable than Sigmoid + BCELoss
        # self.criterion = nn.BCELoss()
        # self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCEWithLogitsLoss() # Recommended

        # Initialize lists to store outputs for epoch-level metrics
        # dont have enough memory for this
        # self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        

    def forward(self, batch):
        """
        batch: a torch_geometric.data.Batch with
          - batch.x:        [N, D] node features
          - batch.edge_index: [2, E] edges for local updates
          - batch.batch:    [N] graph IDs
        returns:
          [N, D] the refined node features
        """
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        roi_features = batch.roi_features
        
        roi_feats = self.conv_roi_embedding(roi_features)     # [N, roi_dim]


        concat_feat = torch.cat([roi_feats, x], dim=1)            # [N, roi_dim+det_dim]
        concat_feat = self.node_fuse(concat_feat)                                   # [N, gnn_in_dim]

        # 1) make dense for global attention
        roi_features_dense, mask = to_dense_batch(concat_feat, batch_idx)  # → [B, Nmax, D], mask [B, Nmax]
        desc = roi_features_dense.transpose(1, 2)               # → [B, D, Nmax]

        # 2) alternate layers
        for layer in self.layers:
            if isinstance(layer, SelfAttnLayer):
                desc = layer(desc)  # global self-attn
            else:
                # local update: collapse, apply, and re-densify
                flat  = desc.transpose(1,2)[mask]    # [N, D]
                delta = layer(flat, edge_index)      # [N, D]
                delta_dense, _ = to_dense_batch(delta, batch_idx)
                desc = desc + delta_dense.transpose(1,2)

        # 3) collapse back to flat [N, D]
        features = desc.transpose(1,2)[mask]  # [N, D]
        
        node_logits = self.mlp_textural_structural(features)

        return node_logits


    def _common_step(self, batch: Batch, batch_idx: int):
        """Common logic for training, validation, and test steps."""
        node_logits = self(batch)
        target_nodes = batch.y.float() # Ensure labels are float


        loss = self.criterion(node_logits, target_nodes)

        # Calculate predictions (probabilities) after loss calc using logits
        node_preds = torch.sigmoid(node_logits)
        # Log validation loss
        pred_binary = (node_preds >= self.hparams.threshold_structural).to(torch.int)
        acc = accuracy_score(target_nodes.cpu(), pred_binary.cpu())
        return loss, node_preds, target_nodes, acc

    # def training_step(self, batch: Batch, batch_idx: int):
    #     loss, _, _, train_acc = self._common_step(batch, batch_idx)

    #     # Log training loss
    #     self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
    #     self.log('train_acc', train_acc, on_step=True, on_epoch=False, prog_bar=True, logger=True)
    #     #self.training_step_outputs.append({'loss': loss}) # Store loss for epoch average
    #     return loss
    
    def training_step(self, batch: Batch, batch_idx: int):
        # 1) run the GNN+MLP to get per-node logits
        node_logits = self(batch)             # [N,1]
        logits = node_logits.view(-1)         # [N]
        labels = batch.y.view(-1).float()     # [N]

        # 2) build pos/neg index lists
        pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
        neg_idx = (labels == 0).nonzero(as_tuple=True)[0]

        # 3) sample negatives to match #positives (or fallback if no positives)
        if pos_idx.numel() > 0:
            perm = torch.randperm(neg_idx.size(0), device=neg_idx.device)
            sampled_neg = neg_idx[perm[: pos_idx.size(0)]]
            keep_idx = torch.cat([pos_idx, sampled_neg], dim=0)
        else:
            # no positives in this batch → take up to 32 negatives
            k = min(32, neg_idx.size(0))
            perm = torch.randperm(neg_idx.size(0), device=neg_idx.device)
            keep_idx = neg_idx[perm[:k]]

        # 4) compute loss on that balanced subset exactly like in _common_step
        sampled_logits = node_logits[keep_idx]          # [M,1]
        sampled_labels = labels[keep_idx].unsqueeze(1)  # [M,1]

        loss = self.criterion(sampled_logits, sampled_labels)

        # 5) get preds & compute accuracy at your threshold
        with torch.no_grad():
            sampled_probs  = torch.sigmoid(sampled_logits)           # [M,1]
            sampled_preds  = (sampled_probs >= self.hparams.threshold_structural).int()
            sampled_labels_int = sampled_labels.int()
            acc = accuracy_score(sampled_labels_int.cpu(), sampled_preds.cpu())

        # 6) log exactly as before
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_acc',      acc,  on_step=True, on_epoch=False, prog_bar=True, logger=True)

        return loss


    # def on_train_epoch_end(self):
    #     # Calculate and log average training loss for the epoch
    #     avg_loss = torch.stack([x['loss'] for x in self.training_step_outputs]).mean()
    #     self.log('train_loss_epoch', avg_loss, on_epoch=True, prog_bar=True, logger=True)
    #     self.training_step_outputs.clear() # Free memory


    def validation_step(self, batch: Batch, batch_idx: int):
        loss, node_preds, target_nodes, val_acc = self._common_step(batch, batch_idx)


        self.log('val_loss_step', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True) # Log step loss if desired
        self.log('val_acc', val_acc, on_step=True, on_epoch=False, prog_bar=False, logger=True) # Log step loss if desired

        # Store predictions and targets for epoch-end metrics
        self.validation_step_outputs.append({
            'loss': loss.detach(),
            'preds': node_preds.detach(),
            'labels': target_nodes.detach()
        })
        return {'loss': loss, 'preds': node_preds, 'labels': target_nodes}


    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            print("Warning: validation_step_outputs is empty.")
            return

        # Aggregate loss, predictions, and labels
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        all_preds = torch.cat([x['preds'] for x in self.validation_step_outputs]).cpu().numpy()
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs]).cpu().numpy()

        # Log average validation loss
        self.log('val_loss_epoch', avg_loss, on_epoch=True, prog_bar=True, logger=True)

        # Calculate and log ROC AUC if there are positive and negative samples
        try:
            if len(np.unique(all_labels)) > 1: # Check for both classes
                 fpr, tpr, _ = roc_curve(all_labels, all_preds)
                 roc_auc = auc(fpr, tpr)
                 self.log('val_auc', roc_auc, on_epoch=True, prog_bar=True, logger=True)
            else:
                 self.log('val_auc', 0.0, on_epoch=True, prog_bar=True, logger=True) # Handle single-class case
                 print(f"Validation Step: Only one class present in labels. AUC set to 0.0.")

        except Exception as e:
            print(f"Could not calculate validation AUC: {e}")
            self.log('val_auc', 0.0, on_epoch=True, prog_bar=True, logger=True) # Log default value on error

        # Calculate and log Accuracy and Recall at the specified threshold
        pred_binary = (all_preds >= self.hparams.threshold_structural).astype(int)
        val_acc = accuracy_score(all_labels, pred_binary)
        # Handle potential division by zero in recall if no positive labels exist
        if np.sum(all_labels) > 0:
            val_recall = recall_score(all_labels, pred_binary, zero_division=0)
        else:
            val_recall = 0.0 # Or NaN, depending on desired behavior

        self.log('val_acc', val_acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recall', val_recall, on_epoch=True, prog_bar=True, logger=True)

        self.validation_step_outputs.clear() # Free memory

    def test_step(self, batch: Batch, batch_idx: int):
        loss, node_preds, target_nodes,_= self._common_step(batch, batch_idx)

        # Store results for aggregation
        self.test_step_outputs.append({
            'loss': loss.detach(),
            'preds': node_preds.detach(),
            'labels': target_nodes.detach()
        })
        return {'loss': loss, 'preds': node_preds, 'labels': target_nodes}


    def on_test_epoch_end(self):
        if not self.test_step_outputs:
            print("Warning: test_step_outputs is empty.")
            return

        # Aggregate loss, predictions, and labels
        avg_loss = torch.stack([x['loss'] for x in self.test_step_outputs]).mean()
        all_preds = torch.cat([x['preds'] for x in self.test_step_outputs]).cpu().numpy()
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs]).cpu().numpy()

        # Log average test loss
        self.log('test_loss', avg_loss, logger=True)

        # Calculate metrics at the specified threshold
        pred_binary = (all_preds >= self.hparams.threshold_structural).astype(int)
        test_acc = accuracy_score(all_labels, pred_binary)
        if np.sum(all_labels) > 0:
            test_recall = recall_score(all_labels, pred_binary, zero_division=0)
        else:
             test_recall = 0.0

        # Calculate ROC AUC
        test_auc = 0.0
        try:
            if len(np.unique(all_labels)) > 1:
                fpr, tpr, _ = roc_curve(all_labels, all_preds)
                test_auc = auc(fpr, tpr)
                 # Optionally plot ROC curve
                 # plot_roc_curve(all_labels, all_preds, title='Test ROC Curve', save_path='test_roc_curve.png')
            else:
                 print(f"Test Step: Only one class present in labels. AUC set to 0.0.")

        except Exception as e:
            print(f"Could not calculate test AUC: {e}")

        # Log test metrics
        test_metrics = {
            'test_acc': test_acc,
            'test_recall': test_recall,
            'test_auc': test_auc
        }
        self.log_dict(test_metrics, logger=True)
        print(f"Test Results: Loss={avg_loss:.4f}, Acc={test_acc:.4f}, Recall={test_recall:.4f}, AUC={test_auc:.4f}")

        # Store aggregated results if needed externally
        self.test_results = {'preds': all_preds, 'labels': all_labels}

        self.test_step_outputs.clear() # Free memory


    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.hparams.learning_rate)
        # Example of adding a learning rate scheduler
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        # return [optimizer], [scheduler]
        return optimizer


class AttentionBoth(pl.LightningModule):
    def __init__(self,
        # Model HParams
        in_channels_DeepLSD: int,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
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
            nn.Linear(2*self.hparams.out_channels, self.hparams.out_channels),
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
            nn.Linear(self.hparams.channels_conv_roi_embedding + self.hparams.in_channels_DeepLSD,
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
        
        self.log_sigma_node = nn.Parameter(torch.zeros(()))
        self.log_sigma_edge = nn.Parameter(torch.zeros(()))


    def forward(self, batch):
        x, edge_index, batch_idx = batch.x, batch.edge_index, batch.batch
        roi_features = batch.roi_features

        # ROI conv and fuse
        roi_feats = self.conv_roi_embedding(roi_features)
        concat_feat = torch.cat([roi_feats, x], dim=1)
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

        # Node logits
        node_logits = self.mlp_textural_structural(features)

        # Edge logits
        src, dst = batch.full_edge_index
        h_src, h_dst = features[src], features[dst]
        edge_in = torch.cat([h_src, h_dst], dim=1)
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
        loss = self.hparams.node_loss_w*node_loss + self.hparams.edge_loss_w*edge_loss
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