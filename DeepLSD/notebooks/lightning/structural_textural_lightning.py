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


class LitGATTexturalStructural(pl.LightningModule):
    def __init__(
        self,
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
    ):
        super().__init__()
        # Store hyperparameters using save_hyperparameters
        self.save_hyperparameters()


        # Calculate output size of the CNN embedding
        # Assuming input (B, 3, H, W), after Conv(3,2,k=3,s=1,p=1), ReLU, MaxPool(k=2,s=2), Conv(2,1,k=3,s=1,p=1), ReLU, Flatten
        # Output size: (B, 1 * (H/2) * (W/2))
        # Commented out for now, too few parameters
        # self.hparams.channels_conv_roi_embedding = (self.hparams.roi_align_embedding_shape[0] // 4) * (self.hparams.roi_align_embedding_shape[1] // 4) * 8
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
        
        # --- Model Architecture ---
        self.gat_roi = pyg_nn.GAT(
            in_channels=self.hparams.channels_conv_roi_embedding,
            hidden_channels=self.hparams.hidden_channels,
            out_channels=self.hparams.out_channels,
            v2=self.hparams.v2,
            num_layers=self.hparams.num_layers,
            dropout=self.hparams.dropout,
            act=self.hparams.act,
            jk=self.hparams.jk_layer
        )
        self.gat_DeepLSD = pyg_nn.GAT(
            in_channels=self.hparams.in_channels_DeepLSD,
            hidden_channels=self.hparams.hidden_channels,
            out_channels=self.hparams.out_channels,
            v2=self.hparams.v2,
            num_layers=self.hparams.num_layers,
            dropout=self.hparams.dropout,
            act=self.hparams.act,
            jk=self.hparams.jk_layer
        )
        # self.merge_features = nn.Sequential(
        #     nn.Linear(self.hparams.in_channels_DeepLSD + channels_conv_roi_embedding, self.hparams.in_channels),
        #     nn.Dropout(p=mlp_dropout),
        #     nn.GELU(),
        # )

        self.mlp_textural_structural = nn.Sequential(
            nn.Linear(2 * self.hparams.out_channels, self.hparams.out_channels),
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


    def forward(self, batch: Batch):
        """
        Performs the forward pass of the model.
        Returns raw logits for the node predictions.
        """
        x = batch.x
        roi_features = batch.roi_features
        edge_index = batch.edge_index

        
        roi_conv_output = self.conv_roi_embedding(roi_features)
        h_out_roi = self.gat_roi(roi_conv_output, edge_index)
        
        h_out_DeepLSD = self.gat_DeepLSD(x, edge_index)
        combined_features = torch.cat([h_out_roi, h_out_DeepLSD], dim=1)

        # Node-level predictions (logits)
        node_logits = self.mlp_textural_structural(combined_features)

        return node_logits # Return logits directly

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

    def training_step(self, batch: Batch, batch_idx: int):
        loss, _, _, train_acc = self._common_step(batch, batch_idx)

        # Log training loss
        self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('train_acc', train_acc, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        #self.training_step_outputs.append({'loss': loss}) # Store loss for epoch average
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
        val_acc_epoch = accuracy_score(all_labels, pred_binary)
        # Handle potential division by zero in recall if no positive labels exist
        if np.sum(all_labels) > 0:
            val_recall = recall_score(all_labels, pred_binary, zero_division=0)
        else:
            val_recall = 0.0 # Or NaN, depending on desired behavior

        self.log('val_acc_epoch', val_acc_epoch, on_epoch=True, prog_bar=False, logger=True)
        self.log('val_recall', val_recall, on_epoch=True, prog_bar=False, logger=True)

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
