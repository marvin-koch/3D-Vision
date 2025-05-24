# lightning_datamodule.py
import os
import pytorch_lightning as pl
from torch.utils.data import random_split
from torch_geometric.loader import DataLoader

# Assuming dataset_inductive.py is in the same directory or accessible via PYTHONPATH
from dataset_inductive import GraphDatasetInductive
from torch.utils.data import WeightedRandomSampler
import numpy as np
import torch

from torch.utils.data import Subset # For type hinting in the new methods
from multiprocessing import Pool, cpu_count
from tqdm import tqdm # Import tqdm

class GraphDataModuleInductive(pl.LightningDataModule):
    def __init__(self, h5_path: str, roi_output_size: tuple = (64, 64),
                 batch_size: int = 32, train_split: float = 0.8,
                 val_split: float = 0.1, num_workers: int = 0,
                 method = 'roi', edge_sample_size = (32,16)):
        super().__init__()
        self.h5_path = h5_path
        self.roi_output_size = roi_output_size
        self.method = method

        self.batch_size = batch_size
        self.train_split = train_split
        self.val_split = val_split
        self.num_workers = num_workers # For DataLoader

        # Ensure splits are valid
        if not (0 < train_split < 1) or not (0 < val_split < 1) or (train_split + val_split >= 1):
             raise ValueError("Invalid train/val split percentages.")

        self.full_dataset = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.edge_sample_size = edge_sample_size
    def prepare_data(self):
        # Optional: Download data, etc. Not needed if data is local.
        # Check if json_dir exists
        if not os.path.isfile(self.h5_path):
            raise FileNotFoundError(f"JSON directory not found: {self.h5_path}")

    def setup(self, stage: str = None):
        # Load data and perform splits
        if not self.full_dataset:
            try:
                 self.full_dataset = GraphDatasetInductive(
                     h5_path=self.h5_path,
                     roi_output_size=self.roi_output_size,
                     method = self.method,
                     edge_sample_size=self.edge_sample_size
                 )
            except Exception as e:
                 print(f"Error loading dataset from {self.h5_path}: {e}")
                 raise

        if stage == 'fit' or stage is None:
            n_total = len(self.full_dataset)
            n_train = int(self.train_split * n_total)
            n_val = int(self.val_split * n_total)
            n_test = n_total - n_train - n_val

            if n_train <= 0 or n_val <= 0 or n_test <= 0:
                 raise ValueError(f"Dataset size {n_total} is too small for the requested splits.")

            self.train_ds, self.val_ds, self.test_ds = random_split(
                self.full_dataset, [n_train, n_val, n_test]
            )
            print(f"Dataset split: Train={len(self.train_ds)}, Val={len(self.val_ds)}, Test={len(self.test_ds)}")


        if stage == 'test' or stage is None:
             # Ensure test_ds is available if it wasn't created during 'fit'
             if not self.test_ds:
                  n_total = len(self.full_dataset)
                  n_train = int(self.train_split * n_total)
                  n_val = int(self.val_split * n_total)
                  n_test = n_total - n_train - n_val
                  # We only need test split here, but random_split requires all lengths
                  _, _, self.test_ds = random_split(
                       self.full_dataset, [n_train, n_val, n_test]
                  )


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,pin_memory=True, persistent_workers=self.num_workers > 0)
    

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,pin_memory=True, persistent_workers=self.num_workers > 0)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers,pin_memory=True, persistent_workers=self.num_workers > 0)
