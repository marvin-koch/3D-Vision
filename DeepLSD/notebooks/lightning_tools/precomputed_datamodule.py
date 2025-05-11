import os
import torch
from torch_geometric.data import Dataset
from pytorch_lightning import LightningDataModule
from torch_geometric.loader import DataLoader as PyGDataLoader
import gzip

class PyGGraphDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        # Sort files for consistent order, look for .pt.gz files
        self.sample_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt.gz')])

    def __len__(self):
        return len(self.sample_files)

    def __getitem__(self, idx: int):
        file_path = os.path.join(self.data_dir, self.sample_files[idx])
        # Load from a gzipped file
        with gzip.open(file_path, 'rb') as f:
            # Loads a single torch_geometric.data.Data object or any saved torch object
            return torch.load(f, weights_only=False)
        return data_obj

class GraphDataModule(LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 32, num_workers: int = 0):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        # save_hyperparameters() is good practice for PL modules
        self.save_hyperparameters('data_path','batch_size', 'num_workers')

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str = None):
        # Called on every DDP process if using DDP
        train_dir = os.path.join(self.data_path, 'train')
        val_dir = os.path.join(self.data_path, 'validation')
        test_dir = os.path.join(self.data_path, 'test')

        if stage == 'fit' or stage is None:
            self.train_dataset = PyGGraphDataset(train_dir)
            self.val_dataset = PyGGraphDataset(val_dir)
        
        if stage == 'validate' and self.val_dataset is None: # Ensure val_dataset is set for standalone validation
             self.val_dataset = PyGGraphDataset(val_dir)

        if stage == 'test' or stage is None:
            self.test_dataset = PyGGraphDataset(test_dir)

    def train_dataloader(self):
        return PyGDataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0)
        )

    def val_dataloader(self):
        return PyGDataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0)
        )

    def test_dataloader(self):
        return PyGDataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=(self.num_workers > 0)
        )