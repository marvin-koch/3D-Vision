# train_lightning.py
import os
import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml
import argparse
from tqdm import tqdm
from torch.utils.data import Subset # For type hinting in the new methods
from structural_textural_lightning import LitGATTexturalStructural, plot_roc_curve
from lightning_datamodule import GraphDataModuleInductive
from gluestick_matin import *
import gzip

# Attempt to import Batch from torch_geometric.data for specific handling
try:
    from torch_geometric.data import Batch as TG_Batch
    IS_TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TG_Batch = None # Placeholder if torch_geometric is not available
    IS_TORCH_GEOMETRIC_AVAILABLE = False

def _debatch_pytorch_standard(batch_data):
    """
    De-batches standard PyTorch batch formats.
    Assumes batch_data is either:
    1. A single tensor (batch_dim=0).
    2. A list/tuple of tensors (each tensor has batch_dim=0, all same batch size).
    Returns a list of samples, where each sample is reconstructed.
    """
    if isinstance(batch_data, torch.Tensor):
        # Unbind along the batch dimension (dim 0)
        return list(torch.unbind(batch_data, dim=0))
    elif isinstance(batch_data, (list, tuple)) and all(isinstance(item, torch.Tensor) for item in batch_data):
        if not batch_data: return [] # Empty list/tuple
        # Assume all tensors in the tuple/list have the same batch size at dim 0.
        num_samples_in_batch = batch_data[0].size(0)
        reconstructed_samples = []
        for i in range(num_samples_in_batch):
            reconstructed_samples.append(tuple(tensor[i] for tensor in batch_data))
        return reconstructed_samples
    elif isinstance(batch_data, (list, tuple)):
        # If it's a list/tuple but not of tensors, assume it's already a list of samples
        # (e.g., from a custom collate_fn or if batch_size=1 and dataset yields complex objects).
        return batch_data
    else:
        # Fallback: if it's a single complex object not covered, treat it as one sample.
        return [batch_data]

def save_datamodule_samples(datamodule: pl.LightningDataModule, save_path: str):
    """
    Draws samples from the datamodule's train, validation, and test dataloaders
    and saves them individually as .pt.gz files to the specified path with progress bars.

    The data is saved in the expected structure:
    save_path/
     ├── train/
     │   ├── sample_0.pt.gz
     │   └── ...
     ├── validation/
     │   ├── sample_0.pt.gz
     │   └── ...
     └── test/
         ├── sample_0.pt.gz
         └── ...

    Args:
        datamodule (pl.LightningDataModule): The PyTorch Lightning DataModule instance.
        save_path (str): The root directory where data will be saved.
    """
    base_dir = os.path.abspath(save_path)
    print(f"Preparing to save individual samples to {base_dir} from datamodule.")

    # 1. Ensure datamodule is prepared and set up
    try:
        print("Calling datamodule.prepare_data()...")
        datamodule.prepare_data()
        print("Calling datamodule.setup(stage=None)...")
        datamodule.setup(stage=None) # Ensure all splits are set up
    except Exception as e:
        print(f"Warning: Error during datamodule.prepare_data() or .setup(): {e}")
        print("Proceeding, but dataloaders might not be correctly initialized.")

    # 2. Define dataloaders to process
    #    Order: train, validation, test
    split_loader_fns = {
        "train": datamodule.train_dataloader,
        "validation": datamodule.val_dataloader,
        "test": datamodule.test_dataloader,
    }

    for split_name, loader_fn in split_loader_fns.items():
        dataloader_instance = None
        try:
            dataloader_instance = loader_fn()
        except Exception as e:
            print(f"Could not get {split_name}_dataloader: {e}. Skipping {split_name} split.")
            continue

        if dataloader_instance is None:
            print(f"{split_name.capitalize()} dataloader is None. Skipping save for {split_name} split.")
            continue
        
        try:
            # Check if the dataset underlying the dataloader has items
            if not hasattr(dataloader_instance, 'dataset') or len(dataloader_instance.dataset) == 0:
                print(f"{split_name.capitalize()} dataloader's dataset is empty. Skipping save for {split_name} split.")
                continue
        except TypeError:
            # IterableDataset might not have __len__. Proceed cautiously.
            print(f"Warning: Could not determine length of {split_name} dataset (possibly IterableDataset). Proceeding with saving.")
            pass # Continue to try iterating

        split_dir = os.path.join(base_dir, split_name)
        try:
            os.makedirs(split_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating directory {split_dir}: {e}. Skipping {split_name} save.")
            continue

        print(f"\nSaving {split_name} samples to {split_dir}...")
        sample_idx_counter = 0
        
        try:
            total_batches = len(dataloader_instance)
        except TypeError: # Dataloader might be an IterableDataset without __len__
            total_batches = None # tqdm will not show ETA

        any_batch_processed = False
        # Assuming batch_size is 1 as per problem description for dataloaders used with this function
        for batch_data in tqdm(dataloader_instance, desc=f"Saving {split_name}", unit="batch", total=total_batches, ncols=100):
            any_batch_processed = True
            samples_in_batch = []

            # The IS_TORCH_GEOMETRIC_AVAILABLE and TG_Batch check is for torch_geometric.data.Batch
            # These are assumed to be defined globally or passed/imported appropriately.
            if IS_TORCH_GEOMETRIC_AVAILABLE and isinstance(batch_data, TG_Batch): # type: ignore
                samples_in_batch = batch_data.to_data_list()
            else:
                # Use the helper for standard PyTorch batch formats
                samples_in_batch = _debatch_pytorch_standard(batch_data)
            
            if not samples_in_batch and batch_data is not None:
                # If de-batching returned empty but original batch was not,
                # save the whole batch as one sample to avoid data loss.
                # This case is less likely if batch_size=1, as _debatch_pytorch_standard([item]) -> [item]
                tqdm.write(f"Warning: De-batching for {split_name} batch of type {type(batch_data)} resulted in empty list or was not handled by specific de-batchers. Saving entire batch as one sample.")
                samples_in_batch = [batch_data]


            for sample_item in samples_in_batch:
                # Changed to .pt.gz
                sample_filename = f"sample_{sample_idx_counter}.pt.gz"
                file_path = os.path.join(split_dir, sample_filename)
                with gzip.open(file_path, 'wb') as f:
                    torch.save(sample_item, f)
                sample_idx_counter += 1
        
        if not any_batch_processed and (total_batches is None or total_batches == 0) and sample_idx_counter == 0:
            print(f"Warning: {split_name.capitalize()} dataloader did not yield any batches, or was empty.")
            
        print(f"Finished saving {sample_idx_counter} {split_name} samples.")

    print(f"\nAll available samples processed. Check {base_dir} for output.")


def main(config_path: str):
    # --- Load Configuration from YAML ---
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return

    print("Configuration loaded successfully:")
    # Limit printing very large configs if necessary
    # print(yaml.dump(cfg, indent=2))

    cfg_data = cfg.get('data', {})
    cfg_model = cfg.get('model', {})
    cfg_train = cfg.get('training', {})

    # --- Set Seed ---
    pl.seed_everything(cfg_train.get('seed', 42), workers=True)

    # --- Data ---
    print("Setting up DataModule...")
    roi_output_size_tuple = tuple(cfg_data.get('roi_output_size', [64, 64]))
    num_workers = cfg_data.get('num_workers', 0)
    # Automatically set num_workers to 0 if no CUDA detected to avoid potential issues
    if not torch.cuda.is_available() and num_workers > 0:
        print(f"Warning: CUDA not available, setting num_workers to 0 (was {num_workers})")
        num_workers = 0
    print("Using following method to extract features: {}".format(cfg_data.get('method', 'roi')))
    data_module = GraphDataModuleInductive(
        json_dir=cfg_data.get('json_dir', './json_output/'),
        roi_output_size=roi_output_size_tuple,
        batch_size=cfg_data.get('batch_size', 1),
        train_split=cfg_data.get('train_split', 0.8),
        val_split=cfg_data.get('val_split', 0.1),
        num_workers=num_workers,
        method = cfg_data.get('method', 'roi'),
        edge_sample_size = cfg_data.get('edge_sample_size', (32,8))
    )

    save_datamodule_samples(datamodule = data_module, save_path = os.path.join(cfg_data.get('data_dir','./'),'graph_data/'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train GAT Textural/Structural Model using PyTorch Lightning")
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to the YAML configuration file (default: config.yaml)'
    )
    args = parser.parse_args()
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    torch.multiprocessing.set_sharing_strategy('file_system')
    main(config_path=args.config)