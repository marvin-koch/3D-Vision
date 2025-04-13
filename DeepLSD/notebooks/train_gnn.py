# %%
import os
# os.environ["PYTORCH_ENABLE_MPS_FALLBACK"]="1" # Keep if needed for MPS # does not work for some reason
import torch
dataset_path = 'graph_dataset/' # Or your preferred path
os.makedirs(dataset_path, exist_ok=True)
data_module_root = dataset_path # Module's root directory

# %%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
image_data = None

# %%
import torch
import numpy as np
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import DataLoader # Use PyG DataLoader
import pytorch_lightning as pl
from torch.utils.data import random_split, Subset # For splitting
from tqdm import tqdm
import logging
import pickle

# Configure logging (ensure it's configured globally or within your main script)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Assume LineGraphRegressionDatasetInMemory class is defined here ---
# (Copying it from the previous response for completeness)
class LineGraphRegressionDataset(Dataset):
    """
    PyTorch Geometric Dataset for graph regression on line features.
    (Copied from previous response - Keeps data in memory, handles processing/saving/loading)
    """
    def __init__(self, root, image_data, load_precomputed_dataset=False):
        super().__init__(root)
        self.root = root
        self.image_data = image_data
        self._processed_file_path = os.path.join(self.root, 'processed_graphs.pkl')
        self.processed_data = []

        if load_precomputed_dataset:
            if os.path.exists(self._processed_file_path):
                logging.info(f"Dataset: Attempting to load precomputed data from: {self._processed_file_path}")
                self.load_dataset(self._processed_file_path)
            else:
                logging.warning(f"Dataset: load_precomputed_dataset=True but file not found at: {self._processed_file_path}. Processing from image_data instead.")
                self._process_data()
        else:
            # Process only if the file doesn't already exist OR if explicitly told not to load precomputed
            # This prevents reprocessing if the file exists but load_precomputed_dataset was False
            if not os.path.exists(self._processed_file_path):
                 logging.info("Dataset: No precomputed file found. Processing dataset from provided image_data...")
                 self._process_data()
            else:
                 logging.info(f"Dataset: Precomputed file exists at {self._processed_file_path}, but load_precomputed_dataset=False. Reprocessing anyways and overwriting.")
                 # Load the existing data even if load_precomputed_dataset is False,
                 # assuming the user wants to use the existing processed data unless it's missing.
                 # self.load_dataset(self._processed_file_path)
                 # If you strictly want to reprocess whenever load_precomputed_dataset is False,
                 # uncomment the next line and comment out the load_dataset call above.
                 self._process_data()

    def _process_data(self):
        processed_list = []
        # Consider filtering image_data ONLY if it's not None
        valid_image_keys = []
        if self.image_data:
             valid_image_keys = [
                 k for k, v in self.image_data.items() if v.get('line_info') and isinstance(v['line_info'], list)
             ]
        else: # If image_data is None (e.g., loading precomputed failed and no fallback)
            logging.error("Dataset: Cannot process data, image_data is None.")
            self.processed_data = []
            return

        if not valid_image_keys:
             logging.warning("Dataset: Input image_data contains no samples with valid 'line_info'. Dataset will be empty.")
             self.processed_data = []
             return

        # --- (Processing loop - same as before) ---
        for image_key in tqdm(valid_image_keys, desc="Processing Images"):
            image_sample = self.image_data[image_key]
            line_info_list = image_sample.get('line_info')
            if not line_info_list: continue

            node_features, node_labels, valid_node_indices = [], [], []
            for line_idx, line_dict in enumerate(line_info_list):
                features_tensor = line_dict.get('features')
                if features_tensor is None: continue
                if not isinstance(features_tensor, (torch.Tensor, np.ndarray)): continue
                if isinstance(features_tensor, np.ndarray): features_tensor = torch.from_numpy(features_tensor)
                if features_tensor.shape != (3, 3, 64, 64):
                    print("Incorrect features_tensor.shape detected for line number {}. Skipping this line.".format(line_idx))
                    continue # Skip incorrect shapes

                node_feat = features_tensor.reshape(-1).float() # Flatten

                score = line_dict.get('score')
                if score is None: continue
                try: node_score = float(score)
                except (ValueError, TypeError): continue

                node_features.append(node_feat)
                node_labels.append(node_score)
                valid_node_indices.append(line_idx)

            if not node_features: continue

            x = torch.stack(node_features, dim=0)
            y = torch.tensor(node_labels, dtype=torch.float).unsqueeze(1)
            num_nodes = x.size(0)

            if num_nodes > 1: # Fully connected edge_index
                 adj = torch.ones((num_nodes, num_nodes), dtype=torch.long)
                 adj.fill_diagonal_(0)
                 edge_index, _ = dense_to_sparse(adj)
            else: edge_index = torch.empty((2, 0), dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y, image_key=image_key, num_nodes=num_nodes)
            processed_list.append(data)
        # --- (End Processing loop) ---

        self.processed_data = processed_list
        logging.info(f"Dataset: Processing complete. Created {len(self.processed_data)} graphs.")
        # Automatically save after processing if image_data was provided
        if self.image_data and len(self.processed_data) > 0:
             self.save_dataset() # Save the newly processed data

    def save_dataset(self, path=None):
        save_path = path if path else self._processed_file_path
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        try:
            with open(save_path, 'wb') as f:
                 pickle.dump(self.processed_data, f)
            logging.info(f"Dataset: Successfully saved {len(self.processed_data)} graphs to: {save_path}")
        except Exception as e:
            logging.error(f"Dataset: Failed to save dataset to {save_path}: {e}")

    def load_dataset(self, path=None):
        load_path = path if path else self._processed_file_path
        if not os.path.exists(load_path):
            logging.error(f"Dataset: Cannot load dataset. File not found: {load_path}")
            self.processed_data = []
            return False # Indicate failure

        try:
            with open(load_path, 'rb') as f:
                loaded_data = pickle.load(f)
            if isinstance(loaded_data, list) and all(isinstance(item, Data) for item in loaded_data):
                self.processed_data = loaded_data
                logging.info(f"Dataset: Successfully loaded {len(self.processed_data)} graphs from: {load_path}")
                return True # Indicate success
            else:
                 logging.error(f"Dataset: Loaded data from {load_path} is not a list of torch_geometric.data.Data objects.")
                 self.processed_data = []
                 return False # Indicate failure
        except Exception as e:
            logging.error(f"Dataset: Failed to load dataset from {load_path}: {e}")
            self.processed_data = []
            return False # Indicate failure

    def len(self): return len(self.processed_data)
    def get(self, idx):
        if not self.processed_data or idx < 0 or idx >= len(self.processed_data):
             # Proper error handling depends on context, raising IndexError is common for get
             raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.processed_data)}")
        return self.processed_data[idx]

    @property
    def raw_file_names(self): return []
    @property
    def processed_file_names(self): return [os.path.basename(self._processed_file_path)]
# --- End of LineGraphRegressionDatasetInMemory class ---


class LineGraphDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the Line Graph Regression task.

    Uses LineGraphRegressionDatasetInMemory to handle data processing,
    saving, and loading. Manages train/validation/test splits and DataLoaders.
    """
    def __init__(self,
                 image_data: dict,
                 root: str = './graph_data_processed',
                 load_precomputed_dataset: bool = False,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 train_val_test_split: tuple = (0.8, 0.1, 0.1),
                 seed: int = 42):
        """
        Args:
            image_data (dict): Raw image data dictionary (required if not loading precomputed).
            root (str): Directory to save/load the processed graph data file ('processed_graphs.pkl').
            load_precomputed_dataset (bool): If True, attempts to load data from 'root'.
                                             If False, processes 'image_data' unless a
                                             processed file already exists in 'root'.
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of workers for DataLoaders.
            train_val_test_split (tuple): Tuple containing the fractions for
                                          train, validation, and test splits.
                                          Must sum to 1.0. Test split is optional
                                          (e.g., (0.8, 0.2)).
            seed (int): Random seed for reproducible splits.
        """
        super().__init__()
        # Save parameters accessible via self.hparams
        # Note: image_data can be large, consider omitting it from hparams if logging full hyperparameters
        self.save_hyperparameters(ignore=['image_data'])

        self.image_data = image_data # Store raw data reference
        self.root = root
        self.load_precomputed_dataset_flag = load_precomputed_dataset # Renamed to avoid potential hparams conflict if kept
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_val_test_split = train_val_test_split
        self.seed = seed

        # Placeholders for datasets after setup
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """
        Handles dataset processing or checking for precomputed file existence.
        This method is called once per node. Assign no state here (self.xyz = ...).
        It ensures that the data file will be available for `setup`.
        """
        logging.info("DataModule: prepare_data() called.")
        # Instantiate the dataset. Its __init__ handles processing vs. loading logic.
        # We don't store this instance (`_`) long-term here.
        _ = LineGraphRegressionDataset(
            root=self.root,
            image_data=self.image_data,
            load_precomputed_dataset=self.load_precomputed_dataset_flag
        )
        logging.info("DataModule: prepare_data() finished.")
        # Now, either the dataset was loaded, or it was processed (and saved if processing happened).

    def setup(self, stage: str = None):
        """
        Handles dataset loading, splitting, and assignment.
        Called on every GPU process.
        """
        logging.info(f"DataModule: setup(stage={stage}) called.")
        # Instantiate the dataset *again*. Now it should load quickly if prepare_data worked.
        # This ensures each process/GPU gets its own dataset instance correctly.
        self.dataset = LineGraphRegressionDataset(
            root=self.root,
            image_data=self.image_data, # Pass again, though likely ignored if loading
            load_precomputed_dataset=True # Force loading now, prepare_data ensured file exists
        )

        dataset_size = len(self.dataset)
        if dataset_size == 0:
            logging.error("DataModule: Dataset is empty after setup. Cannot create splits.")
            # Set splits to empty lists or None to avoid errors in dataloaders
            self.train_dataset, self.val_dataset, self.test_dataset = [], [], []
            return

        logging.info(f"DataModule: Full dataset size: {dataset_size}")

        # Calculate split lengths
        train_frac, val_frac = self.train_val_test_split[0], self.train_val_test_split[1]
        test_frac = self.train_val_test_split[2] if len(self.train_val_test_split) == 3 else 0.0

        if not np.isclose(train_frac + val_frac + test_frac, 1.0):
             logging.warning(f"Split fractions {self.train_val_test_split} do not sum to 1. Adjusting.")
             # Normalize or raise error - let's normalize test fraction implicitly
             if train_frac + val_frac > 1.0:
                 raise ValueError("Sum of train and validation fractions exceeds 1.0")
             test_frac = 1.0 - train_frac - val_frac

        n_train = int(np.floor(train_frac * dataset_size))
        n_val = int(np.floor(val_frac * dataset_size))
        n_test = dataset_size - n_train - n_val # Ensure all samples are used

        if n_train == 0 or n_val == 0 or (test_frac > 0 and n_test == 0) :
             logging.warning(f"Dataset size ({dataset_size}) too small for requested splits ({n_train}/{n_val}/{n_test}). Some splits might be empty.")

        logging.info(f"DataModule: Splitting into Train: {n_train}, Val: {n_val}, Test: {n_test}")

        # Perform the split using torch.utils.data.random_split
        # Important: random_split works directly on datasets implementing __len__ and __getitem__
        generator = torch.Generator().manual_seed(self.seed)
        splits = random_split(self.dataset, [n_train, n_val, n_test], generator=generator)

        self.train_dataset = splits[0]
        self.val_dataset = splits[1]
        # Only assign test_dataset if test split is non-zero
        self.test_dataset = splits[2] if n_test > 0 else None

        logging.info(f"DataModule: setup() finished. Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}, Test size: {len(self.test_dataset) if self.test_dataset else 0}")


    def train_dataloader(self):
        """Creates the DataLoader for the training set."""
        if not self.train_dataset: return None # Handle empty dataset case
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          pin_memory=True, # Often good for GPU training
                          persistent_workers=True if self.num_workers > 0 else False)

    def val_dataloader(self):
        """Creates the DataLoader for the validation set."""
        if not self.val_dataset: return None
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=True if self.num_workers > 0 else False)

    def test_dataloader(self):
        """Creates the DataLoader for the test set."""
        if not self.test_dataset: return None
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=True if self.num_workers > 0 else False)

    def save_dataset(self, path=None):
        """
        Saves the processed dataset using the underlying dataset's save method.
        Note: This saves the *full* dataset, not the splits.
              Requires `setup()` to have been called at least once to ensure
              `self.dataset` is initialized.

        Args:
            path (str, optional): Path to save the dataset file. Defaults to
                                  the path configured in the underlying dataset.
        """
        if self.dataset is None:
             # Attempt to initialize the dataset if setup hasn't run
             logging.warning("DataModule: save_dataset called before setup. Attempting to initialize dataset first.")
             try:
                 # Initialize just to get access to the save method and path logic
                 # Use load_precomputed=True to avoid reprocessing if possible
                  temp_dataset = LineGraphRegressionDataset(
                     root=self.root, image_data=self.image_data, load_precomputed_dataset=True
                 )
                  temp_dataset.save_dataset(path)
             except Exception as e:
                 logging.error(f"DataModule: Failed to initialize dataset for saving: {e}")
                 print("DataModule: Could not save dataset. Ensure data exists or can be processed.")
        else:
            self.dataset.save_dataset(path)

    def load_dataset(self, path=None):
        """
        Loads the processed dataset using the underlying dataset's load method.
        Note: This loads the *full* dataset. It's generally recommended to use
              the `load_precomputed_dataset=True` flag during initialization.
              Calling this manually *after* setup might require re-running setup
              or the Trainer's fit loop to use the newly loaded data.

        Args:
            path (str, optional): Path to load the dataset file from. Defaults to
                                  the path configured in the underlying dataset.

        Returns:
            bool: True if loading was successful, False otherwise.
        """
        if self.dataset is None:
             # Initialize a temporary dataset instance just to call load
             logging.warning("DataModule: load_dataset called before setup. Attempting to initialize and load.")
             try:
                 temp_dataset = LineGraphRegressionDataset(
                     root=self.root, image_data=None, load_precomputed_dataset=False # Don't trigger processing here
                 )
                 return temp_dataset.load_dataset(path)
             except Exception as e:
                 logging.error(f"DataModule: Failed to initialize dataset for loading: {e}")
                 return False
        else:
            # If setup has run, load into the existing dataset instance
            success = self.dataset.load_dataset(path)
            if success:
                logging.info("DataModule: Dataset loaded successfully via load_dataset(). You may need to re-run setup or trainer.fit() to apply changes.")
            return success


# # 1. Make sure you have your 'image_data' dictionary populated
# # Example (replace with your actual data loading):
# # image_data = load_my_data_function(...)

# # Check if image_data exists and is a dictionary
# if 'image_data' in locals() and isinstance(image_data, dict):
#
#     # 2. Define the root directory for processed data
#     dataset_root = './my_graph_dataset'
#     os.makedirs(dataset_root, exist_ok=True) # Ensure directory exists
#
#     # 3. Instantiate the dataset
#     # This will automatically trigger the .process() method if
#     # the processed files don't exist in dataset_root/processed/
#     print(f"Initializing dataset. Processing data if needed...")
#     dataset = LineGraphRegressionDataset(root=dataset_root, image_data=image_data)
#     print(f"Dataset ready. Number of graphs: {len(dataset)}")
#
#     # 4. Access data samples (optional)
#     if len(dataset) > 0:
#         first_graph = dataset[0]
#         print("\n--- First Graph Sample ---")
#         print(first_graph)
#         print(f"Number of nodes: {first_graph.num_nodes}")
#         print(f"Node features shape: {first_graph.x.shape}")
#         print(f"Node labels shape: {first_graph.y.shape}")
#         print(f"Original image key: {first_graph.image_key}") # Example of accessing custom attribute
#
#     # 5. Use with DataLoader (example)
#     from torch_geometric.loader import DataLoader
#     loader = DataLoader(dataset, batch_size=4, shuffle=True)
#
#     print("\n--- DataLoader Example ---")
#     # Iterate through batches
#     # for batch in loader:
#     #    print(f"Processing batch with {batch.num_graphs} graphs...")
#     #    # Your training/evaluation logic here using 'batch'
#     #    pass
#
# else:
#     print("Error: The 'image_data' variable is not defined or is not a dictionary.")
#     print("Please ensure your data is loaded into a dictionary named 'image_data' before instantiating the dataset.")

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GATConv, LayerNorm # Import LayerNorm
import wandb
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
from torch_geometric.data import Dataset, Data
from torch_geometric.utils import dense_to_sparse
from torch_geometric.loader import DataLoader # Use PyG DataLoader
from torch.utils.data import random_split, Subset # For splitting
from tqdm import tqdm
import logging
import pickle
import os # Make sure os is imported


class CNNFeatureExtractor(nn.Module):
    def __init__(self, output_dim=128):
        super().__init__()
        # Assuming input is conceptually (N, 9, 64, 64) where 9 = 3x3 channels
        # Or handle the (3, 3, 64, 64) shape inside forward if needed
        self.conv1 = nn.Conv2d(9, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # -> 32x32
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # -> 16x16
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # -> 8x8
        # Global Average Pooling or Flatten + Linear
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, output_dim)
        
    def forward(self, x):
        # x likely has shape [num_nodes_in_batch, 3, 3, 64, 64]
        # Reshape for Conv2d: [num_nodes_in_batch, 9, 64, 64]
        num_nodes = x.shape[0]
        x = x.view(num_nodes, 9, 64, 64)

        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.adaptive_pool(x) # -> [num_nodes, 128, 1, 1]
        x = torch.flatten(x, 1)   # -> [num_nodes, 128]
        x = F.relu(self.fc(x))    # -> [num_nodes, output_dim]
        return x




# --- Dataset Class (Minimal changes for clarity/robustness) ---
# Configure logging (ensure it's configured globally or within your main script)
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s') # Uncomment if needed

class LineGraphRegressionDataset(Dataset):
    """
    PyTorch Geometric Dataset for graph regression on line features.
    (Keeps data in memory, handles processing/saving/loading)
    """
    def __init__(self, root, image_data, load_precomputed_dataset=False):
        self.root = root
        self.image_data = image_data # Keep reference if needed for reprocessing
        self._processed_file_path = os.path.join(self.root, 'processed_graphs.pkl')
        self.processed_data = []
        # Initialize first, then decide action based on flags/files
        super().__init__(root) # Call super().__init__ after self.root is set
        
        if load_precomputed_dataset:
            if os.path.exists(self._processed_file_path):
                logging.info(f"Dataset: Attempting to load precomputed data from: {self._processed_file_path}")
                loaded = self.load_dataset(self._processed_file_path)
                if not loaded: # If loading failed, process from image_data if available
                    logging.warning("Dataset: Loading failed. Processing from image_data instead.")
                    if self.image_data:
                         self._process_data()
                    else:
                         logging.error("Dataset: Cannot process data after failed load, image_data is None.")
            else:
                logging.warning(f"Dataset: load_precomputed_dataset=True but file not found at: {self._processed_file_path}. Processing from image_data if available.")
                if self.image_data:
                     self._process_data()
                else:
                     logging.error("Dataset: Cannot process data, image_data is None.")
        else:
            # If not loading precomputed, always process (overwrite if exists)
            logging.info("Dataset: load_precomputed_dataset=False. Processing dataset from provided image_data...")
            if self.image_data:
                self._process_data()
            else:
                logging.error("Dataset: Cannot process data, image_data is None.")


    def _process_data(self):
        processed_list = []
        valid_image_keys = []
        if self.image_data:
             valid_image_keys = [
                 k for k, v in self.image_data.items() if v.get('line_info') and isinstance(v['line_info'], list)
             ]
        else:
            logging.error("Dataset: Cannot process data, image_data is None.")
            self.processed_data = []
            return

        if not valid_image_keys:
             logging.warning("Dataset: Input image_data contains no samples with valid 'line_info'. Dataset will be empty.")
             self.processed_data = []
             return

        for image_key in tqdm(valid_image_keys, desc="Processing Images"):
            image_sample = self.image_data[image_key]
            line_info_list = image_sample.get('line_info')
            if not line_info_list: continue

            node_features, node_labels, valid_node_indices = [], [], []
            for line_idx, line_dict in enumerate(line_info_list):
                features_tensor = line_dict.get('features')
                if features_tensor is None: continue
                if not isinstance(features_tensor, (torch.Tensor, np.ndarray)): continue
                if isinstance(features_tensor, np.ndarray): features_tensor = torch.from_numpy(features_tensor)
                if features_tensor.shape != (3, 3, 64, 64):
                    # Use logging instead of print
                    logging.warning(f"Incorrect features_tensor.shape {features_tensor.shape} detected for image {image_key}, line index {line_idx}. Expected (3, 3, 64, 64). Skipping line.")
                    continue # Skip incorrect shapes

                node_feat = features_tensor.reshape(-1).float() # Flatten

                score = line_dict.get('score')
                if score is None: continue
                try:
                    node_score = float(score)
                    # Add check/warning if score is outside expected [0, 1] range for Sigmoid
                    if not (0.0 <= node_score <= 1.0):
                        logging.warning(f"Score {node_score} for image {image_key}, line {line_idx} is outside [0, 1] range. Sigmoid output assumes [0, 1].")
                except (ValueError, TypeError):
                    logging.warning(f"Invalid score type {type(score)} or value for image {image_key}, line {line_idx}. Skipping line.")
                    continue

                node_features.append(node_feat)
                node_labels.append(node_score)
                valid_node_indices.append(line_idx)

            if not node_features: continue

            x = torch.stack(node_features, dim=0)
            y = torch.tensor(node_labels, dtype=torch.float).unsqueeze(1)
            num_nodes = x.size(0)

            if num_nodes > 1: # Fully connected edge_index
                 adj = torch.ones((num_nodes, num_nodes), dtype=torch.long)
                 adj.fill_diagonal_(0)
                 edge_index, _ = dense_to_sparse(adj)
            else: edge_index = torch.empty((2, 0), dtype=torch.long)

            # Ensure data consistency
            if y.shape[0] != num_nodes:
                logging.error(f"Shape mismatch in image {image_key}: num_nodes={num_nodes}, y.shape={y.shape}. Skipping graph.")
                continue

            data = Data(x=x, edge_index=edge_index, y=y, image_key=image_key, num_nodes=num_nodes)
            processed_list.append(data)

        self.processed_data = processed_list
        logging.info(f"Dataset: Processing complete. Created {len(self.processed_data)} graphs.")
        if len(self.processed_data) > 0:
             self.save_dataset() # Save the newly processed data

    def save_dataset(self, path=None):
        save_path = path if path else self._processed_file_path
        save_dir = os.path.dirname(save_path)
        os.makedirs(save_dir, exist_ok=True)
        try:
            with open(save_path, 'wb') as f:
                 pickle.dump(self.processed_data, f)
            logging.info(f"Dataset: Successfully saved {len(self.processed_data)} graphs to: {save_path}")
        except Exception as e:
            logging.error(f"Dataset: Failed to save dataset to {save_path}: {e}")

    def load_dataset(self, path=None):
        load_path = path if path else self._processed_file_path
        if not os.path.exists(load_path):
            logging.error(f"Dataset: Cannot load dataset. File not found: {load_path}")
            self.processed_data = []
            return False

        try:
            with open(load_path, 'rb') as f:
                loaded_data = pickle.load(f)
            if isinstance(loaded_data, list) and all(isinstance(item, Data) for item in loaded_data):
                # Add basic validation check on loaded data
                if not loaded_data:
                    logging.warning(f"Dataset: Loaded file {load_path} contains an empty list.")
                    self.processed_data = []
                    return True # File loaded, but it's empty

                first_item = loaded_data[0]
                if not hasattr(first_item, 'x') or not hasattr(first_item, 'edge_index') or not hasattr(first_item, 'y'):
                     logging.error(f"Dataset: Data in {load_path} seems incomplete (missing x, edge_index, or y).")
                     self.processed_data = []
                     return False

                self.processed_data = loaded_data
                logging.info(f"Dataset: Successfully loaded {len(self.processed_data)} graphs from: {load_path}")
                return True
            else:
                 logging.error(f"Dataset: Loaded data from {load_path} is not a list of torch_geometric.data.Data objects.")
                 self.processed_data = []
                 return False
        except (pickle.UnpicklingError, EOFError, AttributeError, ImportError, IndexError) as e:
            logging.error(f"Dataset: Failed to load or parse dataset from {load_path} (file might be corrupted or incompatible): {e}")
            self.processed_data = []
            return False
        except Exception as e:
            logging.error(f"Dataset: An unexpected error occurred loading dataset from {load_path}: {e}")
            self.processed_data = []
            return False

    def len(self): return len(self.processed_data)
    def get(self, idx):
        if not self.processed_data or idx < 0 or idx >= len(self.processed_data):
             raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.processed_data)}")
        return self.processed_data[idx]

    @property
    def raw_file_names(self): return []
    @property
    def processed_file_names(self): return [os.path.basename(self._processed_file_path)] if hasattr(self, '_processed_file_path') else []

# --- Data Module Class (Mostly unchanged, relies on Dataset's logic) ---
class LineGraphDataModule(pl.LightningDataModule):
    def __init__(self,
                 image_data: dict, # Can be None if load_precomputed_dataset is True and file exists
                 root: str = './graph_data_processed',
                 load_precomputed_dataset: bool = False,
                 batch_size: int = 32,
                 num_workers: int = 0,
                 train_val_test_split: tuple = (0.8, 0.1, 0.1),
                 seed: int = 42):
        super().__init__()
        # image_data can be large, avoid saving to hparams log file
        self.save_hyperparameters(ignore=['image_data'])

        # Store args directly for internal use
        self._image_data = image_data
        self._root = root
        self._load_precomputed_dataset_flag = load_precomputed_dataset
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._train_val_test_split = train_val_test_split
        self._seed = seed

        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        logging.info("DataModule: prepare_data() called.")
        # Instantiate the dataset; its __init__ handles processing/loading logic.
        # We don't need to store the instance ('_') here.
        _ = LineGraphRegressionDataset(
            root=self._root,
            image_data=self._image_data,
            load_precomputed_dataset=self._load_precomputed_dataset_flag
        )
        # After this, the processed file should exist (either loaded or created).
        logging.info("DataModule: prepare_data() finished.")


    def setup(self, stage: str = None):
        logging.info(f"DataModule: setup(stage={stage}) called.")
        # Instantiate dataset *again*. This time, force loading as prepare_data should
        # have ensured the file exists. Pass image_data=None as it's not needed for loading.
        self.dataset = LineGraphRegressionDataset(
            root=self._root,
            image_data=None, # Not needed if loading
            load_precomputed_dataset=True # Force loading attempt
        )

        dataset_size = len(self.dataset)
        if dataset_size == 0:
            logging.error("DataModule: Dataset is empty after setup. Cannot create splits.")
            self.train_dataset, self.val_dataset, self.test_dataset = [], [], []
            return

        logging.info(f"DataModule: Full dataset size: {dataset_size}")

        # Calculate split lengths
        if len(self._train_val_test_split) == 2: # Handle (train, val) tuple
             train_frac, val_frac = self._train_val_test_split
             test_frac = 0.0
        elif len(self._train_val_test_split) == 3:
             train_frac, val_frac, test_frac = self._train_val_test_split
        else:
            raise ValueError("train_val_test_split must be a tuple of 2 or 3 floats.")


        if not np.isclose(train_frac + val_frac + test_frac, 1.0):
             logging.warning(f"Split fractions {self._train_val_test_split} do not sum to 1. Normalizing.")
             total_frac = train_frac + val_frac + test_frac
             train_frac /= total_frac
             val_frac /= total_frac
             test_frac /= total_frac


        n_train = int(np.floor(train_frac * dataset_size))
        n_val = int(np.floor(val_frac * dataset_size))
        # Assign remaining to test split to ensure all data is used
        n_test = dataset_size - n_train - n_val

        # Handle cases where dataset is too small for splits
        if dataset_size > 0 and (n_train == 0 or n_val == 0):
             logging.warning(f"Dataset size ({dataset_size}) is small for the split ratios. Train ({n_train})/Val ({n_val}) splits might be empty or very small.")
        if test_frac > 0 and n_test == 0 and dataset_size > 0:
             logging.warning(f"Test split fraction requested, but calculated size is 0 ({n_test}).")


        logging.info(f"DataModule: Splitting into Train: {n_train}, Val: {n_val}, Test: {n_test}")

        # Perform split
        generator = torch.Generator().manual_seed(self._seed)
        # Ensure split lengths sum exactly to dataset_size
        split_lengths = [n_train, n_val, n_test]
        if sum(split_lengths) != dataset_size:
             # Adjust the largest split (usually train) if there's a rounding discrepancy
             diff = dataset_size - sum(split_lengths)
             split_lengths[0] += diff
             logging.info(f"Adjusting split lengths slightly due to rounding: {split_lengths}")


        # Check if any split length is negative (shouldn't happen with floor and remainder logic)
        if any(s < 0 for s in split_lengths):
             raise ValueError(f"Calculated negative split size: {split_lengths}. Check ratios and dataset size.")


        # Only split if dataset is not empty
        if dataset_size > 0 :
            try:
                splits = random_split(self.dataset, split_lengths, generator=generator)
                self.train_dataset = splits[0]
                self.val_dataset = splits[1]
                self.test_dataset = splits[2] if n_test > 0 else None # Assign None if test size is 0
            except ValueError as e:
                 logging.error(f"Error during random_split: {e}. Dataset size: {dataset_size}, split lengths: {split_lengths}")
                 # Set splits to empty to prevent dataloader errors
                 self.train_dataset, self.val_dataset, self.test_dataset = [], [], []
        else:
            self.train_dataset, self.val_dataset, self.test_dataset = [], [], []


        logging.info(f"DataModule: setup() finished. Train size: {len(self.train_dataset)}, Val size: {len(self.val_dataset)}, Test size: {len(self.test_dataset) if self.test_dataset else 0}")


    def train_dataloader(self):
        if not self.train_dataset: return None
        return DataLoader(self.train_dataset,
                          batch_size=self._batch_size,
                          shuffle=True,
                          num_workers=self._num_workers,
                          pin_memory=True,
                          persistent_workers=True if self._num_workers > 0 else False)

    def val_dataloader(self):
        if not self.val_dataset: return None
        return DataLoader(self.val_dataset,
                          batch_size=self._batch_size,
                          shuffle=False,
                          num_workers=self._num_workers,
                          pin_memory=True,
                          persistent_workers=True if self._num_workers > 0 else False)

    def test_dataloader(self):
        if not self.test_dataset: return None
        return DataLoader(self.test_dataset,
                          batch_size=self._batch_size,
                          shuffle=False,
                          num_workers=self._num_workers,
                          pin_memory=True,
                          persistent_workers=True if self._num_workers > 0 else False)

    # save/load methods delegate to the underlying dataset instance
    def save_dataset(self, path=None):
        if self.dataset is None:
            logging.warning("DataModule: save_dataset called before setup. Dataset not initialized.")
            # Optionally, try to initialize and save if needed, but it's better practice to ensure setup is called first.
            # self.setup() # Could call setup, but might have side effects
            print("DataModule: Cannot save dataset. Run setup() first.")
            return
        self.dataset.save_dataset(path)

    def load_dataset(self, path=None):
         if self.dataset is None:
             logging.warning("DataModule: load_dataset called before setup. Initializing temporary dataset for loading.")
             # Create a temporary instance just to load
             temp_dataset = LineGraphRegressionDataset(root=self._root, image_data=None, load_precomputed_dataset=False)
             return temp_dataset.load_dataset(path)
             # Note: This loaded data isn't automatically used unless setup is run again.
         else:
             # Load into the existing dataset instance
             success = self.dataset.load_dataset(path)
             if success:
                 logging.info("DataModule: Dataset loaded successfully via load_dataset(). Re-run setup() or trainer.fit() to use the new data.")
             return success


# --- GAT Regressor Model (Added LayerNorm,) ---
class GATRegressor(pl.LightningModule):
    def __init__(self,
                 input_dim: int,
                 cnn_output_dim = 128,
                 hidden_dim: int = 128,
                 output_dim: int = 1,
                 n_heads: int = 4,
                 n_layers: int = 2,
                 dropout: float = 0.2,
                 learning_rate: float = 5e-4):
        super().__init__()
        self.save_hyperparameters()
        self.feature_extractor = CNNFeatureExtractor(output_dim=cnn_output_dim)
        # --- Network Architecture ---
        self.input_embed = nn.Linear(cnn_output_dim, self.hparams.hidden_dim)
        # Add LayerNorm after initial embedding
        self.input_norm = LayerNorm(self.hparams.hidden_dim)

        self.gat_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList() # Store LayerNorm layers corresponding to GAT layers
        current_dim = self.hparams.hidden_dim

        for i in range(self.hparams.n_layers):
            is_last_layer = (i == self.hparams.n_layers - 1)
            heads = 1 if is_last_layer else self.hparams.n_heads
            concat = False if is_last_layer else True
            gat_input_dim = current_dim # Input to GAT is output of previous layer (or input_embed)

            conv = GATConv(gat_input_dim,
                           self.hparams.hidden_dim,
                           heads=heads,
                           dropout=self.hparams.dropout,
                           concat=concat)
            self.gat_layers.append(conv)

            # Determine the output dimension of the GAT layer
            if concat:
                 current_dim = self.hparams.hidden_dim * heads
            else: # Last layer with concat=False
                 current_dim = self.hparams.hidden_dim

            # Add LayerNorm after each GAT layer's activation/dropout (except maybe the very last)
            if not is_last_layer: # Don't normalize right before the final linear output layer
                 self.norm_layers.append(LayerNorm(current_dim))
            # Note: current_dim now reflects the output dimension for the *next* layer's input or the final output layer

        # Final Output Layer
        # Input dim is the output dim of the last GAT layer
        final_gat_output_dim = self.hparams.hidden_dim # Since last layer has concat=False, heads=1
        self.output_layer = nn.Linear(final_gat_output_dim, self.hparams.output_dim)

        self.loss_fn = nn.BCEWithLogitsLoss()
        

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # 0. Apply CNN
        x = self.feature_extractor(x)
        # 1. Apply further linear layer, normalization, activation, dropout
        x = self.input_embed(x)
        x = self.input_norm(x) # Apply LayerNorm
        x = F.relu(x)
        x = F.dropout(x, p=self.hparams.dropout, training=self.training)

        # 2. GAT layers with intermediate normalization
        for i, layer in enumerate(self.gat_layers):
            x = layer(x, edge_index)
            is_last_layer = (i == self.hparams.n_layers - 1)

            if not is_last_layer:
                # Apply norm -> activation -> dropout for intermediate layers
                x = self.norm_layers[i](x) # Apply the corresponding LayerNorm
                x = F.relu(x)
                x = F.dropout(x, p=self.hparams.dropout, training=self.training)
            # No activation/dropout/norm after the final GAT layer before the output layer

        # 3. Final output layer
        logits= self.output_layer(x)


        return logits

    def _calculate_accuracy(self, y_hat, y, batch_vector):
        # Ensure tensors are on the same device for comparison
        y = y.to(y_hat.device)
        batch_vector = batch_vector.to(y_hat.device)

        # Use a threshold (e.g., 0.5 if using sigmoid) for accuracy
        # Adjust threshold if needed, or use a different metric (e.g., MAE)
        y_true_thresh = (y >= 0.5).float()
        y_pred_thresh = (y_hat >= 0.5).float()

        accuracies = []
        for graph_idx in torch.unique(batch_vector):
            mask = (batch_vector == graph_idx)
            nodes_in_graph = mask.sum().item()

            if nodes_in_graph > 0:
                # Compare thresholded values
                correct_predictions = (y_true_thresh[mask] == y_pred_thresh[mask]).sum().item()
                graph_accuracy = correct_predictions / nodes_in_graph
                accuracies.append(graph_accuracy)

        if not accuracies:
            return torch.tensor(0.0, device=self.device)

        avg_accuracy = torch.tensor(accuracies, device=self.device).mean()
        return avg_accuracy


    def _shared_step(self, batch, batch_idx):
        if not hasattr(batch, 'y') or not hasattr(batch, 'batch'):
            # Check if batch size is 0, can happen if last batch is skipped or dataset empty
            if batch.num_graphs == 0:
                 logging.warning(f"Skipping step {batch_idx}: Batch contains 0 graphs.")
                 # Return None or dummy values to avoid errors downstream
                 # Returning None might require handling in the Pytorch Lightning training loop internals
                 # Let's return zero loss/acc for now, but this batch contributes nothing.
                 return torch.tensor(0.0, device=self.device, requires_grad=True), torch.tensor(0.0, device=self.device) # Loss needs grad potentially
            else:
                raise ValueError("Batch object must have 'y' (labels) and 'batch' attributes.")

        y_hat = self.forward(batch) # [N, 1] or [N] if output_dim=1 was squeezed earlier

        # Ensure y exists and has data
        if batch.y is None or batch.y.numel() == 0:
             logging.warning(f"Skipping step {batch_idx}: Batch has missing or empty labels 'y'.")
             return torch.tensor(0.0, device=self.device, requires_grad=True), torch.tensor(0.0, device=self.device)


        # --- Shape Handling ---
        # Squeeze prediction if output_dim is 1
        if self.hparams.output_dim == 1 and y_hat.dim() > 1:
            y_hat = y_hat.squeeze(-1) # -> [N]

        # Ensure y is also squeezed if it's [N, 1]
        y = batch.y
        if y.dim() > 1 and y.shape[1] == 1:
            y = y.squeeze(-1) # -> [N]

        # Final shape check
        if y_hat.shape != y.shape:
             # Check if the mismatch is due to an empty graph processed somehow
             if y_hat.numel() == 0 and y.numel() == 0:
                  logging.warning(f"Step {batch_idx}: Both prediction and target are empty (likely empty graph). Returning zero loss.")
                  return torch.tensor(0.0, device=self.device, requires_grad=True), torch.tensor(0.0, device=self.device)
             else:
                 raise RuntimeError(f"Shape mismatch before loss calculation: y_hat {y_hat.shape}, y {y.shape} in batch {batch_idx}")

        # Check for NaNs/Infs *before* loss calculation
        if torch.isnan(y_hat).any() or torch.isinf(y_hat).any():
            logging.error(f"NaN or Inf detected in model output (y_hat) at step {batch_idx}!")
            # Optionally: return a large loss or handle differently
            # For now, let loss calculation proceed, which will likely result in NaN loss
            pass
        if torch.isnan(y).any() or torch.isinf(y).any():
            logging.error(f"NaN or Inf detected in labels (y) at step {batch_idx}!")
             # Return zero loss/acc if labels are bad
            return torch.tensor(0.0, device=self.device, requires_grad=True), torch.tensor(0.0, device=self.device)


        # --- Loss and Accuracy ---
        loss = self.loss_fn(y_hat, y)

        # Check for NaN loss
        if torch.isnan(loss):
            logging.error(f"NaN loss detected at step {batch_idx}! y_hat min/max: {y_hat.min()}/{y_hat.max()}, y min/max: {y.min()}/{y.max()}")
            # Consider alternatives: return 0 loss, raise error, or try to debug further
            # Returning 0 might mask the problem, but prevents crashing training immediately
            # loss = torch.tensor(0.0, device=self.device, requires_grad=True) # Example: replace NaN loss

        # Calculate accuracy only if loss is valid
        accuracy = torch.tensor(0.0, device=self.device) # Default
        if not torch.isnan(loss) and not torch.isinf(loss):
             # Ensure batch vector is valid before calculating accuracy
             if hasattr(batch, 'batch') and batch.batch is not None and batch.batch.numel() == y.numel():
                 accuracy = self._calculate_accuracy(y_hat, y, batch.batch)
             else:
                 logging.warning(f"Cannot calculate accuracy at step {batch_idx}: Invalid or mismatched 'batch' vector.")


        return loss, accuracy

    def training_step(self, batch, batch_idx):
        loss, accuracy = self._shared_step(batch, batch_idx)
        # Log metrics, providing batch_size for correct aggregation
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 0 # Get actual number of graphs
        if batch_size > 0:
             self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
             self.log('train_avg_acc', accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        elif loss is not None: # Log step loss even if batch size is 0, but epoch avg will be wrong
             self.log('train_loss_step', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, accuracy = self._shared_step(batch, batch_idx)
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 0
        if batch_size > 0 and loss is not None and not torch.isnan(loss): # Only log valid steps
            self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
            self.log('val_avg_acc', accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=batch_size)
        # No return needed unless you aggregate manually

    def test_step(self, batch, batch_idx):
        loss, accuracy = self._shared_step(batch, batch_idx)
        batch_size = batch.num_graphs if hasattr(batch, 'num_graphs') else 0
        if batch_size > 0 and loss is not None and not torch.isnan(loss): # Only log valid steps
            self.log('test_loss', loss, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)
            self.log('test_avg_acc', accuracy, on_step=False, on_epoch=True, logger=True, batch_size=batch_size)
        # No return needed

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer

# --- Main Script Execution ---

# Configure logging (do this once at the start)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Load your image_data (replace with your actual loading)
# Placeholder: Assume image_data is loaded correctly
# Example: image_data = {... your data ...}
# Make sure this variable exists and is populated before proceeding
if 'image_data' not in locals() or not isinstance(image_data, dict):
      logging.error("The 'image_data' dictionary is not defined or populated. Load your data first.")
      # Example dummy data for testing structure (replace!)
      # image_data = {'img1': {'line_info': [{'features': np.random.rand(3, 3, 64, 64).astype(np.float32), 'score': 0.8},
      #                                      {'features': np.random.rand(3, 3, 64, 64).astype(np.float32), 'score': 0.3}]},
      #               'img2': {'line_info': [{'features': np.random.rand(3, 3, 64, 64).astype(np.float32), 'score': 0.9}]}}
      # If you don't have data yet, exit or load dummy data.
      exit() # Or raise error


# --- Dataset Instantiation  ---

logging.info(f"Initializing dataset. Processing data if needed...")
# Let's try loading first if the file exists, otherwise process
processed_file = os.path.join(dataset_path, 'processed_graphs.pkl')
should_load_precomputed = os.path.exists(processed_file)
logging.info(f"Precomputed file exists: {should_load_precomputed}. Setting load_precomputed_dataset accordingly.")

# Instantiate dataset (using the variable determined above)
dataset = LineGraphRegressionDataset(root=dataset_path,
                                     image_data=image_data,
                                     load_precomputed_dataset=should_load_precomputed)

if len(dataset) > 0:
    logging.info(f"Dataset ready. Number of graphs: {len(dataset)}")
    first_graph = dataset[0]
    logging.info(f"First graph: Nodes={first_graph.num_nodes}, Features={first_graph.x.shape}, Labels={first_graph.y.shape}, Key={getattr(first_graph, 'image_key', 'N/A')}")
else:
    logging.error("Dataset is empty after initialization. Check data processing or loading.")
    exit()


# --- Data Module Instantiation ---

batch_s = 4 # Keep batch size 1 if higher values caused issues before, but try increasing later
train_val_test_ratio = (0.7, 0.1, 0.2)
# Let DataModule decide whether to load based on file existence via prepare_data
load_precomputed_dm = True # Set to True so setup *attempts* loading first

logging.info(f"\n--- Initializing DataModule ---")
data_module = LineGraphDataModule(
    image_data=image_data, # Pass image data for potential reprocessing if loading fails
    root=data_module_root,
    load_precomputed_dataset=load_precomputed_dm, # Flag for prepare_data
    batch_size=batch_s,
    train_val_test_split=train_val_test_ratio
)

# --- Prepare and Setup Data ---
logging.info(f"\n--- Preparing Data ---")
data_module.prepare_data() # Ensures processed file exists

logging.info(f"\n--- Setting up DataLoaders ---")
data_module.setup() # Loads data and creates splits

# Optional: Save dataset state after setup if needed (usually done by prepare_data if processed)
# data_module.save_dataset()

# --- Verification ---
logging.info(f"\n--- Verifying Train DataLoader ---")
train_loader = data_module.train_dataloader()

if train_loader is not None and len(train_loader) > 0:
    logging.info(f"Number of training batches: {len(train_loader)}")
    try:
        first_batch = next(iter(train_loader))
        logging.info(f"First batch type: {type(first_batch)}")
        if hasattr(first_batch, 'num_graphs'):
            logging.info(f"Number of graphs in first batch: {first_batch.num_graphs}")
            if first_batch.num_graphs > 0:
                first_graph_in_batch = first_batch.get_example(0)
                logging.info("\n--- First Graph in First Training Batch ---")
                logging.info(f"Nodes: {first_graph_in_batch.num_nodes}, Features: {first_graph_in_batch.x.shape}, Labels: {first_graph_in_batch.y.shape}")
                if hasattr(first_graph_in_batch, 'image_key'):
                    logging.info(f"Original image key: {first_graph_in_batch.image_key}")
                logging.info(f"Edge index shape: {first_graph_in_batch.edge_index.shape}")
            else:
                 logging.warning("First batch contains 0 graphs.")
        else:
            logging.warning("First batch object does not have 'num_graphs' attribute.")

    except StopIteration:
        logging.error("Train DataLoader is empty, cannot get first batch.")
    except Exception as e:
        logging.error(f"Error while inspecting first batch: {e}")

elif data_module.train_dataset is not None and len(data_module.train_dataset) == 0:
     logging.warning("Training dataset is empty (likely due to small total dataset size or split configuration).")
else:
    logging.warning("Train DataLoader could not be created or is None.")


# --- Training Setup ---
# Login to Wandb (do this *before* initializing the logger)
try:
    # Use wandb login from CLI or environment variable for API key for better practice
    # wandb.login(key="YOUR_KEY") # Replace with your key if needed, but CLI/env var is preferred
    wandb_logger = pl.loggers.WandbLogger(project="graph-line-regression", log_model="all") # Changed project name slightly
    logging.info("Wandb logger initialized.")
except Exception as e:
    logging.error(f"Failed to initialize Wandb logger: {e}. Training will proceed without Wandb logging.")
    wandb_logger = None # Set logger to None to disable logging


# Check input dimension from the first graph
# It's safer to get this dynamically after data loading
if len(dataset) > 0:
    input_feature_dim = dataset[0].x.shape[1]
    logging.info(f"Determined input feature dimension: {input_feature_dim}")
else:
    logging.error("Cannot determine input feature dimension, dataset is empty.")
    exit()

model = GATRegressor(
        input_dim=input_feature_dim, # Use dynamically determined dim
        cnn_output_dim=128,
        hidden_dim=64,
        n_heads=4,
        n_layers=3,
        dropout=0.1,
        learning_rate=5e-3
    )

checkpoint_callback = ModelCheckpoint(
    dirpath="./checkpoints",
    filename="gat-line-{epoch}-{val_loss:.2f}", # Include val_loss in name
    save_top_k=1,        # Save top 2 models based on monitored metric
    monitor="val_loss",  # Monitor validation loss
    mode="min",          # Save models with minimum validation loss
    save_last=True       # Also save the last epoch's checkpoint
)

trainer = pl.Trainer(
        logger=wandb_logger, # Use the logger instance (can be None)
        max_epochs=100,
        accelerator='cpu', # Automatically choose accelerator (mps, cpu, gpu)
        # gradient_clip_val=1.0, # Added gradient clipping
        log_every_n_steps=2, # Log slightly less often
        callbacks=[checkpoint_callback],
        # precision="16-mixed" # Optional: Use mixed precision if running on GPU for speed/memory
        # detect_anomaly=True # Optional: Enable anomaly detection during training for debugging NaNs
    )

# --- Run Training ---
logging.info("Starting training...")
try:
    trainer.fit(model, datamodule=data_module)
    logging.info("Training finished.")
except Exception as e:
    logging.error(f"An error occurred during training: {e}", exc_info=True) # Log traceback


if data_module.test_dataloader() is not None:
    logging.info("Starting testing...")
    try:
        trainer.test(model, datamodule=data_module)
        logging.info("Testing finished.")
    except Exception as e:
        logging.error(f"An error occurred during testing: {e}", exc_info=True)
else:
    logging.info("No test dataset available, skipping testing.")


# --- Finish Wandb ---
if wandb_logger is not None:
    wandb.finish()
    logging.info("Wandb run finished.")

logging.info("Script execution complete.")


