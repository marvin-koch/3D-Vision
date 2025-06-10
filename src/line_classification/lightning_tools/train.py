# train_lightning.py
import os
import torch
import pytorch_lightning as pl
from lightning.pytorch.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import yaml
import argparse

from lightning_datamodule import GraphDataModuleInductive
from gluestick_fully_linear import *

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


    cfg_data = cfg.get('data', {})
    cfg_model = cfg.get('model', {})
    cfg_train = cfg.get('training', {})

    # --- Set Seed ---
    pl.seed_everything(cfg_data.get('seed', 42), workers=True)

    # --- Data ---
    print("Setting up DataModule with batch_size {} and accum_grad {}.".format(cfg_data.get('batch_size', 1),cfg_data.get('accum_grads', 1)))
    print("Effective batch_size is {}".format(cfg_data.get('batch_size', 1) * cfg_data.get('accum_grads', 1)))
    roi_output_size_tuple = tuple(cfg_data.get('roi_output_size', [64, 64]))
    num_workers = cfg_data.get('num_workers', 0)
    # Automatically set num_workers to 0 if no CUDA detected to avoid potential issues
    use_precomputed_data = cfg_data.get('use_precomputed', False)
    print("use_precomputed_data is {}".format(use_precomputed_data))
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
  
    # --- Model ---
    print("Initializing Model...")
    jk_layer_val = cfg_model.get('jk_layer')
    if isinstance(jk_layer_val, str) and jk_layer_val.lower() == 'null':
        jk_layer_val = None
   
    model = AttentionEdgeSampleLinearAverage(
        in_channels_DeepLSD=cfg_model.get('in_channels_DeepLSD', 1280),
        in_channels=cfg_model.get('in_channels', 1024),
        hidden_channels=cfg_model.get('hidden_channels', 128),
        out_channels=cfg_model.get('out_channels', 64),
        geom_channels=cfg_model.get('geom_channels', 5),
        roi_align_embedding_shape=roi_output_size_tuple,
        num_layers=cfg_model.get('num_layers', 3),
        dropout=cfg_model.get('dropout', 0.2),
        act=cfg_model.get('act', 'relu'),
        v2=cfg_model.get('v2', True),
        jk_layer=jk_layer_val,
        learning_rate=cfg_train.get('learning_rate', 1e-3),
        node_loss_w=cfg_train.get('node_loss_w', 1.0),
        edge_loss_w=cfg_train.get('edge_loss_w', 1.0),
        threshold_structural=cfg_train.get('threshold_structural', 0.5),
        mlp_dropout=cfg_model.get('mlp_dropout',0.0),
        skip_init=cfg_model.get('skip_init', False),
        edge_sample_size=cfg_data.get('edge_sample_size', (32,8)),
        edge_downsample_dim=cfg_model.get('edge_downsample_dim', 20),
   
    )
    
    model = torch.compile(model)
    

    # --- Logging and Checkpointing ---
    log_dir = cfg_train.get('log_dir', "lightning_logs")
    os.makedirs(log_dir, exist_ok=True)

    # Use WandbLogger
    logger = WandbLogger(
        name=cfg_train.get('experiment_name', "gat_yaml_config"), # Run name
        save_dir=log_dir,                                         # Local directory for wandb files
        project=cfg_train.get('wandb_project', "lightning_project"), # W&B project name (REQUIRED)
        entity=cfg_train.get('wandb_entity', None),               # Optional: W&B username or team
        log_model=True,                                           # Optional: Log model checkpoints to W&B
        # offline=False # Set to True to log locally without syncing to W&B servers
    )
    logger.log_hyperparams(cfg)
    monitor_metric = cfg_train.get('monitor_metric', 'val_auc')
    monitor_mode = cfg_train.get('monitor_mode', 'max')
    checkpoint_callback = ModelCheckpoint(
        # monitor=monitor_metric,
        # mode=monitor_mode,
        filename=f'best-model-{{epoch:02d}}-{{{monitor_metric}:.4f}}',
        save_top_k=-1,
        every_n_epochs=1,
        verbose=True
    )
    early_stop_callback = EarlyStopping(
        monitor=monitor_metric,
        patience=cfg_train.get('early_stopping_patience', 5),
        mode=monitor_mode,
        verbose=True
    )

    # --- Resolve Accelerator and Devices ---
    # Get desired settings from config
    accelerator_cfg = cfg_train.get('accelerator', 'auto')
    devices_cfg = cfg_train.get('devices', 'auto')

    # Determine actual accelerator (handles 'auto')
    if accelerator_cfg == 'auto':
        if torch.cuda.is_available():
            actual_accelerator = 'gpu'

        else:
            actual_accelerator = 'cpu'
    else:
        actual_accelerator = accelerator_cfg # Use the specific value from config

    # Determine actual devices based on accelerator
    if actual_accelerator == 'cpu':
        # CPUAccelerator requires devices to be an int > 0
        actual_devices = 1
        print(f"Info: Accelerator resolved to CPU. Setting devices=1.")
    else:
        # For GPU, MPS, etc., 'auto' or specific list/int is usually fine
        actual_devices = devices_cfg
        print(f"Info: Accelerator resolved to {actual_accelerator}. Using devices='{actual_devices}'.")


    # --- Trainer ---
    print("Initializing Trainer...")
    trainer = pl.Trainer(
        max_epochs=cfg_train.get('epochs', 20),
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator=actual_accelerator, # Use resolved accelerator
        devices=actual_devices,       # Use resolved devices
        log_every_n_steps=10,
        deterministic=cfg_train.get('seed', 42) is not None, # Enable deterministic if seed is set
        accumulate_grad_batches=cfg_data.get('accum_grads', 1),
        gradient_clip_val=1.0
    )

    resume_from_ckpt_path_config = cfg_model.get("load_model_path", None)
    ckpt_to_resume_for_fit = None # Default to None (start training from scratch)

    if resume_from_ckpt_path_config and os.path.isfile(resume_from_ckpt_path_config):
        print(f"Found model checkpoint at '{resume_from_ckpt_path_config}'. Attempting to load and continue training.")
        ckpt_to_resume_for_fit = resume_from_ckpt_path_config
    elif resume_from_ckpt_path_config: # Path was specified in config, but file does not exist
        print(f"Warning: Model checkpoint path '{resume_from_ckpt_path_config}' provided in config but file not found. Starting training from scratch.")
    else: # No load_model_path specified in config for resuming
        print("No resume model checkpoint path provided in cfg_model.load_model_path. Starting training from scratch.")
    
    # --- Training ---
    print("\n--- Starting Training ---")
    trainer.fit(model, datamodule=data_module, ckpt_path=ckpt_to_resume_for_fit)
    print("--- Training Finished ---")
    print(f"Best model saved at: {checkpoint_callback.best_model_path}")

    # --- Testing ---
    print("\n--- Starting Testing (using best model) ---")
    test_results = trainer.test(model, datamodule=data_module, ckpt_path='best')
    print("Test Results:", test_results)
    print("--- Testing Finished ---")

    # --- Plotting ---
    if hasattr(model, 'test_results') and model.test_results.get('labels') is not None:
        print("\n--- Plotting Test ROC Curve ---")
        log_dir_path = logger.log_dir or os.path.join(cfg_train.get('log_dir', "lightning_logs"), cfg_train.get('experiment_name', "gat_yaml_config"))
        os.makedirs(log_dir_path, exist_ok=True)
        roc_save_path = os.path.join(log_dir_path, 'final_test_roc.png')

        plot_roc_curve(
            model.test_results['labels'],
            model.test_results['preds'],
            title=f'Test ROC Curve ({cfg_train.get("experiment_name", "default")})',
            save_path=roc_save_path
        )
    else:
        print("Test results not available for ROC curve plotting.")


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