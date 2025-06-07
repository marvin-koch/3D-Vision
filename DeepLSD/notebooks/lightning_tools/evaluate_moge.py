import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader as PyGDataLoader
import cv2
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    roc_curve, auc, average_precision_score, f1_score, confusion_matrix
)
from dataset_inductive import GraphDatasetInductive  # import your dataset with groundtruth loader


import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score
import random
def plot_coplanar_lines(ax, lines, labels, image):
    """
    Visualize lines on an image with colors corresponding to their plane labels.
    Outliers (label -1) are drawn in grey. Designed to be used with a subplot axis.
    """
    unique_labels = sorted(set(labels))
    num_clusters = len(unique_labels)

    # Generate random colors for clusters (excluding -1 if present)
    random.seed(52)
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

    # ax.set_title("Coplanar Lines")
    ax.axis('off')
    
def cluster_lines_hdbscan(distance_matrix: np.ndarray,
                          min_cluster_size: int = 2,
                          min_samples: int = None,
                          scan_min_size: bool = False,
                          size_grid: int = 10,
                          verbose: bool = False):
    """
    Cluster an NxN coplanarity-distance matrix using HDBSCAN.

    Parameters
    ----------
    distance_matrix : (N, N) ndarray
        Symmetric, zero diagonal.
    min_cluster_size : int, default 2
        The minimum size of clusters; passed to HDBSCAN.
    min_samples : int | None, default None
        The number of samples in a neighborhood for a point to be considered
        a core point. If None, uses the same value as min_cluster_size.
    scan_min_size : bool, default False
        If True, will try `size_grid` different min_cluster_size values between
        `min_cluster_size` and ⌈√N⌉+1, and pick the one maximizing silhouette.
    size_grid : int, default 10
        Number of min_cluster_size values to try if scan_min_size is True.
    verbose : bool, default False
        Print silhouette scores for each tried size.

    Returns
    -------
    labels : (N,) ndarray of int
        Cluster labels (noise = -1).
    best_size : int
        The min_cluster_size that was used.
    best_silhouette : float | None
        The silhouette score (None if all points noise or only one cluster).
    """
    D = np.asarray(distance_matrix, dtype=float)
    if D.shape[0] != D.shape[1]:
        raise ValueError("Distance matrix must be square")
    N = D.shape[0]

    # helper to fit & score
    def fit_and_score(mcs):
        model = hdbscan.HDBSCAN(
            metric='precomputed',
            min_cluster_size=mcs,
            min_samples=min_samples or mcs
        )
        labels = model.fit_predict(D)
        # need at least 2 non-noise clusters for silhouette
        if len(set(labels) - {-1}) < 2:
            return labels, None
        score = silhouette_score(D, labels, metric='precomputed')
        return labels, score

    # if not scanning, just do one run
    if not scan_min_size:
        labels, score = fit_and_score(min_cluster_size)
        return labels, min_cluster_size, score

    # scan over a grid of sizes
    max_size = int(np.sqrt(N)) + 1
    sizes = np.unique(
        np.linspace(min_cluster_size, max_size, size_grid, dtype=int)
    )
    best_score = -1.0
    best = (None, None)
    for mcs in sizes:
        labels, score = fit_and_score(mcs)
        if verbose:
            print(f"min_cluster_size={mcs}, silhouette={score}")
        if score is not None and score > best_score:
            best_score = score
            best = (labels, mcs)
    if best[0] is None:
        raise RuntimeError("HDBSCAN found fewer than 2 clusters for all tried sizes.")
    return best[0], best[1], best_score
def evaluate(
    json_dir: str,
    batch_size: int = 1,
    num_workers: int = 4,
    threshold_struct: float = 0.5,
    threshold_coplanar: float = 0.5,
    plot=False

):
    # 1) Load model

    # 2) Prepare dataset & loader

    dataset = GraphDatasetInductive(
        h5_path="/mnt/c/Users/shan2/Documents/ETH/MA4/3DV/GitHub/3D-Vision/DeepLSD/notebooks/diode_data.h5",
        roi_output_size=[64,32],
        method="sample",
        edge_sample_size=[32,24],
        augment=False,
    )

    dataset_moge = GraphDatasetInductive(
        h5_path="/mnt/c/Users/shan2/Documents/ETH/MA4/3DV/GitHub/3D-Vision/DeepLSD/notebooks/midas_data.h5",
        roi_output_size=[64,32],
        method="sample",
        edge_sample_size=[32,24],
        augment=False,
    )
    
    
    print("Datset size: ", len(dataset))
    loader = PyGDataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
    loader_moge = PyGDataLoader(dataset_moge, batch_size=batch_size, num_workers=num_workers)


    # 3) Accumulators
    all_node_moge, all_node_labels = [], []
    all_edge_moge, all_edge_labels = [], []
    keep_idxs= []

    # 4) Inference loop
    with torch.no_grad():
        for idx, data in enumerate(loader):
       


            N = data.coordinates.shape[0]

            # Reshape the flattened (N²,) arrays → (N, N)
            edge_labels_array = data.full_edge_labels.cpu().numpy().reshape((N, N))
 
            # 1) Compute row‐sums of the ground‐truth adjacency:
            row_sums = edge_labels_array.sum(axis=1)

            # 2) Build a mask of “keep” indices: only lines i with row_sums[i] > 0
            keep_mask = (row_sums > 0)
            kept_idx = np.where(keep_mask)[0]
            
            labels_sub = edge_labels_array[np.ix_(kept_idx, kept_idx)].ravel()

            # 4) Append filtered‐flattened to accumulators
            all_edge_labels.append(labels_sub)
            keep_idxs.append(kept_idx)
            
            all_node_labels.append(data.y.cpu().numpy().ravel())
            # all_edge_preds.append(edge_probs)
            # all_edge_labels.append(data.full_edge_labels.cpu().numpy().ravel())

        for idx, data in enumerate(loader_moge):
    


            N = data.coordinates.shape[0]

            # Reshape the flattened (N²,) arrays → (N, N)
            edge_labels_array = data.full_edge_labels.cpu().numpy().reshape((N, N))
 
            # 1) Compute row‐sums of the ground‐truth adjacency:
            row_sums = edge_labels_array.sum(axis=1)

            # 2) Build a mask of “keep” indices: only lines i with row_sums[i] > 0
            kept_idx = keep_idxs[idx]
            
            labels_sub = edge_labels_array[np.ix_(kept_idx, kept_idx)].ravel()

            # 4) Append filtered‐flattened to accumulators
            all_edge_moge.append(labels_sub)
            
            all_node_moge.append(data.y.cpu().numpy().ravel())
            # all_edge_preds.append(edge_probs)
            # all_edge_labels.append(data.full_edge_labels.cpu().numpy().ravel())
   
   
    all_node_moge = np.concatenate(all_node_moge)
    all_node_labels = np.concatenate(all_node_labels)
    all_edge_moge = np.concatenate(all_edge_moge)
    all_edge_labels = np.concatenate(all_edge_labels)
    
   

    
  
    # 5) Compute metrics
    node_metrics = compute_metrics(all_node_labels, all_node_moge, 'node', threshold_struct, invert=True)
    edge_metrics = compute_metrics(all_edge_labels, all_edge_moge, 'edge', threshold_coplanar)

    # 6) Print all results
    print("=== Node Metrics ===")
    for k, v in node_metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\n=== Edge Metrics ===")
    for k, v in edge_metrics.items():
        print(f"  {k}: {v:.4f}")

    combined_auc = 0.5 * (node_metrics['node_roc_auc'] + edge_metrics['edge_roc_auc'])
    print(f"\nCombined ROC‐AUC: {combined_auc:.4f}")

def compute_metrics(y_true, y_prob, prefix, threshold, invert = False):
    """
    Utility to compute accuracy, recall, precision, specificity, F1, ROC‐AUC, PR‐AUC, and confusion‐matrix counts.
    """
    if invert:
        y_true = 1-y_true
        y_prob = 1- y_prob
        threshold = 1 -threshold
        

        
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    
    acc   = accuracy_score(y_true, y_pred)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    spec  = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1    = f1_score(y_true, y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    pr_auc  = average_precision_score(y_true, y_prob)
    return {
        f"{prefix}_accuracy":    acc,
        f"{prefix}_recall":      rec,
        f"{prefix}_precision":   prec,
        f"{prefix}_specificity": spec,
        f"{prefix}_f1":          f1,
        f"{prefix}_roc_auc":     roc_auc,
        f"{prefix}_pr_auc":      pr_auc,
        f"{prefix}_tp":          tp,
        f"{prefix}_tn":          tn,
        f"{prefix}_fp":          fp,
        f"{prefix}_fn":          fn,
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate model on ScanNet-derived ground truth')
    # parser.add_argument('--ckpt', default= "/Users/marvin/Documents/ETHZ/MA4/3DVision/repo/DeepLSD/notebooks/lightning_tools/lightning_logs/lightning_project/edgesampling7000/checkpoints/best-model-epoch=11-val_combined_auc_epoch=0.8713.ckpt", help='Path to trained model checkpoint')
    # parser.add_argument('--ckpt', default= "/Users/marvin/Documents/ETHZ/MA4/3DVision/repo/DeepLSD/notebooks/lightning_tools/lightning_logs/lightning_project/vbqkdl9g/checkpoints/best-model-epoch=04-val_edge_pr_auc_epoch=0.6855.ckpt", help='Path to trained model checkpoint')

    parser.add_argument('--json_dir',  default='../diode_output')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--threshold_struct', type=float, default=0.5)
    parser.add_argument('--threshold_coplanar', type=float, default=0.75)

    args = parser.parse_args()

    evaluate(
        json_dir=args.json_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
        threshold_struct=args.threshold_struct,
        threshold_coplanar=args.threshold_coplanar,
        plot=False

    )
