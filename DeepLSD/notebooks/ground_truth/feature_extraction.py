import torch

df_intermediate_features = None
angle_intermediate_features = None

def hook_df(module, input, output):
    global df_intermediate_features
    df_intermediate_features = output.detach()

def hook_angle(module, input, output):
    global angle_intermediate_features
    angle_intermediate_features = output.detach()
    
    
def sample_line_features(feature_map, line, num_samples=10, downsample_ratio=None):
    """
    Sample and concatenate the features from each sample point into a single vector.

    Args:
        feature_map (torch.Tensor): Combined feature map of shape (B, C, H, W).
        line (array-like): A 2x2 array where line[0] is (x1, y1) and line[1] is (x2, y2)
                            in the original image coordinate system.
        num_samples (int): Number of points sampled along the line.
        downsample_ratio (float): The factor by which the original image is downsampled.
        
    Returns:
        concatenated_feature (torch.Tensor): Concatenated embedding for the line,
                                                of shape (C * num_samples,).
    """
    # Unpack endpoints (each endpoint is [x, y])
    (x1, y1), (x2, y2) = line
    # Convert coordinates to match the feature map resolution.
    x1, y1, x2, y2 = x1 / downsample_ratio, y1 / downsample_ratio, x2 / downsample_ratio, y2 / downsample_ratio

    # Create uniformly spaced sampling points along the line using linear interpolation.
    t_vals = torch.linspace(0, 1, steps=num_samples, device=feature_map.device)
    xs = x1 + t_vals * (x2 - x1)
    ys = y1 + t_vals * (y2 - y1)

    # Build a sampling grid; grid_sample expects normalized coordinates in [-1, 1].
    grid = torch.stack([xs, ys], dim=-1)  # shape: (num_samples, 2)
    # Get spatial dimensions of the feature map (assumed shape: (B, C, H, W))
    _, _, H, W = feature_map.shape
    grid[..., 0] = (grid[..., 0] / (W - 1)) * 2 - 1  # Normalize x coordinates.
    grid[..., 1] = (grid[..., 1] / (H - 1)) * 2 - 1  # Normalize y coordinates.
    grid = grid.unsqueeze(0).unsqueeze(2)  # Reshape to (1, num_samples, 1, 2) for grid_sample

    # Use bilinear interpolation to sample the feature map.
    # This returns a tensor of shape (B, C, num_samples, 1)
    sampled_features = torch.nn.functional.grid_sample(feature_map, grid, align_corners=True)
    # Remove the extra dimension to obtain shape (B, C, num_samples)
    sampled_features = sampled_features.squeeze(-1)

    # Instead of aggregating via mean, we concatenate the features at the sampled points.
    # Reshape sampled_features from shape (B, C, num_samples) to (B, C * num_samples)
    concatenated_feature = sampled_features.view(sampled_features.shape[0], -1)

    # Remove the batch dimension if batch_size == 1, resulting in a tensor of shape (C * num_samples,).
    return concatenated_feature.squeeze(0)