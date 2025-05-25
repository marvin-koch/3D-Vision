import numpy as np
import cv2
import torch
from torchvision.ops import roi_align
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import Union, List, Tuple

def sample_lines_grid(
    img: torch.Tensor,
    lines: Union[torch.Tensor, List[Tuple[Tuple[float,float], Tuple[float,float]]]],
    num_samples: int = 100,
    width: float = 1.0,
    # num_width_samples parameter is removed
    align_corners: bool = True
) -> torch.Tensor:
    """
    Samples strips of a specified pixel `width` around multiple lines
    defined by start and end points in `img`, using bilinear interpolation
    via grid_sample. The number of samples across the width is determined
    by the integer value of the `width` parameter.

    Args:
        img:           (C,H,W) or (B,C,H,W) image tensor.
        lines:         A list of N ((x0,y0),(x1,y1)) tuples, or a tensor
                       of shape (N, 2, 2) where lines[i,0,:] is the start
                       and lines[i,1,:] is the end of the i-th line in
                       pixel coordinates.
        width:         Total thickness (pixels) orthogonal to the lines. The number
                       of samples across this width will be int(round(width)),
                       ensuring at least 1 sample.
        num_samples:   # points along each line segment.
        align_corners: Forwarded to grid_sample.

    Returns:
        If input img was (C,H,W), returns (N, C, num_samples, n_width_samples).
        If input img was (B,C,H,W), returns (B, N, C, num_samples, n_width_samples).
        Where N is the number of lines and n_width_samples = max(1, int(round(width))).
    """
    # Determine the number of width samples based on the width parameter
    # Use round() for sensible integer conversion, max(1, ...) ensures at least one sample.
    n_width_samples = int(max(1, round(width)))

    # ensure batch‐dim for image
    batched_input = True
    if img.dim() == 3:
        img = img.unsqueeze(0)
        batched_input = False
    B, C, H, W = img.shape


    # # --- Input Line Processing ---
    # if isinstance(lines, list):
    #     if not lines:
    #         # Handle empty list case
    #         # Use n_width_samples determined above
    #         out_shape = (B, 0, C, num_samples, n_width_samples) if batched_input else (0, C, num_samples, n_width_samples)
    #         return torch.empty(out_shape, dtype=img.dtype, device=device)
    #     # Convert list of tuples to tensor (N, 2, 2) - ensure correct device
    #     lines_tensor = torch.tensor(lines, dtype=torch.float32, device=device)
    # elif isinstance(lines, torch.Tensor):
    #     # Ensure correct device and dtype
    #     lines_tensor = lines.to(device=device, dtype=torch.float32)
    #     if lines_tensor.dim() != 3 or lines_tensor.shape[1:] != (2, 2):
    #          raise ValueError(f"lines tensor must have shape (N, 2, 2), but got {lines_tensor.shape}")
    # else:
    #     raise TypeError("lines must be a list of line tuples or a tensor(N, 2, 2)")

    # if lines_tensor.numel() == 0:
    #     # Handle empty tensor case
    #     # Use n_width_samples determined above
    #     out_shape = (B, 0, C, num_samples, n_width_samples) if batched_input else (0, C, num_samples, n_width_samples)
    #     return torch.empty(out_shape, dtype=img.dtype, device=device)

    N = lines.shape[0] # Number of lines

    # --- Vectorized Line Parameter Calculation ---
    starts = lines[:, 0, :]  # Shape: (N, 2)
    ends = lines[:, 1, :]    # Shape: (N, 2)

    x0 = starts[:, 0]  # Shape: (N,)
    y0 = starts[:, 1]  # Shape: (N,)
    x1 = ends[:, 0]    # Shape: (N,)
    y1 = ends[:, 1]    # Shape: (N,)

    dx = x1 - x0       # Shape: (N,)
    dy = y1 - y0       # Shape: (N,)

    # Add epsilon to prevent division by zero for zero-length lines
    length = torch.sqrt(dx*dx + dy*dy).clamp(min=1e-6) # Shape: (N,)
    ux = dx / length   # unit direction x, Shape: (N,)
    uy = dy / length   # unit direction y, Shape: (N,)

    # Perpendicular unit vector
    px = -uy           # Shape: (N,)
    py = ux            # Shape: (N,)

    # --- Create Sampling Grid Parameters ---
    # t: samples along the line segment
    t = torch.linspace(0, 1, steps=num_samples,) # Shape: (num_samples,)

    # s: samples orthogonally across the specified 'width'
    # The number of steps is now determined by n_width_samples derived from 'width'
    # The range remains centered, spanning 'width' pixels.
    s = torch.linspace(-width/2, width/2, steps=n_width_samples,) # Shape: (n_width_samples,)

    # Create meshgrid for sampling points relative to each line
    # tt: varies along the line, ss: varies across the width
    tt, ss = torch.meshgrid(t, s, indexing='ij') # Shape: (num_samples, n_width_samples)

    # --- Vectorized Absolute Coordinate Calculation ---
    # Use broadcasting to calculate coordinates for all N lines simultaneously
    # Resulting X, Y shape: (N, num_samples, n_width_samples)
    X = x0.view(N, 1, 1) + dx.view(N, 1, 1) * tt + px.view(N, 1, 1) * ss
    Y = y0.view(N, 1, 1) + dy.view(N, 1, 1) * tt + py.view(N, 1, 1) * ss

    # --- Normalize Coordinates to [-1, +1] ---
    Xn = 2 * X / (W - 1) - 1  # Shape: (N, num_samples, n_width_samples)
    Yn = 2 * Y / (H - 1) - 1  # Shape: (N, num_samples, n_width_samples)

    # Stack to create the grid for grid_sample
    # Shape: (N, num_samples, n_width_samples, 2)
    grid = torch.stack([Xn, Yn], dim=-1)

    # --- Prepare for grid_sample ---
    # Grid shape is now (N, S, nWs, 2)
    # Repeat grid B times: (B*N, S, nWs, 2)
    grid = grid.repeat(B, 1, 1, 1)

    # Repeat image N times: (B*N, C, H, W)
    img_rep = img.repeat_interleave(N, dim=0)

    # --- Perform Sampling ---
    # out shape: (B*N, C, num_samples, n_width_samples)
    out = F.grid_sample(
        img_rep, grid,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=align_corners
    )

    # --- Reshape Output ---
    # Reshape to (B, N, C, num_samples, n_width_samples)
    out = out.view(B, N, C, num_samples, n_width_samples)

    if not batched_input:
        # Remove the original batch dim if input was single image
        out = out.squeeze(0) # Shape: (N, C, num_samples, n_width_samples)

    return out




def extract_line_feature_ROIAlign(img: np.ndarray, 
                                lines: np.ndarray, 
                                output_size: tuple = (32, 32), 
                                margin: int = 5,
                                plot_results: bool = False):
    """
    Extracts features along multiple line segments in an image using ROIAlign.

    Args:
        img: Input image as a NumPy array (H, W) or (H, W, C).
        lines: NumPy array of shape (N, 2, 2) containing N lines, 
               where each line is [[x1, y1], [x2, y2]].
        output_size: The desired output size (height, width) for each ROI.
        margin: Padding added around the bounding box of each line.
        plot_results: If True, displays the original image with lines/boxes 
                      and the corresponding ROIAlign results.

    Returns:
        torch.Tensor: A tensor of shape (N, C, output_size[0], output_size[1]) 
                      containing the extracted features for each line ROI.
                      Returns None if lines is empty.
    """
    if lines is None or len(lines) == 0:
        return None
        
    H, W = img.shape[:2]
    
    # Ensure lines is a numpy array
    lines_np = np.array(lines)
    if lines_np.ndim != 3 or lines_np.shape[1:] != (2, 2):
        raise ValueError("lines must be a NumPy array of shape (N, 2, 2)")
        
    num_lines = lines_np.shape[0]

    # --- Vectorized Bounding Box Calculation ---
    # Shape: (N, 2, 2) -> x_coords (N, 2), y_coords (N, 2)
    x_coords = lines_np[..., 0] 
    y_coords = lines_np[..., 1]

    # Calculate min/max for each line (axis=1) -> Shape (N,)
    min_x = np.maximum(0, np.floor(x_coords.min(axis=1)) - margin).astype(int)
    min_y = np.maximum(0, np.floor(y_coords.min(axis=1)) - margin).astype(int)
    max_x = np.minimum(W - 1, np.ceil(x_coords.max(axis=1)) + margin).astype(int)
    max_y = np.minimum(H - 1, np.ceil(y_coords.max(axis=1)) + margin).astype(int)

    # --- Prepare Image Tensor ---
    # Handle grayscale: add channel dim if needed (H, W) -> (H, W, 1)
    if img.ndim == 2:
        img_proc = np.expand_dims(img, axis=-1)
    else:
        img_proc = img.copy()

    # Convert to float32 tensor (C, H, W) and add batch dim (1, C, H, W)
    input_tensor = torch.from_numpy(img_proc.astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
    C = input_tensor.shape[1] # Number of channels

    # --- Prepare Boxes Tensor ---
    # Create bounding box tensor (N, 4) -> [x1, y1, x2, y2]
    # Note: roi_align expects [x1, y1, x2, y2] (inclusive coordinates)
    box_coords = torch.tensor(np.stack([min_x, min_y, max_x, max_y], axis=1), dtype=torch.float32)

    # Add batch index (all zeros since we have one image) -> Shape (N, 1)
    batch_indices = torch.zeros((num_lines, 1), dtype=torch.float32)

    # Combine batch indices and coordinates -> Shape (N, 5) [batch_idx, x1, y1, x2, y2]
    boxes_tensor = torch.cat([batch_indices, box_coords], dim=1)

    # --- Apply ROIAlign ---
    # Output shape: (N, C, output_size[0], output_size[1])
    roi_results = roi_align(input_tensor, boxes_tensor, output_size=output_size, spatial_scale=1.0, aligned=True)

    # --- Optional Plotting ---
    if plot_results:
        # Determine grid size (e.g., max 3x3 grid)
        num_plots = min(num_lines, 9)
        grid_size = int(np.ceil(np.sqrt(num_plots)))
        fig, axes = plt.subplots(grid_size, grid_size * 2, figsize=(grid_size * 4, grid_size * 2))
        axes = axes.ravel() # Flatten axes array for easy iteration

        img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
        
        # Plot original image with all boxes/lines once if grid is large enough
        if len(axes) > num_plots * 2:
             ax_overview = axes[num_plots * 2]
             for i in range(num_lines):
                 pt1 = tuple(np.round(lines_np[i, 0]).astype(int))
                 pt2 = tuple(np.round(lines_np[i, 1]).astype(int))
                 cv2.line(img_display, pt1, pt2, color=(0, 0, 255), thickness=1) # Blue line
                 cv2.rectangle(img_display, (min_x[i], min_y[i]), (max_x[i], max_y[i]), color=(0, 255, 0), thickness=1) # Green box
             ax_overview.imshow(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB) if img_display.shape[-1] == 3 else img_display, cmap='gray')
             ax_overview.set_title("All Lines & Boxes")
             ax_overview.axis('off')
             # Turn off remaining unused axes
             for k in range(num_plots * 2 + 1, len(axes)):
                 axes[k].axis('off')


        for i in range(num_plots):
            ax1_idx = i * 2
            ax2_idx = i * 2 + 1
            
            if ax1_idx < len(axes):
                # Plot 1: Original image with current line and box highlighted
                img_single_highlight = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
                pt1 = tuple(np.round(lines_np[i, 0]).astype(int))
                pt2 = tuple(np.round(lines_np[i, 1]).astype(int))
                cv2.line(img_single_highlight, pt1, pt2, color=(0, 0, 255), thickness=2) # Thicker blue line
                cv2.rectangle(img_single_highlight, (min_x[i], min_y[i]), (max_x[i], max_y[i]), color=(0, 255, 0), thickness=2) # Thicker green box
                
                axes[ax1_idx].imshow(cv2.cvtColor(img_single_highlight, cv2.COLOR_BGR2RGB) if img_single_highlight.shape[-1] == 3 else img_single_highlight, cmap='gray')
                axes[ax1_idx].set_title(f"Line {i}")
                axes[ax1_idx].axis('off')

            if ax2_idx < len(axes):
                # Plot 2: ROIAlign result
                roi_img = roi_results[i].permute(1, 2, 0).cpu().numpy() # (H, W, C)
                # Handle single channel output for grayscale display
                if roi_img.shape[2] == 1:
                    roi_img = roi_img.squeeze(axis=2)
                axes[ax2_idx].imshow(roi_img, cmap='gray' if roi_img.ndim == 2 else None)
                axes[ax2_idx].set_title(f"ROI {i} Result")
                axes[ax2_idx].axis('off')

        # Hide any remaining unused axes pairs if num_plots < grid_size*grid_size
        for i in range(num_plots * 2, grid_size*grid_size*2):
             if i < len(axes): # Make sure index is valid
                  axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

    return roi_results
