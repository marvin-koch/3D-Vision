import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Union, List, Tuple
import numpy as np
import cv2
from torchvision.ops import roi_align
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from typing import Union, List, Tuple


class LineSampler(pl.LightningModule):
    def __init__(
        self,
        num_samples: int = 100,
        width: float = 1.0,
    ):
        """
        Initializes the LineSampler.

        Args:
            num_samples: Number of points to sample along the line's length.
            width: Width of the strip to sample around the line, in pixels.
                   Values > 1 will result in multiple samples across the width.
        """
        super().__init__()
        self.num_samples = num_samples
        self.width = width

        # Precompute and cache the local sampling grid (t, s)
        # n_width determines how many discrete steps across the width are sampled
        n_width = max(1, round(width))
        t = torch.linspace(0, 1, steps=num_samples)
        # If width=1, n_width=1, s will be just [0.]
        # If width=3, n_width=3, s will be [-1., 0., 1.] (approx, depends on linspace)
        s = torch.linspace(-width / 2.0, width / 2.0, steps=n_width)
        tt, ss = torch.meshgrid(t, s, indexing='ij') # tt:(Ns, Nw), ss:(Ns, Nw)

        # Register as buffers so they move with the module’s device/dtype
        # and are saved with the state_dict
        self.register_buffer('tt', tt)
        self.register_buffer('ss', ss)

    def sample_lines_grid(
        self,
        img: torch.Tensor,
        lines: Union[torch.Tensor, List[Tuple[Tuple[float, float], Tuple[float, float]]]],
        align_corners: bool = True
    ) -> torch.Tensor:
        """
        Samples strips of pixel width around multiple lines from a single image
        using the cached (t,s) grid.

        Args:
            img: Input image tensor (C, H, W).
            lines: Lines defined by start/end points.
                   Can be a tensor of shape (N, 2, 2) or a list of
                   N tuples ((x0,y0), (x1,y1)). Coordinates should be
                   in pixel space (x corresponding to W, y to H).
            align_corners: Argument passed to F.grid_sample. If True, the
                           extreme values -1 and 1 are considered pixel centers.
                           If False, they are considered pixel corners.

        Returns:
            Tensor containing the sampled strips, shape (N, C, num_samples, n_width),
            where n_width is max(1, round(self.width)).
        """
        # --- Input Validation and Preparation ---
        if img.dim() != 3:
            raise ValueError(f"Input image must be a 3D tensor (C, H, W), but got shape {img.shape}")
        if lines.dim() != 3 or lines.shape[1:] != (2, 2):
             raise ValueError(f"Lines tensor must have shape (N, 2, 2), but got {lines.shape}")

        C, H, W = img.shape

        # Determine number of width samples from cached grid
        n_width = self.ss.shape[1]
        
        N = lines.shape[0] # Number of lines

        # --- Coordinate Calculation ---
        # Extract start/end points and compute line vectors
        starts = lines[:, 0, :]  # (N, 2) [x0, y0]
        ends   = lines[:, 1, :]  # (N, 2) [x1, y1]
        x0, y0 = starts.unbind(-1) # (N,), (N,)
        x1, y1 = ends.unbind(-1)   # (N,), (N,)

        # Line direction vector (v = end - start)
        dx, dy = x1 - x0, y1 - y0 # (N,), (N,)
        length = (dx*dx + dy*dy).sqrt().clamp(min=1e-6) # (N,) Avoid division by zero

        # Perpendicular vector (p). Rotated 90 degrees from direction vector (ux, uy)
        # ux = dx / length, uy = dy / length
        # px = -uy, py = ux
        px = -dy / length # (N,)
        py =  dx / length # (N,)

        # Use cached (t,s) grid. Ensure s has the correct dtype for calculations.
        tt = self.tt                # (num_samples, n_width) - Parameter t (along line)
        ss = self.ss.to(img.dtype)  # (num_samples, n_width) - Parameter s (across line width)

        # Map local (t,s) coordinates to absolute image coordinates (X, Y)
        # Formula: point = start + t * direction_vector + s * perpendicular_vector
        # Unsqueeze N dim for broadcasting: (N,1,1) op (Ns,Nw) -> (N,Ns,Nw)
        X = x0.view(N,1,1) + tt * dx.view(N,1,1) + ss * px.view(N,1,1) # (N, Ns, Nw)
        Y = y0.view(N,1,1) + tt * dy.view(N,1,1) + ss * py.view(N,1,1) # (N, Ns, Nw)

        # Normalize coordinates to the range [-1, +1] for grid_sample
        # Note: W-1 and H-1 are used because pixel coordinates range from 0 to W-1 / H-1
        Xn = 2 * X / (W - 1) - 1
        Yn = 2 * Y / (H - 1) - 1

        # Stack normalized coordinates to create the grid expected by grid_sample
        # Shape: (N, num_samples, n_width, 2) where the last dim is (x, y)
        grid = torch.stack([Xn, Yn], dim=-1)

        # --- Sampling ---
        # grid_sample expects input image shape (N, C, Hin, Win) and grid shape (N, Hout, Wout, 2)
        # Our image is (C, H, W) and grid is (N, Ns, Nw, 2).
        # We need to repeat the image N times to match the first dimension of the grid.
        # img_rep shape: (N, C, H, W)
        img_rep = img.unsqueeze(0).expand(N, C, H, W)

        # Perform the sampling
        # Output shape: (N, C, Ns, Nw) which is (N, C, num_samples, n_width)
        out = F.grid_sample(
            img_rep,
            grid,
            mode='bilinear',       # Common interpolation mode
            padding_mode='zeros',  # How to handle samples outside the image boundaries
            align_corners=align_corners
        )

        return out


class EdgeSampler(pl.LightningModule):
    """
    Samples pixel data from quadrilateral regions defined by pairs of lines.

    For each ordered pair of distinct lines (line_i, line_j), it defines a
    quadrilateral region connecting their endpoints (start_i, end_i, end_j, start_j).
    It then samples pixel values from this region using bilinear interpolation
    via F.grid_sample. This is intended for generating edge attributes in a
    graph representation where lines are nodes.

    Assumes usage within a PyTorch Lightning setup where device placement is handled.
    """
    def __init__(
        self,
        num_samples_u: int = 50, # Samples along the effective line direction
        num_samples_v: int = 5,  # Samples between the pair of lines
    ):
        """
        Initializes the EdgeSampler.

        Args:
            num_samples_u: Number of points to sample along the 'u' direction,
                           interpolating between the start-points-line and end-points-line.
            num_samples_v: Number of points to sample along the 'v' direction,
                           interpolating between line_i and line_j.
        """
        super().__init__()
        if num_samples_u <= 0 or num_samples_v <= 0:
            raise ValueError("Number of samples (u and v) must be positive.")

        self.num_samples_u = num_samples_u
        self.num_samples_v = num_samples_v

        # Precompute and cache the local sampling grid (u, v)
        # Ensure float type for interpolation calculations
        u = torch.linspace(0, 1, steps=num_samples_u, dtype=torch.float32)
        v = torch.linspace(0, 1, steps=num_samples_v, dtype=torch.float32)
        uu, vv = torch.meshgrid(u, v, indexing='ij') # uu:(Nu, Nv), vv:(Nu, Nv)

        # Register as buffers so they move with the module’s device
        # and are saved with the state_dict. PTL handles the device placement.
        self.register_buffer('uu', uu)
        self.register_buffer('vv', vv)

    def forward(
        self,
        img: torch.Tensor,
        lines: torch.Tensor,
        align_corners: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples quadrilateral regions between pairs of lines from an image.

        Args:
            img: Input image tensor (C, H, W). Expected to be on the correct device.
            lines: Lines defined by start/end points.
                   Tensor of shape (N, 2, 2) where N is the number of lines.
                   Coordinates should be in pixel space (x corresponding to W, y to H).
                   Expected to be on the correct device.
            align_corners: Argument passed to F.grid_sample. If True, the
                           extreme values -1 and 1 are considered pixel centers.
                           If False, they are considered pixel corners.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
            - sampled_patches: Tensor containing the sampled quadrilaterals,
              shape (M, C, num_samples_u, num_samples_v), where M = N * (N - 1)
              is the number of ordered pairs of distinct lines. On the same device as input.
            - edge_indices: Tensor indicating the pairs of line indices (i, j)
              corresponding to each sampled patch, shape (2, M) (PyG format).
              On the same device as input.
        """
        # --- Input Validation and Preparation ---
        if img.dim() != 3:
            raise ValueError(f"Input image must be a 3D tensor (C, H, W), but got shape {img.shape}")
        if not isinstance(lines, torch.Tensor):
             raise TypeError(f"Lines must be a torch.Tensor, but got {type(lines)}")
        if lines.dim() != 3 or lines.shape[1:] != (2, 2):
             raise ValueError(f"Lines tensor must have shape (N, 2, 2), but got {lines.shape}")

        # Ensure lines tensor is float for coordinate calculations
        # (It usually is, but good to be safe or cast if necessary upstream)
        if not torch.is_floating_point(lines):
             # If lines are integer coordinates, cast them for interpolation
             lines = lines.float()
             # Consider raising a warning or error if precision loss is critical

        C, H, W = img.shape
        N = lines.shape[0] # Number of lines
        img_dtype = img.dtype # Use image dtype for grid sampling compatibility
        computation_dtype = lines.dtype # Use lines dtype for coordinate math

        if N <= 1:
             # Cannot form pairs if N < 2
             M = 0
             out_shape = (M, C, self.num_samples_u, self.num_samples_v)
             # Create edge_indices on the same device as lines implicitly via device=lines.device
             # Although lines might be empty tensor if N=0, check N=1 case first
             edge_indices = torch.empty((M, 2), dtype=torch.long, device=lines.device)
             # Create empty tensor on the same device as img
             sampled_patches = torch.empty(out_shape, dtype=img_dtype, device=img.device)
             return sampled_patches, edge_indices.t() # Return (2, M) format

        # --- Generate Pairs of Line Indices ---
        # Create index tensor on the same device as the input 'lines' tensor
        idx = torch.arange(N, device=lines.device)
        # Use cartesian_prod and filter self-loops
        all_pairs = torch.cartesian_prod(idx, idx) # Shape (N*N, 2)
        mask = all_pairs[:, 0] != all_pairs[:, 1]
        edge_indices = all_pairs[mask] # Shape (M, 2), M = N * (N-1). Inherits device from idx.
        idx_i = edge_indices[:, 0] # Shape (M,)
        idx_j = edge_indices[:, 1] # Shape (M,)
        M = edge_indices.shape[0] # Number of pairs

        # --- Extract Corner Points for Each Pair ---
        # Indexing preserves the device
        lines_i = lines[idx_i]  # (M, 2, 2)
        lines_j = lines[idx_j]  # (M, 2, 2)

        # P00 = start_i, P10 = end_i, P01 = start_j, P11 = end_j
        # Coordinates are (x, y)
        P00 = lines_i[:, 0, :] # (M, 2) [x, y]
        P10 = lines_i[:, 1, :] # (M, 2)
        P01 = lines_j[:, 0, :] # (M, 2)
        P11 = lines_j[:, 1, :] # (M, 2)

        # --- Bilinear Interpolation for Sampling Grid ---
        # Buffers self.uu and self.vv are automatically on the correct device.
        # Cast them to the computation dtype (float type of lines) for coordinate math.
        uu = self.uu.to(dtype=computation_dtype) # (Nu, Nv)
        vv = self.vv.to(dtype=computation_dtype) # (Nu, Nv)

        # Reshape points and grid for broadcasting:
        # Points: (M, 1, 1, 2)
        # Grid params: (1, Nu, Nv, 1)
        # View operations preserve the device
        P00 = P00.view(M, 1, 1, 2)
        P10 = P10.view(M, 1, 1, 2)
        P01 = P01.view(M, 1, 1, 2)
        P11 = P11.view(M, 1, 1, 2)
        uu = uu.view(1, self.num_samples_u, self.num_samples_v, 1)
        vv = vv.view(1, self.num_samples_u, self.num_samples_v, 1)

        # Bilinear interpolation formula:
        # P(u,v) = (1-v) * ((1-u)P00 + u*P10) + v * ((1-u)P01 + u*P11)
        # The calculation happens on the device of the operands (which is the correct device).
        P_i = (1 - uu) * P00 + uu * P10 # (M, Nu, Nv, 2)
        P_j = (1 - uu) * P01 + uu * P11 # (M, Nu, Nv, 2)
        XY = (1 - vv) * P_i + vv * P_j  # (M, Nu, Nv, 2)

        # Extract X and Y coordinates
        X = XY[..., 0] # (M, Nu, Nv)
        Y = XY[..., 1] # (M, Nu, Nv)

        # --- Normalize Coordinates ---
        # Normalize to the range [-1, +1] for grid_sample
        # Handle cases where W or H might be 1
        # Use float division
        W_eff = torch.tensor(max(1, W - 1), dtype=computation_dtype, device=lines.device)
        H_eff = torch.tensor(max(1, H - 1), dtype=computation_dtype, device=lines.device)
        Xn = 2 * X / W_eff - 1
        Yn = 2 * Y / H_eff - 1

        # Stack normalized coordinates to create the grid expected by grid_sample
        # Shape: (M, num_samples_u, num_samples_v, 2) where the last dim is (x, y)
        # Ensure the grid dtype matches the image dtype for grid_sample,
        # although grid_sample often handles float grids with various image types.
        # Casting here ensures compatibility if needed.
        grid = torch.stack([Xn, Yn], dim=-1).to(dtype=img_dtype)

        # --- Sampling ---
        # grid_sample expects input image shape (N, C, Hin, Win) and grid shape (N, Hout, Wout, 2)
        # Our image is (C, H, W) and grid is (M, Nu, Nv, 2).
        # We need to repeat the image M times to match the first dimension of the grid.
        # Expand preserves device.
        img_rep = img.unsqueeze(0).expand(M, C, H, W)

        # Perform the sampling
        # Output shape: (M, C, Nu, Nv) which is (M, C, num_samples_u, num_samples_v)
        # Output tensor will be on the same device as img_rep and grid.
        sampled_patches = F.grid_sample(
            img_rep,
            grid,
            mode='bilinear',       # Common interpolation mode
            padding_mode='zeros',  # How to handle samples outside the image boundaries
            align_corners=align_corners
        )

        # edge_indices is already on the correct device (derived from lines.device)
        return sampled_patches, edge_indices.t() # Return edge_indices transposed (2, M) for PyG standard




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