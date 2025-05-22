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
        super().__init__()
        self.num_samples = num_samples
        self.width = width
        # Precompute and cache the local sampling grid (t, s)
        n_width = max(1, round(width))
        t = torch.linspace(0, 1, steps=num_samples)
        s = torch.linspace(-width / 2, width / 2, steps=n_width)
        tt, ss = torch.meshgrid(t, s, indexing='ij')
        # Register as buffers so they move with the module’s device/dtype
        self.register_buffer('tt', tt)
        self.register_buffer('ss', ss)

    def sample_lines_grid(
        self,
        img: torch.Tensor,
        lines: Union[torch.Tensor, List[Tuple[Tuple[float, float], Tuple[float, float]]]],
        align_corners: bool = True
    ) -> torch.Tensor:
        """
        Samples strips of pixel width around multiple lines using the cached (t,s) grid.

        Args:
            img: (C,H,W) or (B,C,H,W) tensor
            lines: (N,2,2) tensor or list of N ((x0,y0),(x1,y1))
            align_corners: passed to grid_sample

        Returns:
            (N, C, num_samples, width) or (B, N, C, num_samples, width)
        """
        # Determine number of width samples
        n_width = self.ss.shape[1]

        # Ensure batch dimension
        batched = True
        if img.dim() == 3:
            img = img.unsqueeze(0)
            batched = False
        B, C, H, W = img.shape

        # Convert lines to tensor on correct device/dtype
        if isinstance(lines, list):
            lines = torch.tensor(lines, dtype=img.dtype)
        else:
            lines = lines.to(dtype=img.dtype)
        N = lines.shape[0]

        # Compute direction and perpendicular vectors
        starts = lines[:, 0, :]  # (N,2)
        ends   = lines[:, 1, :]  # (N,2)
        x0, y0 = starts.unbind(-1)
        x1, y1 = ends.unbind(-1)
        dx, dy = x1 - x0, y1 - y0
        length = (dx*dx + dy*dy).sqrt().clamp(min=1e-6)
        ux, uy = dx / length, dy / length
        px, py = -uy, ux

        # Use cached (t,s) grid
        tt = self.tt
        ss = self.ss.to(dtype=img.dtype)
        # Map (t,s) → absolute coords (N, num_samples, n_width)
        X = x0.view(N,1,1) + dx.view(N,1,1) * tt + px.view(N,1,1) * ss
        Y = y0.view(N,1,1) + dy.view(N,1,1) * tt + py.view(N,1,1) * ss

        # Normalize to [-1, +1]
        Xn = 2 * X / (W - 1) - 1
        Yn = 2 * Y / (H - 1) - 1
        grid = torch.stack([Xn, Yn], dim=-1)  # (N, num_samples, n_width, 2)

        # Expand without copy: (B, N, num_samples, n_width, 2) → (B*N, num_samples, n_width, 2)
        grid = grid.unsqueeze(0).expand(B, N, self.num_samples, n_width, 2)
        grid = grid.reshape(B * N, self.num_samples, n_width, 2)

        # Expand image: (B,1,C,H,W) → (B,N,C,H,W) → (B*N,C,H,W)
        img_rep = img.unsqueeze(1).expand(B, N, C, H, W)
        img_rep = img_rep.reshape(B * N, C, H, W)

        # Sample
        out = F.grid_sample(
            img_rep,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=align_corners
        )

        # Reshape back: (B, N, C, num_samples, n_width)
        out = out.view(B, N, C, self.num_samples, n_width)
        if not batched:
            out = out.squeeze(0)  # (N, C, num_samples, n_width)

        return out



import torch
import torch.nn.functional as F
import pytorch_lightning as pl

class EdgeSampler(pl.LightningModule):
    def __init__(
        self,
        num_samples_u: int = 50,  # samples along the line direction
        num_samples_v: int = 5,   # samples across between the two lines
    ):
        super().__init__()
        if num_samples_u <= 0 or num_samples_v <= 0:
            raise ValueError("Number of samples (u and v) must be positive.")

        self.num_samples_u = num_samples_u
        self.num_samples_v = num_samples_v

        # Precompute and register the (u,v) sampling grid
        u = torch.linspace(0, 1, steps=num_samples_u, dtype=torch.float32)
        v = torch.linspace(0, 1, steps=num_samples_v, dtype=torch.float32)
        uu, vv = torch.meshgrid(u, v, indexing='ij')  # shapes (Nu, Nv)
        self.register_buffer('uu', uu)
        self.register_buffer('vv', vv)
    def forward(self,
                img:    torch.Tensor,   # (C, H, W)
                quads:  torch.Tensor,   # (E, 4, 2)
                align_corners: bool = True
              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Now only samples the provided E quads, returns:
        - patches:     (E, C, num_samples_u, num_samples_v)
        - edge_index:  (2, E)  in the same order as `quads`
        """
        C, H, W = img.shape
        E       = quads.size(0)

        # unpack corner points
        P00 = quads[:,0]  # (E,2)
        P10 = quads[:,1]
        P11 = quads[:,2]
        P01 = quads[:,3]

        # build (u,v)-grid as before, broadcast to (E, Nu, Nv)
        uu = self.uu.view(1, self.num_samples_u, self.num_samples_v, 1)
        vv = self.vv.view(1, self.num_samples_u, self.num_samples_v, 1)

        # bilinear interp: same formula you had
        P00 = P00.view(E,1,1,2);  P10 = P10.view(E,1,1,2)
        P01 = P01.view(E,1,1,2);  P11 = P11.view(E,1,1,2)
        P_i = (1 - uu)*P00 + uu*P10
        P_j = (1 - uu)*P01 + uu*P11
        XY  = (1 - vv)*P_i  + vv*P_j   # (E, Nu, Nv, 2)
        X, Y = XY[...,0], XY[...,1]

        # normalize to [-1,1]
        Xn = 2*X/(W-1) - 1
        Yn = 2*Y/(H-1) - 1
        grid = torch.stack([Xn, Yn], dim=-1).to(img.dtype)

        # repeat image E times
        img_rep = img.unsqueeze(0).expand(E, C, H, W)

        patches = F.grid_sample(
            img_rep, grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=align_corners
        )

    
        return patches



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




import cv2
import numpy as np
import torch

def extract_fixed_oriented_patches(
    img: np.ndarray,
    lines: np.ndarray,
    patch_size: tuple = (64, 128),        # (height, width) of every patch
    draw_line: bool = True,               # re‑draw the line inside the patch?
    border_value: int | tuple = 0         # padding colour if patch sticks out
) -> torch.Tensor:
    """
    Parameters
    ----------
    img        : H×W×C  (uint8/float32)  –  RGB/BGR or grayscale.
    lines      : (N, 2, 2)  –  [[x1,y1],[x2,y2]] in pixel coords.
    patch_size : (patch_h, patch_w)  –  SAME for every line.
    draw_line  : If True, cv2.line is drawn on the patch (helps the CNN).
    border_value : Background colour for areas outside the image.

    Returns
    -------
    torch.Tensor  (N, C, patch_h, patch_w)
    """
    if lines is None or len(lines) == 0:
        raise ValueError("`lines` is empty!")

    # ensure 3‑D image (H,W,C)
    if img.ndim == 2:
        img = img[..., None]
    H, W, C = img.shape
    patch_h, patch_w = patch_size
    half_h, half_w   = patch_h // 2, patch_w // 2

    # container for all patches
    patches = []

    for p1, p2 in lines:
        x1, y1 = p1
        x2, y2 = p2

        # -- 1. centre & angle ------------------------------------------------
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        angle  = np.degrees(np.arctan2(y2 - y1, x2 - x1))   # line's angle

        # -- 2. rotate ENTIRE image around the centre (nearest‑neighbour) ----
        M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)  # negative = align
        rotated = cv2.warpAffine(
            img,
            M,
            dsize=(W, H),
            flags=cv2.INTER_NEAREST,       # *no* interpolation smoothing
            borderValue=border_value
        )

        # -- 3. find new centre position after rotation ----------------------
        cx_r, cy_r, _ = np.dot(M, np.array([cx, cy, 1.0]))

        # -- 4. crop a regular axis‑aligned rectangle around the centre -------
        patch = cv2.getRectSubPix(
            rotated,
            patchSize=(patch_w, patch_h),   # note: (width, height) order
            center=(cx_r, cy_r)
        )

        # -- 5. (optional) draw the line again in local coords ---------------
        if draw_line:
            # endpoints in the rotated frame
            p1_r = tuple(np.dot(M, np.array([x1, y1, 1.0]))[:2] -
                         np.array([cx_r - half_w, cy_r - half_h]))
            p2_r = tuple(np.dot(M, np.array([x2, y2, 1.0]))[:2] -
                         np.array([cx_r - half_w, cy_r - half_h]))
            cv2.line(patch, p1_r, p2_r, color=(255, 255, 255), thickness=1)

        patches.append(patch)

    # -- 6. stack and convert to PyTorch (N, C, H, W) -------------------------
    patches_np = np.stack(patches)                       # (N, H, W, C)
    patches_t  = torch.from_numpy(patches_np).permute(0, 3, 1, 2)
    return patches_t


import cv2
import numpy as np
import torch

def extract_oriented_feature_patches_feature_map(
    feature_map: torch.Tensor,
    lines: np.ndarray,
    patch_size_img: tuple,              # same (H,W) you used on the RGB image
    downsample_ratio: int,              # e.g. 4 for a stride‑4 backbone
    draw_line: bool = False,
    border_value: float | tuple = 0.0,
) -> torch.Tensor:
    """
    Returns
    -------
    patches_t : (N, C, H_f, W_f)  torch tensor
                H_f = round(patch_size_img[0]/downsample_ratio)
                W_f = round(patch_size_img[1]/downsample_ratio)
    """
    # ------------------------------------------------------------------ #
    # 0.  book‑keeping                                                   #
    # ------------------------------------------------------------------ #
    if feature_map.dim() == 4:          # (1,C,Hf,Wf) → (C,Hf,Wf)
        feature_map = feature_map.squeeze(0)
    C, Hf, Wf = feature_map.shape
    fmap_np = feature_map.permute(1, 2, 0).cpu().numpy()      # (Hf,Wf,C)

    # size we want to crop on the *feature* grid
    H_patch = int(round(patch_size_img[0] / downsample_ratio))
    W_patch = int(round(patch_size_img[1] / downsample_ratio))
    half_h, half_w = H_patch // 2, W_patch // 2

    # scale line coordinates to feature‑map resolution
    lines_f = np.asarray(lines, dtype=np.float32) / downsample_ratio
    patches = []

    # ------------------------------------------------------------------ #
    # 1.  iterate over lines                                             #
    # ------------------------------------------------------------------ #
    for (x1f, y1f), (x2f, y2f) in lines_f:
        # centre & angle
        cx, cy = (x1f + x2f) / 2.0, (y1f + y2f) / 2.0
        angle  = np.degrees(np.arctan2(y2f - y1f, x2f - x1f))

        # rotate entire feature map (nearest = no interpolation blur)
        M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        rot = cv2.warpAffine(
            fmap_np,
            M,
            dsize=(Wf, Hf),
            flags=cv2.INTER_NEAREST,
            borderValue=border_value,
        )

        # new feature‑map coords of the patch centre
        cx_r, cy_r, _ = M @ np.array([cx, cy, 1.0])

        # crop axis‑aligned rectangle
        patch = cv2.getRectSubPix(
            rot,
            patchSize=(W_patch, H_patch),    # (width, height)
            center=(cx_r, cy_r),
        )

        if draw_line:
            # endpoints in rotated frame, then local coords in the patch
            p1_r = (M @ np.array([x1f, y1f, 1.]))[:2] - [cx_r - half_w, cy_r - half_h]
            p2_r = (M @ np.array([x2f, y2f, 1.]))[:2] - [cx_r - half_w, cy_r - half_h]
            cv2.line(patch,
                     tuple(np.round(p1_r).astype(int)),
                     tuple(np.round(p2_r).astype(int)),
                     color=(1.0,) * C, thickness=1)

        patches.append(patch)

    # ------------------------------------------------------------------ #
    # 2.  stack & back to torch                                          #
    # ------------------------------------------------------------------ #
    patches_np = np.stack(patches)                          # (N, Hf_p, Wf_p, C)
    patches_t  = torch.from_numpy(patches_np).permute(0, 3, 1, 2)  # (N,C,Hf_p,Wf_p)
    return patches_t

