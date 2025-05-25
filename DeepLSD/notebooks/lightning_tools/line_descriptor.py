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
from typing import List, Tuple, Dict, Any

def extract_resized_line_bands(
    img: np.ndarray,
    angle_field: np.ndarray,
    distance_field: np.ndarray,
    lines: List[Tuple[int, int, int, int]],
    width: int,
    target_length: int,
    *,
    downsampling_h: float ,
    downsampling_w: float,
    padding_mode: str = "constant",
) -> Dict[str, List[np.ndarray]]:
    """
    Extract fixed-size rectangular bands centred on 2-D line segments from:

        • an RGB image        (H × W × 3)
        • an angle-field map  (H/↓h × W/↓w × Ca)
        • a distance-field    (H/↓h × W/↓w [+ C])

    Parameters
    ----------
    img, angle_field, distance_field
        Co-registered inputs.  `angle_field` and `distance_field` are assumed to
        be down-sampled by (`downsampling_h`, `downsampling_w`) relative to `img`.
    lines : list[(x1, y1, x2, y2)]
        Line segments in **original image coordinates**.
    width : int
        Band thickness (pixels) perpendicular to each line *in the image grid*.
    target_length : int
        Longitudinal size of every returned band after resizing.
    downsampling_h, downsampling_w : int | float, default 1
        Vertical / horizontal down-sampling factors of the feature-maps.
    padding_mode : str, default "constant"
        How to pad when a band crosses the border
        {"constant", "edge"/"replicate", "reflect", "wrap"}.

    Returns
    -------
    dict
        {
          "rgb"            : [...],   # list[ np.ndarray(width, target_length, 3) ]
          "angle_field"    : [...],   # list[ …  (width, target_length, Ca) ]
          "distance_field" : [...]    # list[ …  (width, target_length[, C]) ]
        }
    """
    # ---------- border‐mode lookup -------------------------------------------------
    _borders = {
        "constant": cv2.BORDER_CONSTANT,
        "edge":     cv2.BORDER_REPLICATE,
        "replicate":cv2.BORDER_REPLICATE,
        "reflect":  cv2.BORDER_REFLECT_101,
        "reflect_101": cv2.BORDER_REFLECT_101,
        "wrap":     cv2.BORDER_WRAP,
    }
    border_flag = _borders.get(padding_mode.lower(), cv2.BORDER_CONSTANT)

    # ---------- helper: warp & resize a single rectangular strip -------------------
    def _band(im: np.ndarray, p1, p2, band_w_px: int) -> np.ndarray:
        (x1, y1), (x2, y2) = p1, p2
        dx, dy = x2 - x1, y2 - y1
        seg_len = float(np.hypot(dx, dy))
        if seg_len < 1e-3:                            # degenerate → zeros
            out_shape = (width, target_length) if im.ndim == 2 else \
                        (width, target_length, im.shape[2])
            return np.zeros(out_shape, dtype=im.dtype)

        ux, uy = dx / seg_len, dy / seg_len          # unit tangent
        nx, ny = -uy, ux                             # unit normal
        half_w = band_w_px / 2.0

        # source quadrilateral (float32)
        src = np.float32([
            [x1 - nx * half_w, y1 - ny * half_w],
            [x1 + nx * half_w, y1 + ny * half_w],
            [x2 + nx * half_w, y2 + ny * half_w],
            [x2 - nx * half_w, y2 - ny * half_w],
        ])
        dst_h = max(1, int(round(seg_len)))
        dst = np.float32([
            [0,            0],
            [band_w_px-1,  0],
            [band_w_px-1,  dst_h-1],
            [0,            dst_h-1],
        ])

        M  = cv2.getPerspectiveTransform(src, dst)
        raw = cv2.warpPerspective(
            im, M,
            (band_w_px, dst_h),                       # (out_w, out_h)
            flags=cv2.INTER_LINEAR,
            borderMode=border_flag,
        )
        return cv2.resize(raw, (target_length, width), interpolation=cv2.INTER_LINEAR)

    # ---------- sanity-check feature-map scale -------------------------------------
    H, W = img.shape[:2]
    for name, fmap in (("angle_field", angle_field), ("distance_field", distance_field)):
        hf, wf = fmap.shape[:2]
        if abs(H / hf - downsampling_h) > 1e-3 or abs(W / wf - downsampling_w) > 1e-3:
            raise ValueError(
                f"{name} has shape {(hf, wf)} but down-sampling factors "
                f"({downsampling_h}, {downsampling_w}) do not match the RGB "
                f"image {(H, W)}."
            )
    angle_field = np.ascontiguousarray(angle_field, dtype=np.float32)
    distance_field = np.ascontiguousarray(distance_field, dtype=np.float32)

  
    # ------ pre-compute constants for feature-map coordinate conversion ------------
    inv_ds_h, inv_ds_w = 1.0 / downsampling_h, 1.0 / downsampling_w
    # choose feature-band thickness so projected area ≈ image band area
    if downsampling_h == downsampling_w:
        ds_factor = float(downsampling_h)
    else:
        ds_factor = np.sqrt(downsampling_h * downsampling_w)
    #f_band_w = max(1, int(round(width / ds_factor)))
    f_band_w = int(round(width / ds_factor))

    # ---------- iterate over lines and collect bands -------------------------------
    rgb_bands, ang_bands, dist_bands = [], [], []
    for l in lines:
        # RGB band (same coords)
        x1, y1 = l[0][0], l[0][1]
        x2, y2 = l[1][0], l[1][1]
        rgb_band = _band(img, (x1, y1), (x2, y2), width)
        rgb_bands.append(rgb_band)

        # feature-map coordinates (float → sub-pixel accurate)
        xf1, yf1 = x1 * inv_ds_w, y1 * inv_ds_h
        xf2, yf2 = x2 * inv_ds_w, y2 * inv_ds_h


        ang_band = _band(angle_field, (xf1, yf1), (xf2, yf2), f_band_w)
        if ang_band.ndim == 2:
            ang_band = ang_band[..., np.newaxis]
        ang_bands.append(ang_band)

        dist_band = _band(distance_field, (xf1, yf1), (xf2, yf2), f_band_w)

        if dist_band.ndim == 2:
            dist_band = dist_band[..., np.newaxis]

        dist_bands.append(dist_band)

         
    return {
        "rgb": rgb_bands,
        "angle_field": ang_bands,
        "distance_field": dist_bands,
    }
