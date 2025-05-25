import os
import glob
import logging
import json
from typing import Optional, Tuple, Dict, Any

import cv2
import numpy as np
import h5py

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Utility Functions ---

def compute_normal_map_from_points(points: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Compute a normal map from a 3D point cloud using spatial gradients.

    Args:
        points (np.ndarray): 3D points of shape (H, W, 3), expected dtype float32.
        ksize (int): Kernel size for the Sobel operator.

    Returns:
        np.ndarray: Normal map of shape (H, W, 3) with unit normals (float32).
                    Returns a zero array if input is invalid.
    """
    if points.ndim != 3 or points.shape[2] != 3:
        logging.error("Input points must have shape (H, W, 3).")
        # Return a zero array of the expected shape if possible, else raise error
        # For simplicity returning zeros matching potential H, W if ndim is 3
        h, w = points.shape[:2] if points.ndim >= 2 else (0, 0)
        return np.zeros((h, w, 3), dtype=np.float32)

    points_cont = np.ascontiguousarray(points, dtype=np.float32)
    h, w, _ = points.shape

    grad_x = np.zeros_like(points_cont)
    grad_y = np.zeros_like(points_cont)

    for channel in range(3):
        # Ensure channel data is contiguous for cv2.Sobel
        channel_data = np.ascontiguousarray(points_cont[..., channel])
        grad_x[..., channel] = cv2.Sobel(channel_data, cv2.CV_32F, 1, 0, ksize=ksize, borderType=cv2.BORDER_DEFAULT)
        grad_y[..., channel] = cv2.Sobel(channel_data, cv2.CV_32F, 0, 1, ksize=ksize, borderType=cv2.BORDER_DEFAULT)

    # Compute cross product to obtain normals.
    # Note: The order might matter depending on coordinate system conventions (right-hand vs left-hand)
    # Adjust np.cross(grad_x, grad_y) or np.cross(grad_y, grad_x) if normals point inwards.
    normals = np.cross(grad_y, grad_x) # Typical order for image gradients to world normals

    # Normalize the normals.
    norm = np.linalg.norm(normals, axis=2, keepdims=True)

    # Handle zero-norm vectors safely
    zero_norm_mask = (norm == 0)
    norm[zero_norm_mask] = 1.0  # Avoid division by zero

    normalized_normals = normals / norm
    normalized_normals[zero_norm_mask.squeeze()] = 0 # Set zero-norm vectors explicitly to zero

    return normalized_normals.astype(np.float32)


def calculate_normal_map_from_depth(depth_map: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Compute a normal map from a depth map using spatial gradients (camera space).

    Assumes depth_map represents Z distance in camera coordinates.
    Normals calculated are in camera coordinate space.

    Args:
        depth_map (np.ndarray): Depth map of shape (H, W), expected dtype float32.
        ksize (int): Kernel size for the Sobel operator.

    Returns:
        np.ndarray: Normal map of shape (H, W, 3) with unit normals (float64).
    """
    depth_map_f32 = depth_map.astype(np.float32)

    # Compute gradients
    grad_x = cv2.Sobel(depth_map_f32, cv2.CV_32F, 1, 0, ksize=ksize, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(depth_map_f32, cv2.CV_32F, 0, 1, ksize=ksize, borderType=cv2.BORDER_DEFAULT)

    # Compute normal vectors in camera space (assuming standard perspective projection)
    # Normal = [-dz/dx, -dz/dy, 1] normalized
    normal_x = -grad_x
    normal_y = -grad_y
    normal_z = np.ones_like(depth_map_f32)

    # Normalize the normals
    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    
    # Handle zero-norm vectors safely
    zero_norm_mask = (norm == 0)
    norm[zero_norm_mask] = 1.0 # Avoid division by zero

    normal_x /= norm
    normal_y /= norm
    normal_z /= norm

    normal_map = np.stack([normal_x, normal_y, normal_z], axis=-1)
    
    # Set zero-norm vectors explicitly to zero after stacking
    normal_map[zero_norm_mask] = 0 

    # Note: Original code returned float64, retaining that here. Consider float32 for consistency.
    return normal_map.astype(np.float64)


def raydepth2depth(raydepth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convert ray depth (distance along viewing ray) to perspective depth (Z distance).

    Args:
        raydepth (np.ndarray): Ray depth map of shape (H, W).
        K (np.ndarray): Camera intrinsic matrix (3x3).

    Returns:
        np.ndarray: Perspective depth map (Z distance) of shape (H, W).
    """
    if K.shape != (3, 3):
        raise ValueError("Intrinsic matrix K must be 3x3.")
    if raydepth.ndim != 2:
        raise ValueError("Ray depth map must be 2D (H, W).")

    try:
        K_inv = np.linalg.inv(K)
    except np.linalg.LinAlgError:
        logging.error("Intrinsic matrix K is singular and cannot be inverted.")
        return np.zeros_like(raydepth) # Or raise an error

    h, w = raydepth.shape
    
    # Create pixel coordinates
    x_coords, y_coords = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    
    # Create homogeneous coordinates (x, y, 1) for each pixel
    # Shape: (3, H*W)
    coords_homo = np.stack([x_coords.ravel(), y_coords.ravel(), np.ones(h * w, dtype=np.float32)])

    # Transform homogeneous coordinates to camera coordinates (X/Z, Y/Z, 1) using K_inv
    # Then calculate the norm of the direction vector (X/Z, Y/Z, 1)
    # Shape: (H*W,)
    coeffs = np.linalg.norm(K_inv @ coords_homo, axis=0)

    # Reshape coefficients back to (H, W)
    coeffs_map = coeffs.reshape(h, w)

    # Avoid division by zero where coeffs might be zero (e.g., principal point mapping to origin)
    # Although highly unlikely for standard intrinsics
    coeffs_map[coeffs_map == 0] = 1e-6 # Replace with a small number

    depth = raydepth / coeffs_map
    return depth


# --- Generic File Handling ---

def _find_file(search_pattern: str) -> Optional[str]:
    """Finds the first file matching the glob pattern."""
    logging.debug(f"Searching for pattern: {search_pattern}")
    files = glob.glob(search_pattern, recursive=True)
    if not files:
        logging.warning(f"No file found matching pattern: {search_pattern}")
        return None
    if len(files) > 1:
        logging.warning(f"Multiple files found for pattern: {search_pattern}. Using first one: {files[0]}")
    return files[0]

def _load_image(filepath: str, color_conversion: Optional[int] = None) -> Optional[np.ndarray]:
    """Loads an image using OpenCV."""
    if not os.path.exists(filepath):
        logging.error(f"Image file not found: {filepath}")
        return None
    try:
        img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED) # Load as is (handles color, grayscale, alpha)
        if img is None:
            logging.error(f"Failed to load image (cv2.imread returned None): {filepath}")
            return None
        if color_conversion is not None:
            img = cv2.cvtColor(img, color_conversion)
        return img
    except Exception as e:
        logging.error(f"Error loading image {filepath}: {e}")
        return None

def _load_hdf5(filepath: str, key: str = 'dataset') -> Optional[np.ndarray]:
    """Loads a dataset from an HDF5 file."""
    if not os.path.exists(filepath):
        logging.error(f"HDF5 file not found: {filepath}")
        return None
    try:
        with h5py.File(filepath, 'r') as f:
            if key not in f:
                logging.error(f"Key '{key}' not found in HDF5 file: {filepath}. Available keys: {list(f.keys())}")
                return None
            data = np.array(f[key])
        return data
    except Exception as e:
        logging.error(f"Error loading HDF5 file {filepath}: {e}")
        return None

def _load_json(filepath: str) -> Optional[Dict[str, Any]]:
    """Loads data from a JSON file."""
    if not os.path.exists(filepath):
        logging.error(f"JSON file not found: {filepath}")
        return None
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON file {filepath}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error loading JSON file {filepath}: {e}")
        return None


# --- Dataset Loader Base Class (Optional but good practice) ---
class BaseDataLoader:
    """Abstract base class for dataset loaders."""
    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        if not os.path.isdir(base_dir):
             logging.warning(f"Base directory does not exist: {base_dir}") # Warn instead of error?

    def _get_path(self, *args) -> str:
        """Helper to construct paths relative to the base directory."""
        return os.path.join(self.base_dir, *args)


# --- Specific Dataset Loaders ---

class HypersimLoader(BaseDataLoader):
    """Loads data for the Hypersim dataset structure."""

    def _find_hypersim_file(self, image_id: str, cam_view: str, pattern: str) -> Optional[str]:
        """Finds a file within the Hypersim directory structure."""
        search_pattern = self._get_path(image_id, "images", cam_view, pattern)
        return _find_file(search_pattern)

    def load_color_image(self, image_id: str, frame_str: str, cam_view: str) -> Optional[np.ndarray]:
        """Loads a color image (JPG) and converts to RGB."""
        filename = f"frame.{frame_str}.color.jpg"
        filepath = self._find_hypersim_file(image_id, cam_view, filename)
        if filepath:
            # Hypersim images are typically saved as JPG (BGR), convert to RGB
            return _load_image(filepath, color_conversion=cv2.COLOR_BGR2RGB)
        logging.warning(f"Color image not found for {image_id}/{cam_view}/{frame_str}")
        return None

    def load_depth_map(self, image_id: str, frame_str: str, cam_view: str) -> Optional[np.ndarray]:
        """Loads a depth map (HDF5)."""
        filename = f"frame.{frame_str}.depth_meters.hdf5"
        filepath = self._find_hypersim_file(image_id, cam_view, filename)
        if filepath:
            depth = _load_hdf5(filepath)
            return depth.astype(np.float32) if depth is not None else None
        logging.warning(f"Depth map not found for {image_id}/{cam_view}/{frame_str}")
        return None

    def load_normal_map(self, image_id: str, frame_str: str, cam_view: str) -> Optional[np.ndarray]:
        """Loads a world-space normal map (HDF5)."""
        filename = f"frame.{frame_str}.normal_world.hdf5"
        filepath = self._find_hypersim_file(image_id, cam_view, filename)
        if filepath:
            normal = _load_hdf5(filepath)
            return normal.astype(np.float32) if normal is not None else None
        logging.warning(f"Normal map not found for {image_id}/{cam_view}/{frame_str}")
        return None

    def load_world_coordinates(self, image_id: str, frame_str: str, cam_view: str) -> Optional[np.ndarray]:
        """Loads world coordinates (position map) (HDF5)."""
        filename = f"frame.{frame_str}.position.hdf5"
        filepath = self._find_hypersim_file(image_id, cam_view, filename)
        if filepath:
            wc = _load_hdf5(filepath)
            return wc.astype(np.float32) if wc is not None else None
        logging.warning(f"World coordinates not found for {image_id}/{cam_view}/{frame_str}")
        return None


class MogeGtLoader(BaseDataLoader):
    """Loads data for the MOGE Ground Truth structure."""

    def _find_moge_gt_file(self, pattern: str) -> Optional[str]:
        """Finds a file within the MOGE GT directory structure."""
        search_pattern = self._get_path(pattern)
        return _find_file(search_pattern)

    def load_color_image(self) -> Optional[np.ndarray]:
        """Loads the GT color image (JPG) and converts to RGB."""
        filename = "image.jpg"
        filepath = self._find_moge_gt_file(filename)
        if filepath:
            # Assume standard BGR, convert to RGB
            return _load_image(filepath, color_conversion=cv2.COLOR_BGR2RGB)
        logging.warning(f"MOGE GT Color image not found in {self.base_dir}")
        return None

    def load_depth_map_png(self) -> Optional[np.ndarray]:
        """Loads the GT depth map (PNG)."""
        filename = "depth.png"
        filepath = self._find_moge_gt_file(filename)
        if filepath:
            # Load as grayscale
            depth = _load_image(filepath, color_conversion=None) # Load as is (should be grayscale)
            if depth is not None and depth.ndim == 2:
                 # Original code returned float64, retain this.
                return depth.astype(np.float64)
            elif depth is not None:
                logging.warning(f"Loaded depth map from {filepath} is not grayscale (shape: {depth.shape}). Returning None.")
                return None
        logging.warning(f"MOGE GT Depth PNG not found in {self.base_dir}")
        return None

    def load_intrinsics_matrix(self) -> Optional[np.ndarray]:
        """Loads the camera intrinsics matrix from meta.json."""
        filename = "meta.json"
        filepath = self._find_moge_gt_file(filename)
        if filepath:
            data = _load_json(filepath)
            if data and 'intrinsics' in data:
                try:
                    intrinsics_matrix = np.array(data['intrinsics'])
                    if intrinsics_matrix.shape == (3, 3):
                         return intrinsics_matrix.astype(np.float32)
                    else:
                         logging.error(f"Intrinsics matrix in {filepath} does not have shape (3, 3). Shape found: {intrinsics_matrix.shape}")
                         return None
                except Exception as e:
                    logging.error(f"Could not convert 'intrinsics' to numpy array from {filepath}: {e}")
                    return None
            else:
                logging.error(f"'intrinsics' key not found in {filepath} or file is empty.")
                return None
        logging.warning(f"MOGE GT meta.json not found in {self.base_dir}")
        return None


class MogePredLoader(BaseDataLoader):
    """Loads data for the MOGE Prediction structure."""

    def _find_moge_pred_file(self, pattern: str) -> Optional[str]:
        """Finds a file within the MOGE Prediction directory structure."""
        # Assumes predictions are within a 'moge' subdirectory relative to base_dir
        search_pattern = self._get_path("moge", pattern)
        return _find_file(search_pattern)

    def load_depth_map(self, file_id: str) -> Optional[np.ndarray]:
        """Loads a predicted depth map (HDF5)."""
        filename = f"frame.{file_id}.depth_meters.hdf5"
        filepath = self._find_moge_pred_file(filename)
        if filepath:
            depth = _load_hdf5(filepath)
            return depth.astype(np.float32) if depth is not None else None
        logging.warning(f"MOGE Pred Depth map not found for {file_id} in {self.base_dir}/moge")
        return None

    def load_normal_map(self, file_id: str) -> Optional[np.ndarray]:
        """Loads a predicted world-space normal map (HDF5)."""
        filename = f"frame.{file_id}.normal_world.hdf5"
        filepath = self._find_moge_pred_file(filename)
        if filepath:
            normal = _load_hdf5(filepath)
            return normal.astype(np.float32) if normal is not None else None
        logging.warning(f"MOGE Pred Normal map not found for {file_id} in {self.base_dir}/moge")
        return None

    def load_world_coordinates(self, file_id: str) -> Optional[np.ndarray]:
        """Loads predicted world coordinates (position map) (HDF5)."""
        filename = f"frame.{file_id}.position.hdf5"
        filepath = self._find_moge_pred_file(filename)
        if filepath:
            wc = _load_hdf5(filepath)
            return wc.astype(np.float32) if wc is not None else None
        logging.warning(f"MOGE Pred World coordinates not found for {file_id} in {self.base_dir}/moge")
        return None

    def load_intrinsics_K(self, file_id: str) -> Optional[np.ndarray]:
        """Loads the predicted intrinsics matrix K (HDF5)."""
        filename = f"frame.{file_id}.K.hdf5"
        filepath = self._find_moge_pred_file(filename)
        if filepath:
            k_matrix = _load_hdf5(filepath)
            if k_matrix is not None and k_matrix.shape == (3, 3):
                 return k_matrix.astype(np.float32)
            elif k_matrix is not None:
                 logging.error(f"Loaded K matrix from {filepath} does not have shape (3, 3). Shape found: {k_matrix.shape}")
                 return None
        logging.warning(f"MOGE Pred K matrix not found for {file_id} in {self.base_dir}/moge")
        return None

    def load_mask(self, file_id: str) -> Optional[np.ndarray]:
        """Loads a prediction mask (HDF5)."""
        filename = f"frame.{file_id}.mask.hdf5"
        filepath = self._find_moge_pred_file(filename)
        if filepath:
            mask = _load_hdf5(filepath)
            # Masks are often boolean or uint8, float32 might be okay depending on use
            return mask.astype(np.float32) if mask is not None else None
        logging.warning(f"MOGE Pred Mask not found for {file_id} in {self.base_dir}/moge")
        return None


# --- Example Usage ---

if __name__ == "__main__":
    # Create dummy directories and files for testing
    # NOTE: You'll need to create these directories/files or point to real data
    # For HDF5 files, you'll need h5py to create dummy data.
    # For images, you can create simple dummy images with numpy/cv2.
    
    # Example: Create dummy structure (replace with actual paths)
    dummy_base = "dummy_data"
    os.makedirs(os.path.join(dummy_base, "hypersim_scene", "images", "cam_01"), exist_ok=True)
    os.makedirs(os.path.join(dummy_base, "moge_gt_sample"), exist_ok=True)
    os.makedirs(os.path.join(dummy_base, "moge_pred_sample", "moge"), exist_ok=True)

    # Create dummy files (basic examples)
    cv2.imwrite(os.path.join(dummy_base, "hypersim_scene", "images", "cam_01", "frame.0001.color.jpg"), np.zeros((10, 10, 3), dtype=np.uint8))
    with h5py.File(os.path.join(dummy_base, "hypersim_scene", "images", "cam_01", "frame.0001.depth_meters.hdf5"), 'w') as f:
        f.create_dataset('dataset', data=np.ones((10, 10), dtype=np.float32))
    
    cv2.imwrite(os.path.join(dummy_base, "moge_gt_sample", "image.jpg"), np.zeros((12, 12, 3), dtype=np.uint8))
    cv2.imwrite(os.path.join(dummy_base, "moge_gt_sample", "depth.png"), np.ones((12, 12), dtype=np.uint16)*1000) # Example depth
    with open(os.path.join(dummy_base, "moge_gt_sample", "meta.json"), 'w') as f:
        json.dump({"intrinsics": [[500, 0, 6], [0, 500, 6], [0, 0, 1]]}, f)

    with h5py.File(os.path.join(dummy_base, "moge_pred_sample", "moge", "frame.sample01.depth_meters.hdf5"), 'w') as f:
        f.create_dataset('dataset', data=np.ones((15, 15), dtype=np.float32)*2.5)
    with h5py.File(os.path.join(dummy_base, "moge_pred_sample", "moge", "frame.sample01.K.hdf5"), 'w') as f:
        f.create_dataset('dataset', data=np.eye(3, dtype=np.float32))

    print("\n--- Testing HypersimLoader ---")
    hypersim_loader = HypersimLoader(os.path.join(dummy_base)) # Base dir contains scenes
    color_h = hypersim_loader.load_color_image("hypersim_scene", "0001", "cam_01")
    depth_h = hypersim_loader.load_depth_map("hypersim_scene", "0001", "cam_01")
    # normals_h = hypersim_loader.load_normal_map("hypersim_scene", "0001", "cam_01") # Needs dummy file
    # wc_h = hypersim_loader.load_world_coordinates("hypersim_scene", "0001", "cam_01") # Needs dummy file

    if color_h is not None:
        print(f"Hypersim Color shape: {color_h.shape}, dtype: {color_h.dtype}")
    if depth_h is not None:
        print(f"Hypersim Depth shape: {depth_h.shape}, dtype: {depth_h.dtype}")
        # Example: Calculate normals from depth
        # normals_from_depth = calculate_normal_map_from_depth(depth_h)
        # print(f"Calculated Normals shape: {normals_from_depth.shape}, dtype: {normals_from_depth.dtype}")


    print("\n--- Testing MogeGtLoader ---")
    moge_gt_loader = MogeGtLoader(os.path.join(dummy_base, "moge_gt_sample"))
    color_gt = moge_gt_loader.load_color_image()
    depth_gt_png = moge_gt_loader.load_depth_map_png()
    intrinsics_gt = moge_gt_loader.load_intrinsics_matrix()

    if color_gt is not None:
        print(f"MOGE GT Color shape: {color_gt.shape}, dtype: {color_gt.dtype}")
    if depth_gt_png is not None:
        print(f"MOGE GT Depth PNG shape: {depth_gt_png.shape}, dtype: {depth_gt_png.dtype}")
    if intrinsics_gt is not None:
        print(f"MOGE GT Intrinsics:\n{intrinsics_gt}")


    print("\n--- Testing MogePredLoader ---")
    moge_pred_loader = MogePredLoader(os.path.join(dummy_base, "moge_pred_sample"))
    depth_pred = moge_pred_loader.load_depth_map("sample01")
    k_pred = moge_pred_loader.load_intrinsics_K("sample01")
    # normal_pred = moge_pred_loader.load_normal_map("sample01") # Needs dummy file
    # wc_pred = moge_pred_loader.load_world_coordinates("sample01") # Needs dummy file
    # mask_pred = moge_pred_loader.load_mask("sample01") # Needs dummy file

    if depth_pred is not None:
        print(f"MOGE Pred Depth shape: {depth_pred.shape}, dtype: {depth_pred.dtype}")
    if k_pred is not None:
        print(f"MOGE Pred K matrix:\n{k_pred}")


    print("\n--- Testing Utility Functions ---")
    # Example: Create dummy points and compute normals
    points_dummy = np.random.rand(10, 10, 3).astype(np.float32) * 10
    normals_from_points = compute_normal_map_from_points(points_dummy)
    print(f"Normals from points shape: {normals_from_points.shape}, dtype: {normals_from_points.dtype}")

    # Example: Ray depth conversion (needs depth and K)
    if depth_pred is not None and k_pred is not None:
        # Assuming depth_pred is actually raydepth for this example
        persp_depth = raydepth2depth(depth_pred, k_pred)
        print(f"Converted perspective depth shape: {persp_depth.shape}, dtype: {persp_depth.dtype}")

    # Clean up dummy files/dirs if needed (optional)
    # import shutil
    # shutil.rmtree(dummy_base)