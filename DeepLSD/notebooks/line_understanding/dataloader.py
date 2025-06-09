import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import h5py
import numpy as np


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")



class FileLoader:
    """
    Generic file loading utilities.
    """

    @staticmethod
    def find_first(pattern: Union[str, Path]) -> Optional[Path]:
        path_obj = Path(pattern)
        paths = list(path_obj.parent.glob(path_obj.name)) if path_obj.parent != Path('.') else list(path_obj.glob())
        if not paths:
            logging.warning("No files match: %s", pattern)
            return None
        if len(paths) > 1:
            logging.warning("Multiple files match, using first: %s", paths[0])
        return paths[0]

    @staticmethod
    def load_image(path: Optional[Path], color_conv: Optional[int] = None, is_grey=False) -> Optional[np.ndarray]:
        if path is None or not path.exists():
            logging.error("Image not found: %s", path)
            return None
        if not is_grey:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            logging.error("Failed to load image: %s", path)
            return None
        return cv2.cvtColor(img, color_conv) if color_conv is not None else img

    @staticmethod
    def load_hdf5(path: Optional[Path], key: str = "dataset") -> Optional[np.ndarray]:
        if path is None or not path.exists():
            logging.error("HDF5 not found: %s", path)
            return None
        with h5py.File(path, "r") as f:
            if key not in f:
                logging.error("Key '%s' missing in %s", key, path)
                return None
            return np.array(f[key])

    @staticmethod
    def load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
        if path is None or not path.exists():
            logging.error("JSON not found: %s", path)
            return None
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError as err:
            logging.error("JSON decode error in %s: %s", path, err)
            return None


class BaseDataLoader:
    """
    Abstract base for dataset loaders; accepts str or Path.
    """

    def __init__(
        self, base_dir: Union[str, Path]
    ):
        self.base_dir: Path = Path(base_dir)
        if not self.base_dir.is_dir():
            logging.warning("Base directory missing or not a directory: %s", self.base_dir)

    def path(self, *segments: str) -> Path:
        return self.base_dir.joinpath(*segments)


class HypersimLoader(BaseDataLoader):
    """Loader for Hypersim dataset."""

    def _find(
        self, image_id: str, view: str, pattern: str
    ) -> Optional[Path]:
        return FileLoader.find_first(self.path(image_id, "images", view, pattern))

    def load_color_image(
        self, image_id: str, frame: str, view: str
    ) -> Optional[np.ndarray]:
        path = self._find(image_id, view, f"frame.{frame}.color.jpg")
        return FileLoader.load_image(path, cv2.COLOR_BGR2RGB)

    def load_depth(
        self, image_id: str, frame: str, view: str
    ) -> Optional[np.ndarray]:
        path = self._find(image_id, view, f"frame.{frame}.depth_meters.hdf5")
        data = FileLoader.load_hdf5(path)
        return data.astype(np.float32) if data is not None else None

    def load_normal(
        self, image_id: str, frame: str, view: str
    ) -> Optional[np.ndarray]:
        path = self._find(image_id, view, f"frame.{frame}.normal_world.hdf5")
        data = FileLoader.load_hdf5(path)
        return data.astype(np.float32) if data is not None else None

    def load_position(
        self, image_id: str, frame: str, view: str
    ) -> Optional[np.ndarray]:
        path = self._find(image_id, view, f"frame.{frame}.position.hdf5")
        data = FileLoader.load_hdf5(path)
        return data.astype(np.float32) if data is not None else None
    
    @staticmethod
    def raydepth2depth(
        ray_depth: np.ndarray, K: np.ndarray
    ) -> np.ndarray:
        """
        Convert ray depth to Z-distance using camera intrinsics.

        Args:
            ray_depth (np.ndarray): Ray depth map (H, W).
            K (np.ndarray): Camera intrinsic matrix (3, 3).

        Returns:
            np.ndarray: Z-distance depth map (H, W).
        """
        if ray_depth.ndim != 2 or K.shape != (3, 3):
            raise ValueError("ray_depth must be 2D and K must be 3x3.")

        K_inv = np.linalg.inv(K)
        h, w = ray_depth.shape
        yy, xx = np.indices((h, w), dtype=np.float32)

        coords = np.stack([xx.ravel(), yy.ravel(), np.ones(h * w, dtype=np.float32)])
        coeffs = np.linalg.norm(K_inv @ coords, axis=0).reshape(h, w)
        coeffs[coeffs == 0] = 1e-6

        return (ray_depth / coeffs).astype(np.float32)

class ETH3DLoader(BaseDataLoader):
    """Loader for MOGE ground truth data."""

    def _find(self, name: str) -> Optional[Path]:
        return FileLoader.find_first(self.path(name))

    def load_color_image(self) -> Optional[np.ndarray]:
        path = self._find("image.jpg")
        return FileLoader.load_image(path, cv2.COLOR_BGR2RGB)

    def load_depth_png(self) -> Optional[np.ndarray]:
        path = self._find("depth.png")
        img = FileLoader.load_image(path, is_grey=True)
        return img.astype(np.float64) if img is not None and img.ndim == 2 else None

    def load_intrinsics(self) -> Optional[np.ndarray]:
        path = self._find("meta.json")
        data = FileLoader.load_json(path)
        if data and "intrinsics" in data:
            k = np.array(data["intrinsics"])
            return k.astype(np.float32) if k.shape == (3, 3) else None
        return None

import torch
from moge.model.v1 import MoGeModel

class MogeLoader(BaseDataLoader):
    """Loader for MOGE ground truth data."""
    
    def __init__(self, base_dir: Union[str, Path]):
        super().__init__(base_dir)
        moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        input_image = torch.tensor(self.load_color_image() / 255, dtype=torch.float32, device=(torch.device("cuda" if torch.cuda.is_available() else "cpu"))).permute(2, 0, 1)
        self.output = moge.infer(input_image)

    def _find(self, name: str) -> Optional[Path]:
        return FileLoader.find_first(self.path(name))

    def load_color_image(self) -> Optional[np.ndarray]:
        path = self._find("image.jpg")
        return FileLoader.load_image(path, cv2.COLOR_BGR2RGB)

    def load_depth_png(self) -> Optional[np.ndarray]:
        depth_map = self.output["depth"].cpu().numpy()
        return depth_map.astype(np.float64) 

    def load_intrinsics(self) -> Optional[np.ndarray]:
      
        k = self.output["intrinsics"].cpu().numpy()

        return k.astype(np.float32) if k.shape == (3, 3) else None
    


class MogeLoader(BaseDataLoader):
    """Loader for MOGE ground truth data."""
    
    def __init__(self, base_dir: Union[str, Path]):
        super().__init__(base_dir)
        moge = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        input_image = torch.tensor(self.load_color_image() / 255, dtype=torch.float32, device=(torch.device("cuda" if torch.cuda.is_available() else "cpu"))).permute(2, 0, 1)
        self.output = moge.infer(input_image)

    def _find(self, name: str) -> Optional[Path]:
        return FileLoader.find_first(self.path(name))

    def load_color_image(self) -> Optional[np.ndarray]:
        path = self._find("image.jpg")
        return FileLoader.load_image(path, cv2.COLOR_BGR2RGB)

    def load_depth_png(self) -> Optional[np.ndarray]:
        depth_map = self.output["depth"].cpu().numpy()
        return depth_map.astype(np.float64) 

    def load_intrinsics(self) -> Optional[np.ndarray]:
      
        k = self.output["intrinsics"].cpu().numpy()

        return k.astype(np.float32) if k.shape == (3, 3) else None

class MogePredLoader(BaseDataLoader):
    """Loader for MOGE predictions."""

    def _find(self, pattern: str) -> Optional[Path]:
        return FileLoader.find_first(self.path("moge", pattern))

    def load_hdf5_map(
        self, file_id: str, suffix: str
    ) -> Optional[np.ndarray]:
        path = self._find(f"frame.{file_id}.{suffix}.hdf5")
        data = FileLoader.load_hdf5(path)
        return data.astype(np.float32) if data is not None else None

    def load_depth(self, file_id: str) -> Optional[np.ndarray]:
        return self.load_hdf5_map(file_id, "depth_meters")

    def load_normal(self, file_id: str) -> Optional[np.ndarray]:
        return self.load_hdf5_map(file_id, "normal_world")

    def load_position(self, file_id: str) -> Optional[np.ndarray]:
        return self.load_hdf5_map(file_id, "position")

    def load_intrinsics(self, file_id: str) -> Optional[np.ndarray]:
        return self.load_hdf5_map(file_id, "K")

    def load_mask(self, file_id: str) -> Optional[np.ndarray]:
        return self.load_hdf5_map(file_id, "mask")

import geocalib
from geocalib import GeoCalib
import torch
import torch.nn.functional as F
import cv2
import numpy as np

from pathlib import Path
from typing import Union, Optional


# (2) GeoCalib import (installed via `pip install -e .` in the GeoCalib repo)
from geocalib import GeoCalib


class GeoMidasLoader(BaseDataLoader):
    """Loader that uses GeoCalib for intrinsics and MiDaS_small for monocular depth."""
    def __init__(self, base_dir: Union[str, Path]):
        super().__init__(base_dir)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # ────────────────────────────────────────────────
        # (A) Instantiate GeoCalib and MiDaS_small
        # ────────────────────────────────────────────────

        # GeoCalib model (estimates a full 3×3 intrinsics K tensor)
        self.geocalib_model = GeoCalib().to(device).eval()

        # MiDaS_small model + its transforms (for depth)
        # This will auto‐download the correct weights on first run.
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(device).eval()
        self.midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        # ────────────────────────────────────────────────
        # (B) Locate & load the color image from base_dir
        # ────────────────────────────────────────────────

        img_path = self._find("image.jpg")
        if img_path is None:
            # No image found → mark as empty
            self.output = None
            return

        # Load color image in H×W×3 RGB format
        img_bgr = FileLoader.load_image(img_path, cv2.COLOR_BGR2RGB)
        if img_bgr is None:
            self.output = None
            return

        img_rgb = img_bgr  # Already H×W×3 in RGB order
        h0, w0 = img_rgb.shape[:2]

        # ────────────────────────────────────────────────
        # (C) Run GeoCalib on that image to get a Pinhole object
        # ────────────────────────────────────────────────

        # GeoCalib.load_image(...) returns a tensor of shape [1, 1, 3, H, W]
        with torch.no_grad():
            geo_input_5d = self.geocalib_model.load_image(str(img_path)).to(device)
            # Squeeze out the singleton “view” dimension (dim=1) → now [1, 3, H, W]
            geo_input_4d = geo_input_5d.squeeze(1)
            geo_result = self.geocalib_model.calibrate(geo_input_4d)

        # Extract the Pinhole‐type camera object
        cam = geo_result.get("camera", None)
        if cam is None:
            self.output = None
            return

        # ────────────────────────────────────────────────
        # (D) Obtain the 3×3 intrinsics K from cam.K
        # ────────────────────────────────────────────────

        # In GeoCalib’s API, `cam.K` is a Tensor of shape [1×3×3] or [3×3].
        K_tensor = cam.K
        if K_tensor.dim() == 3 and K_tensor.size(0) == 1:
            K_tensor = K_tensor.squeeze(0)
        if K_tensor.shape != (3, 3):
            raise RuntimeError(f"Expected intrinsics K of shape [3×3], but got {K_tensor.shape}")
        K = K_tensor.cpu().numpy().astype(np.float32)

        # ────────────────────────────────────────────────
        # (E) Run MiDaS_small to get a low-res depth, then upsample to H×W
        # ────────────────────────────────────────────────

        # 1. Preprocess for MiDaS_small:
        #    small_transform expects a dict {"image": <H×W×3 numpy>}, not a raw numpy.
        sample = self.midas_transforms.small_transform(img_rgb)
        if isinstance(sample, dict):
            image_tensor = sample.get("image", None)
            if image_tensor is None:
                raise RuntimeError("MiDaS small_transform returned dict but no 'image' key found")
        elif torch.is_tensor(sample):
            image_tensor = sample  # already a Tensor
        else:
            raise RuntimeError(f"Unexpected return type from small_transform: {type(sample)}")


        # If 3D, unsqueeze to batch; if 4D, use as-is
        if image_tensor.dim() == 3:
            input_batch = image_tensor.unsqueeze(0).to(device)  # [1, 3, H_low, W_low]
        elif image_tensor.dim() == 4:
            input_batch = image_tensor.to(device)  # already [1, 3, H_low, W_low]
        else:
            raise RuntimeError(f"Unexpected tensor dims from small_transform: {image_tensor.dim()}")

        # 2. Inference (low‐resolution depth)
        with torch.no_grad():
            depth_low = self.midas(input_batch)  # now shape: [1×1×H_low×W_low]


            # **Keep it 4D for interpolate**:
        depth_4d = F.interpolate(
            depth_low.unsqueeze(0),                           # [1,1,192,256]
            size=(h0, w0),                       # e.g. (768, 1024)
            mode="bicubic",
            align_corners=False
        )  # → [1,1,768,1024]

        # Now squeeze off the batch & channel dims to get a 2D [768, 1024] array
        depth_map = depth_4d.squeeze(0).squeeze(0).cpu().numpy().astype(np.float64)


        # ────────────────────────────────────────────────
        # (F) Store results in self.output
        # ────────────────────────────────────────────────

        self.output = {
            "intrinsics": K,       # NumPy array [3×3] float32
            "depth": depth_map      # NumPy array [H0×W0] float64
        }


    def _find(self, name: str) -> Optional[Path]:
        return FileLoader.find_first(self.path(name))

    def load_color_image(self) -> Optional[np.ndarray]:
        """
        Returns the RGB image as a NumPy array (H×W×3, uint8).
        """
        path = self._find("image.jpg")
        if path is None:
            return None
        return FileLoader.load_image(path, cv2.COLOR_BGR2RGB)

    def load_depth_png(self) -> Optional[np.ndarray]:
        """
        Returns the MiDaS_small–predicted depth as a full‐resolution H×W array (float64).
        """
        if self.output is None or "depth" not in self.output:
            return None
        return self.output["depth"]

    def load_intrinsics(self) -> Optional[np.ndarray]:
        """
        Returns the 3×3 float32 intrinsics matrix from GeoCalib.
        """
        if self.output is None or "intrinsics" not in self.output:
            return None
        return self.output["intrinsics"]
