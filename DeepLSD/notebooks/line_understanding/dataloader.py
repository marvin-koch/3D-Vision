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
    def load_image(path: Optional[Path], color_conv: Optional[int] = None) -> Optional[np.ndarray]:
        if path is None or not path.exists():
            logging.error("Image not found: %s", path)
            return None
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
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
        img = FileLoader.load_image(path)
        return img.astype(np.float64) if img is not None and img.ndim == 2 else None

    def load_intrinsics(self) -> Optional[np.ndarray]:
        path = self._find("meta.json")
        data = FileLoader.load_json(path)
        if data and "intrinsics" in data:
            k = np.array(data["intrinsics"])
            return k.astype(np.float32) if k.shape == (3, 3) else None
        return None


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
