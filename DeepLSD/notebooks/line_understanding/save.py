# import json
# import os
# import numpy as np

# OUTPUT_DIR = "json_output"

# def convert_np(o):
#     """Recursively convert NumPy scalars to native Python types."""
#     if isinstance(o, np.generic):
#         return o.item()
#     elif isinstance(o, dict):
#         return {k: convert_np(v) for k, v in o.items()}
#     elif isinstance(o, (list, tuple)):
#         return [convert_np(i) for i in o]
#     return o

# def save_lines_to_json(
#     image_id,
#     frame_str,
#     line_info,
#     coplanarity_matrix,
#     features,
#     output_dir=OUTPUT_DIR,
#     save=True,
#     file_path_str=None
# ):
#     """
#     Save line detections to JSON, but write large arrays (.npy) for:
#       - the full coplanarity matrix
#       - each per-line embedding

#     JSON will contain only the file paths.
#     """
#     # ensure output directory exists
#     os.makedirs(output_dir, exist_ok=True)

#     # --- 1) Save the coplanarity matrix ---
#     copla_fname = f"{image_id}_{frame_str}_coplanarity.npz"
#     copla_path = os.path.join(output_dir, copla_fname)
#     np.savez(copla_path, coplanarity_matrix)

#     ft_fname = f"{image_id}_{frame_str}_feature_map.npz"
#     ft_path = os.path.join(output_dir, ft_fname)
#     np.savez(ft_path, features.cpu().numpy())

#     # --- 2) Process each line: dump its embedding ---
#     lines_data = []
#     for idx, entry in enumerate(line_info):
     
#         # record the rest of the metadata
#         lines_data.append({
#             "coordinates": entry["base_line"],
#             "struct_score": entry["score"],
#             # JSON now references the .npy file path
#         })

#     # --- 3) Build JSON dict ---
#     json_dict = {
#         "image_id": image_id,
#         "lines": lines_data,
#         # reference to coplanarity .npy
#         "coplanarity_matrix_path": copla_path,
#         "feature_map_path": ft_path

#     }
#     if file_path_str is not None:
#         json_dict["file_path"] = file_path_str

#     # --- 4) Write JSON file ---
#     if save:
#         json_file = os.path.join(output_dir, f"{image_id}_{frame_str}.json")
#         with open(json_file, "w") as f:
#             json.dump(convert_np(json_dict), f, indent=4)

#     return json_dict



import os
import json
import h5py
import numpy as np

def save_lines_to_hdf5(
    image_id: str,
    frame_str: str,
    line_info: list,
    coplanarity_matrix: np.ndarray,
    df_np: np.ndarray, 
    angle_np: np.ndarray, 
    downsample_h: float , 
    downsample_w : float,  
    h5_path: str = "lines_data.h5",
    file_path_str: str = None,
    compression: str = "gzip",
    compression_level: int = 4
):
    """
    Save line detections and associated arrays into a single HDF5 file.

    Each sample is stored as a group named "{image_id}_{frame_str}" containing:
      - coplanarity: float32 dataset, chunked per-sample, compressed
      - features:    float32 dataset, chunked per-sample, compressed
      - metadata:    JSON-encoded string dataset

    Params:
      file_path:    arbitrary root path for context (stored in metadata)
      image_id, frame_str: identifiers for naming the group
      line_info:    list of dicts with keys "base_line" (coords) and "score"
      coplanarity_matrix: numpy array (N x N)
      features:     numpy array (C x H x W)
      h5_path:      path to the HDF5 file (will be created if not exists)
      compression:  compression filter name (e.g., "gzip", "lzf")
      compression_level: integer level for compression (0-9 for gzip)
    Returns:
      None
    """
    # Ensure output directory exists
    out_dir = os.path.dirname(h5_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Open HDF5 (create or append)
    with h5py.File(h5_path, "a") as h5f:
        grp_name = f"{image_id}_{frame_str}"
        if grp_name in h5f:
            # overwrite existing group
            del h5f[grp_name]
        grp = h5f.create_group(grp_name)

        # Cast arrays to float32 for space savings
        copla32 = coplanarity_matrix.astype(np.float32)
        df32 = df_np.astype(np.float32)
        angle32 = angle_np.astype(np.float32)

        # Create datasets within the group
        grp.create_dataset(
            "coplanarity",
            data=copla32,
            compression=compression,
            compression_opts=compression_level,
            chunks=True
        )
        grp.create_dataset(
            "distance_field",
            data=df32,
            compression=compression,
            compression_opts=compression_level,
            chunks=True
        )

        grp.create_dataset(
            "angle_field",
            data=angle32,
            compression=compression,
            compression_opts=compression_level,
            chunks=True
        )

        # Prepare metadata JSON
        meta = {
            "image_id": image_id,
            "frame": frame_str,
            "file_path": file_path_str if file_path_str is not None else file_path,
            "downsample_h": downsample_h,
            "downsample_w": downsample_w,
            "lines": [
                {"coordinates": entry["base_line"],
                 "struct_score": float(entry["score"])}
                for entry in line_info
            ]
        }
        # Store metadata as a variable-length UTF-8 string
        dt = h5py.string_dtype(encoding="utf-8")
        grp.create_dataset(
            "metadata",
            data=json.dumps(meta),
            dtype=dt
        )
