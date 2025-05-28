import json
import os
import numpy as np

OUTPUT_DIR = "scannet_output"

def convert_np(o):
    """Recursively convert NumPy scalars to native Python types."""
    if isinstance(o, np.generic):
        return o.item()
    elif isinstance(o, dict):
        return {k: convert_np(v) for k, v in o.items()}
    elif isinstance(o, (list, tuple)):
        return [convert_np(i) for i in o]
    return o

def save_lines_to_json(
    image_id,
    frame_str,
    line_info,
    coplanarity_matrix,
    output_dir=OUTPUT_DIR,
    save=True,
    file_path_str=None
):
    """
    Save line detections to JSON, but write large arrays (.npy) for:
      - the full coplanarity matrix
      - each per-line embedding

    JSON will contain only the file paths.
    """
    # ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # --- 1) Save the coplanarity matrix ---
    copla_fname = f"{image_id}_{frame_str}_coplanarity.npy"
    copla_path = os.path.join(output_dir, copla_fname)
    np.save(copla_path, coplanarity_matrix)

    # --- 2) Process each line: dump its embedding ---
    lines_data = []
    for idx, entry in enumerate(line_info):
        # save embedding
        emb_fname = f"{image_id}_{frame_str}_emb_{idx:03d}.npy"
        emb_path = os.path.join(output_dir, emb_fname)
        np.save(emb_path, entry["line_embedding"])

        # record the rest of the metadata
        lines_data.append({
            "coordinates": entry["base_line"],
            "struct_score": entry["score"],
            # JSON now references the .npy file path
            "embedding_DeepLSD_path": emb_path
        })

    # --- 3) Build JSON dict ---
    json_dict = {
        "image_id": image_id,
        "lines": lines_data,
        # reference to coplanarity .npy
        "coplanarity_matrix_path": copla_path
    }
    if file_path_str is not None:
        json_dict["file_path"] = file_path_str

    # --- 4) Write JSON file ---
    if save:
        json_file = os.path.join(output_dir, f"{image_id}_{frame_str}.json")
        with open(json_file, "w") as f:
            json.dump(convert_np(json_dict), f, indent=4)

    return json_dict
