import json
import os
import numpy as np

def convert_np(o):
    """Recursively convert NumPy types to native Python types."""
    if isinstance(o, np.ndarray):  # Convert NumPy arrays to lists
        return o.tolist()
    elif isinstance(o, np.generic):  # Convert NumPy scalars (e.g., np.int32, np.float64)
        return o.item()
    elif isinstance(o, dict):  # Recursively process dictionaries
        return {k: convert_np(v) for k, v in o.items()}
    elif isinstance(o, (list, tuple)):  # Recursively process lists/tuples
        return [convert_np(i) for i in o]
    return o  # Return unchanged if already a native Python type


def save_lines_to_json(image_id, line_info, coplanarity_matrix, output_dir="json_output", save=True):
    """
    Save detected lines along with their type and coplanarity labels into a JSON file.

    {
        image_id: "image_id",
        lines:[{coordinates: [ab,cd], confidence_score:0.8}, {coords: [ef,gh], confidence_score:0.6}]
        coplanarity_matrix (n_original_lines x n_original_lines):[........]
    },


    """
    lines_data = []
    for entry in line_info:
        line_dict = {
            "coordinates": entry["base_line"],
            "confidence_score": entry["score"],
            
        }
        lines_data.append(line_dict)

    json_dict = {
        "image_id": image_id,
        "lines": lines_data,
        "coplanarity_matrix": coplanarity_matrix
    }

    if save:
        os.makedirs(output_dir, exist_ok=True)
        json_file = os.path.join(output_dir, f"{image_id}.json")
        with open(json_file, "w") as f:
            # Convert numpy types to native Python types before dumping.
            json.dump(convert_np(json_dict), f, indent=4)

    return json_dict
