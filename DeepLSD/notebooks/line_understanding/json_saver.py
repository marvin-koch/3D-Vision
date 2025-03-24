import json
import os
import numpy as np

def convert_np(o):
    """Recursively convert NumPy types to native Python types."""
    if isinstance(o, np.generic):
        return o.item()
    elif isinstance(o, dict):
        return {k: convert_np(v) for k, v in o.items()}
    elif isinstance(o, (list, tuple)):
        return [convert_np(i) for i in o]
    return o

def save_lines_to_json(image_id, line_info, output_dir="json_output"):
    """
    Save detected lines along with their type and coplanarity labels into a JSON file.
    """
    lines_data = []
    for entry in line_info:
        if entry["type"] == "structural":
            line_dict = {
                "base_line": entry["base_line"],
                "type": "structural",
                "offset_lines": entry.get("offset_lines", []),
                "coplanarity_labels": entry.get("coplanarity_labels", [])
            }
        else:
            line_dict = {
                "base_line": entry["base_line"],
                "type": "textural",
                "coplanarity_label": entry.get("coplanarity_labels")
            }
        lines_data.append(line_dict)

    json_dict = {
        "image_id": image_id,
        "lines": lines_data
    }

    os.makedirs(output_dir, exist_ok=True)
    json_file = os.path.join(output_dir, f"{image_id}.json")
    with open(json_file, "w") as f:
        # Convert numpy types to native Python types before dumping.
        json.dump(convert_np(json_dict), f, indent=4)
    print(f"Saved JSON data to {json_file}")
