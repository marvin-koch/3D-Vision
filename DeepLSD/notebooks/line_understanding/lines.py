import logging
import cv2
import numpy as np
import torch
import line_understanding.feature_extraction as ft
from line_understanding.geometry import *
from line_understanding.edges import *

from typing import Any, Optional, Tuple, List, Dict


def detect_lines(
    color_img: np.ndarray,
    net: Any,
    device: torch.device
) -> Tuple[np.ndarray, torch.Tensor, float, float]:
    """
    Run DeepLSD line detection and return predicted lines, combined features, downsample ratio, and duration.
    """
    gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    # df_handle = net.df_head[5].register_forward_hook(ft.hook_df)

    # ang_handle = net.angle_head[5].register_forward_hook(ft.hook_angle)
    tensor = torch.tensor(gray, dtype=torch.float32, device=device)[None, None] / 255.0

    with torch.no_grad():
        out = net({'image': tensor})

    # df_handle.remove()
    # ang_handle.remove()

    lines = out['lines'][0]
    df_norm = out['df_norm']
            # Closest line direction prediction
    angle_filed = out['line_level']

    df_np        = df_norm.squeeze(0).cpu().numpy()
    angle_np     = angle_filed.squeeze(0).cpu().numpy()

    assert df_np.shape == angle_np.shape, "df and angle maps should have the same shape"
    H_out, W_out = df_np.shape


    downsample_h = color_img.shape[1] / H_out
    downsample_w = color_img.shape[2] / W_out

    if isinstance(lines, torch.Tensor):
        lines = lines.cpu().numpy()

    #features = torch.cat([ft.df_intermediate_features, ft.angle_intermediate_features], dim=1)


    if device == "cuda":
        del tensor, out
        torch.cuda.empty_cache()

    return lines, df_np, angle_np, downsample_h, downsample_w



def classify_lines(
    lines: np.ndarray,
    tn: np.ndarray,
    td: np.ndarray
) -> Tuple[List[bool], List[bool]]:
    """
    Classify lines as structural and depth-separated.
    """
    is_struct, is_depth_sep = [], []
    for l in lines:
        md, mn = sobel_line(td, tn, l)
        on_d = bool(np.any(md)); on_n = bool(np.any(mn))
        is_struct.append(on_d or on_n)
        is_depth_sep.append(on_d)
    return is_struct, is_depth_sep

def get_line_pixels_trim(line, maps, trim_ratio=0.25):
    """
    Get all pixel coordinates along a line using cv2.line.
    """
  

    p1 = np.array(line[0], dtype=np.float32)
    p2 = np.array(line[1], dtype=np.float32)

    # Direction vector and length
    direction = p2 - p1
    length = np.linalg.norm(direction)
    unit_dir = direction / (length + 1e-8)

    # Shorten line by trim_ratio from both ends
    trim_len = length * trim_ratio
    new_p1 = p1 + unit_dir * trim_len
    new_p2 = p2 - unit_dir * trim_len

    # Convert to integer pixel coordinates
    x1, y1 = int(round(new_p1[0])), int(round(new_p1[1]))
    x2, y2 = int(round(new_p2[0])), int(round(new_p2[1]))

    height, width = maps.shape[:2]

    blank_image = np.zeros((height, width), dtype=np.uint8)
    cv2.line(blank_image, (x1, y1), (x2, y2), color=255, thickness=10) #TODO it was 3 before, we changed it to 1
    
    y_coords, x_coords = np.where(blank_image == 255)
    return list(zip(x_coords, y_coords))




def create_optimal_offset_lines_fast(line, normal_map, offset_amount=1.0, num_samples=100, angle_steps=36):
    p1, p2 = line
    H, W = normal_map.shape[:2]
    xs = np.linspace(p1[0], p2[0], num_samples)
    ys = np.linspace(p1[1], p2[1], num_samples)
    base_pts = np.stack([xs, ys], axis=1)

    best_score = -np.inf
    best_d = None

    for theta in np.linspace(0, np.pi, angle_steps, endpoint=False):
        d = np.array([np.cos(theta), np.sin(theta)])
        offset_vec = offset_amount * d
        pts1 = base_pts + offset_vec
        pts2 = base_pts - offset_vec

        # Vectorized rounding & clipping
        coords1 = np.round(pts1).astype(np.int32)
        coords2 = np.round(pts2).astype(np.int32)
        coords1[:, 0] = np.clip(coords1[:, 0], 0, W - 1)
        coords1[:, 1] = np.clip(coords1[:, 1], 0, H - 1)
        coords2[:, 0] = np.clip(coords2[:, 0], 0, W - 1)
        coords2[:, 1] = np.clip(coords2[:, 1], 0, H - 1)

        normals1 = normal_map[coords1[:, 1], coords1[:, 0]]  # [N, 3]
        normals2 = normal_map[coords2[:, 1], coords2[:, 0]]  # [N, 3]

        score = np.sum(np.linalg.norm(normals1 - normals2, axis=1))
        if score > best_score:
            best_score = score
            best_d = d

    if best_d is None:
        # Handle the case where no valid offset vector was found
        print("Warning: No valid offset found!")
        # You can decide on a default value or return a default behavior
        best_d = np.array([0.0, 0.0])  # Default direction (no offset)

    offset = offset_amount * best_d
    return np.array([p1 + offset, p2 + offset]), np.array([p1 - offset, p2 - offset])


def compute_shifted_line_fast(line, depth_map, w, h, offset=1.0, num_samples=100):
    tangent = line[1] - line[0]
    t_norm = tangent / np.linalg.norm(tangent)
    perp = np.array([-t_norm[1], t_norm[0]]) * offset

    xs = np.linspace(line[0, 0], line[1, 0], num_samples)
    ys = np.linspace(line[0, 1], line[1, 1], num_samples)
    points = np.stack([xs, ys], axis=1)

    pos_pts = points + perp
    neg_pts = points - perp

    pos_coords = np.round(pos_pts).astype(np.int32)
    neg_coords = np.round(neg_pts).astype(np.int32)
    pos_coords[:, 0] = np.clip(pos_coords[:, 0], 0, w - 1)
    pos_coords[:, 1] = np.clip(pos_coords[:, 1], 0, h - 1)
    neg_coords[:, 0] = np.clip(neg_coords[:, 0], 0, w - 1)
    neg_coords[:, 1] = np.clip(neg_coords[:, 1], 0, h - 1)

    depth_pos = depth_map[pos_coords[:, 1], pos_coords[:, 0]]
    depth_neg = depth_map[neg_coords[:, 1], neg_coords[:, 0]]

    avg_depth_pos = np.mean(depth_pos)
    avg_depth_neg = np.mean(depth_neg)

    chosen_offset = perp if avg_depth_pos < avg_depth_neg else -perp
    return line + chosen_offset



def draw_and_split(
    color_img: np.ndarray,
    lines: np.ndarray,
    is_struct: List[bool],
    is_depth_sep: List[bool],
    normal_map: np.ndarray,
    depth_map: np.ndarray,
    struct_color: Tuple[int,int,int],
    text_color: Tuple[int,int,int],
    thickness=1
) -> Tuple[np.ndarray, List[np.ndarray], List[Dict[str,Any]]]:
    """
    Draw lines on image, split or shift based on classification.
    """
    comp = color_img.copy()
    new_ls, info = [], []
    for i, l in enumerate(lines):
        line = l.reshape(2,2) if l.shape==(4,) else l
        

        if is_struct[i] and not is_depth_sep[i]:
            l1, l2 = create_optimal_offset_lines_fast(line, normal_map, offset_amount=1.0)
            new_ls.extend([l1, l2])
            idx1, idx2 = len(new_ls)-2, len(new_ls)-1
            info.append({
                "base_line": line.tolist(),
                "score": 1,
                "offset_lines": [l1.tolist(), l2.tolist()],
                "new_line_indices": [idx1, idx2],
            })
            new_thickness = thickness + 1

            cv2.line(comp, tuple(map(int, map(round, l1[0]))),
                     tuple(map(int, map(round, l1[1]))), (128,0,128), new_thickness)
            cv2.line(comp, tuple(map(int, map(round, l2[0]))),
                     tuple(map(int, map(round, l2[1]))), (0,165,255), new_thickness)
        elif is_struct[i] and is_depth_sep[i]:
            shifted = compute_shifted_line_fast(
                line, depth_map, comp.shape[1], comp.shape[0],
                offset=1.0, num_samples=100
            )
            new_ls.append(shifted)
            idx = len(new_ls)-1
            info.append({
                "base_line": line.tolist(),
                "score": 1,
                "new_line_indices": [idx],
                "shifted": True,
            })
            cv2.line(comp, tuple(map(int, map(round, shifted[0]))),
                     tuple(map(int, map(round, shifted[1]))), struct_color, thickness)
        else:
            new_ls.append(line)
            idx = len(new_ls)-1
            info.append({
                "base_line": line.tolist(),
                "score": 0,
                "new_line_indices": [idx],
            })
            cv2.line(comp, tuple(map(int, map(round, line[0]))),
                     tuple(map(int, map(round, line[1]))), text_color, thickness)

    comp_rgb = cv2.cvtColor(comp, cv2.COLOR_BGR2RGB)
    return comp_rgb, new_ls, info

