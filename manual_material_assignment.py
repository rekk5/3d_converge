import argparse
import os
import sys
import yaml
import numpy as np
import open3d as o3d
from plyfile import PlyData, PlyElement


MATERIAL_MAP = {
    'vacuum': 0,
    'itu_concrete': 1,
    'itu_brick': 2,
    'itu_plasterboard': 3,
    'itu_wood': 4,
    'itu_glass': 5,
    'itu_ceiling_board': 6,
    'itu_chipboard': 7,
    'itu_plywood': 8,
    'itu_marble': 9,
    'itu_floorboard': 10,
    'itu_metal': 11,
    'itu_very_dry_ground': 12,
    'itu_medium_dry_ground': 13,
    'itu_wet_ground': 14,
}
ID_TO_MATERIAL = {v: k for k, v in MATERIAL_MAP.items()}
DEFAULT_MATERIAL_NAME = 'itu_concrete'


def validate_input_file(file_path: str) -> bool:
    if not os.path.isfile(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    _, ext = os.path.splitext(file_path)
    if ext.lower() != '.ply':
        print("Error: Only .ply files are supported.")
        return False
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if len(np.asarray(pcd.points)) == 0:
            print("Error: Empty point cloud.")
            return False
        if not pcd.has_colors():
            print("Error: Input point cloud must be segmented (have colors).")
            return False
        return True
    except Exception as e:
        print(f"Error reading point cloud: {e}")
        return False


def find_color_segments(pcd: o3d.geometry.PointCloud, atol: float = 1e-3):
    colors = np.asarray(pcd.colors)
    # Round to reduce floating noise and get stable unique colors
    rounded = np.round(colors / atol) * atol
    unique_colors, inverse_idx = np.unique(rounded, axis=0, return_inverse=True)
    segments = {}
    for seg_id, color in enumerate(unique_colors):
        indices = np.where(inverse_idx == seg_id)[0]
        segments[seg_id] = {
            'indices': indices,
            'color': color,
            'size': len(indices),
        }
    return segments


def list_materials():
    print("\nAvailable materials:")
    print("-" * 40)
    for name, mid in sorted(MATERIAL_MAP.items(), key=lambda kv: kv[1]):
        print(f"{mid:2d}: {name}")


def load_mapping(path: str):
    with open(path, 'r') as f:
        data = yaml.safe_load(f) or {}
    mapping = {}
    for k, v in data.items():
        try:
            seg_id = int(k)
        except Exception:
            print(f"Warning: Ignoring non-integer segment key: {k}")
            continue
        if isinstance(v, int):
            # treat as material id
            if v in ID_TO_MATERIAL:
                mapping[seg_id] = v
            else:
                print(f"Warning: Unknown material id {v} for segment {seg_id}; using default")
        elif isinstance(v, str):
            if v in MATERIAL_MAP:
                mapping[seg_id] = MATERIAL_MAP[v]
            else:
                print(f"Warning: Unknown material name '{v}' for segment {seg_id}; using default")
        else:
            print(f"Warning: Unsupported mapping value for segment {seg_id}: {v}; using default")
    return mapping


def save_mapping(path: str, mapping: dict):
    # Save mapping using material names for readability
    out = {int(seg): ID_TO_MATERIAL.get(int(mat), int(mat)) for seg, mat in mapping.items()}
    with open(path, 'w') as f:
        yaml.safe_dump(out, f, sort_keys=True)
    print(f"Saved mapping to {path}")


def interactive_assign(segments: dict, existing_mapping: dict | None = None) -> dict:
    mapping = dict(existing_mapping or {})
    print("\nDetected segments:")
    print("-" * 60)
    for seg_id, info in segments.items():
        material_name = ID_TO_MATERIAL.get(mapping.get(seg_id, MATERIAL_MAP[DEFAULT_MATERIAL_NAME]), DEFAULT_MATERIAL_NAME)
        col = (info['color'] * 255).astype(int)
        print(f"Segment {seg_id:3d} | points: {info['size']:7d} | color RGB: ({col[0]:3d},{col[1]:3d},{col[2]:3d}) | material: {material_name}")

    list_materials()
    print("\nEnter material for each segment by name or id. Press Enter to keep current/default.")
    for seg_id in sorted(segments.keys()):
        current_mid = mapping.get(seg_id, MATERIAL_MAP[DEFAULT_MATERIAL_NAME])
        current_name = ID_TO_MATERIAL.get(current_mid, str(current_mid))
        while True:
            user = input(f"Segment {seg_id} material [{current_name}]: ").strip()
            if user == '':
                mapping[seg_id] = current_mid
                break
            # try parse id
            if user.isdigit():
                mid = int(user)
                if mid in ID_TO_MATERIAL:
                    mapping[seg_id] = mid
                    break
                else:
                    print("Invalid id. Choose from list above.")
                    continue
            # parse name
            if user in MATERIAL_MAP:
                mapping[seg_id] = MATERIAL_MAP[user]
                break
            else:
                print("Invalid name. Choose from list above or use id.")
                continue
    return mapping


def write_point_cloud_with_materials(pcd: o3d.geometry.PointCloud, segments: dict, mapping: dict, output_path: str):
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    normals = np.asarray(pcd.normals) if pcd.has_normals() else np.zeros_like(points)

    # Build per-vertex label and material arrays
    num_points = len(points)
    labels = np.zeros(num_points, dtype=np.uint32)
    materials = np.zeros(num_points, dtype=np.uint32)

    for seg_id, info in segments.items():
        idx = info['indices']
        labels[idx] = seg_id
        materials[idx] = int(mapping.get(seg_id, MATERIAL_MAP[DEFAULT_MATERIAL_NAME]))

    colors_uint8 = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)

    vertex_data = np.empty(num_points, dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('label', 'u4'), ('material', 'u4')
    ])

    vertex_data['x'] = points[:, 0].astype(np.float32)
    vertex_data['y'] = points[:, 1].astype(np.float32)
    vertex_data['z'] = points[:, 2].astype(np.float32)
    vertex_data['red'] = colors_uint8[:, 0]
    vertex_data['green'] = colors_uint8[:, 1]
    vertex_data['blue'] = colors_uint8[:, 2]
    vertex_data['nx'] = normals[:, 0].astype(np.float32)
    vertex_data['ny'] = normals[:, 1].astype(np.float32)
    vertex_data['nz'] = normals[:, 2].astype(np.float32)
    vertex_data['label'] = labels
    vertex_data['material'] = materials

    ply = PlyData([PlyElement.describe(vertex_data, 'vertex')], text=True)
    ply.write(output_path)
    print(f"Saved: {output_path}")
    print(f"Points: {num_points}")


def main():
    parser = argparse.ArgumentParser(description='Manually assign materials to segmented point cloud segments and save as PLY with labels/materials.')
    parser.add_argument('input_ply', type=str, help='Path to input segmented .ply file')
    parser.add_argument('output_ply', type=str, help='Path to output .ply file with label/material properties')
    parser.add_argument('--load_map', type=str, default=None, help='Path to YAML mapping file to preload (segment_id: material_name_or_id)')
    parser.add_argument('--save_map', type=str, default=None, help='Path to save YAML mapping after assignment')
    parser.add_argument('--non_interactive', action='store_true', help='Do not prompt; require --load_map or use default for all segments')
    args = parser.parse_args()

    if not validate_input_file(args.input_ply):
        sys.exit(1)

    pcd = o3d.io.read_point_cloud(args.input_ply)

    # Ensure normals exist for completeness in output
    if not pcd.has_normals():
        try:
            pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        except Exception:
            pass

    segments = find_color_segments(pcd)
    if len(segments) == 0:
        print("Error: No segments detected.")
        sys.exit(1)

    print(f"Detected {len(segments)} segments.")

    mapping = {}
    if args.load_map:
        if os.path.isfile(args.load_map):
            print(f"Loading mapping from {args.load_map}")
            mapping = load_mapping(args.load_map)
        else:
            print(f"Warning: --load_map file not found: {args.load_map}")

    if args.non_interactive:
        # Fill missing segments with default
        for seg_id in segments.keys():
            mapping.setdefault(seg_id, MATERIAL_MAP[DEFAULT_MATERIAL_NAME])
    else:
        mapping = interactive_assign(segments, mapping)

    # Optionally save mapping
    if args.save_map:
        save_mapping(args.save_map, mapping)

    write_point_cloud_with_materials(pcd, segments, mapping, args.output_ply)


if __name__ == '__main__':
    main() 