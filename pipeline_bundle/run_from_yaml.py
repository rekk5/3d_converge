#!/usr/bin/env python3
import os
import sys
import yaml
import shlex
import subprocess

CONFIG_DEFAULT = "/home/t/Desktop/3d_converge/pipeline_bundle/config.yaml"
MAIN = "/home/t/Desktop/3d_converge/main.py"

# Map YAML to CLI args without changing code

def build_command(cfg: dict) -> list:
    args = [sys.executable, MAIN]

    # general
    if cfg.get('image_path'): args += ['--image_path', cfg['image_path']]
    if cfg.get('output_dir'): args += ['--output_dir', cfg['output_dir']]
    if cfg.get('project_name'): args += ['--project_name', cfg['project_name']]

    # reconstruction
    if cfg.get('point_cloud_path'): args += ['--point_cloud_path', cfg['point_cloud_path']]
    if cfg.get('recon_method'): args += ['--recon_method', cfg['recon_method']]

    # VGGT
    vggt = cfg.get('vggt', {})
    if 'conf_threshold' in vggt: args += ['--conf_threshold', str(vggt['conf_threshold'])]
    if 'fps' in vggt: args += ['--fps', str(vggt['fps'])]
    if vggt.get('show_cameras', True):
        args += ['--show_cameras']
    else:
        args += ['--no_cameras']

    # TSDF
    tsdf = cfg.get('tsdf', {})
    if tsdf.get('tsdf_data_path'): args += ['--tsdf_data_path', tsdf['tsdf_data_path']]
    if 'voxel_length' in tsdf: args += ['--voxel_length', str(tsdf['voxel_length'])]
    if 'sdf_trunc' in tsdf: args += ['--sdf_trunc', str(tsdf['sdf_trunc'])]

    # Comparison
    comp = cfg.get('comparison', {})
    if comp.get('enabled'):
        if comp.get('base_cloud_path'): args += ['--base_cloud_path', comp['base_cloud_path']]
        if 'threshold' in comp: args += ['--comparison_threshold', str(comp['threshold'])]
        if 'voxel_size' in comp: args += ['--comparison_voxel_size', str(comp['voxel_size'])]
        if comp.get('selective_segmentation'): args += ['--selective_segmentation']
        if comp.get('enhanced', True): args += ['--enhanced_comparison']
        if comp.get('use_multi_scale', True): args += ['--use_multi_scale']
        if comp.get('use_confidence', True): args += ['--use_confidence']
        if comp.get('use_color_coding', True): args += ['--use_color_coding']

    # Segmentation
    seg = cfg.get('segmentation', {})
    if not seg.get('enabled', True):
        args += ['--skip_segmentation']
    else:
        if 'voxel_size' in seg: args += ['--voxel_size', str(seg['voxel_size'])]
        if 'max_planes' in seg: args += ['--max_planes', str(seg['max_planes'])]
        if 'distance_threshold' in seg: args += ['--distance_threshold', str(seg['distance_threshold'])]
        if seg.get('enhanced'): args += ['--enhanced_segmentation']

    # Diffraction
    diff = cfg.get('diffraction', {})
    if not diff.get('enabled', True):
        args += ['--skip_diffraction']
    else:
        if diff.get('method') in ('segmentation', 'normals'):
            args += ['--diffraction_method', diff['method']]
        if 'angle_threshold' in diff: args += ['--angle_threshold', str(diff['angle_threshold'])]
        if 'min_edge_length' in diff: args += ['--min_edge_length', str(diff['min_edge_length'])]
        normals = diff.get('normals', {})
        if 'normal_radius' in normals: args += ['--normal_radius', str(normals['normal_radius'])]
        if 'normal_max_nn' in normals: args += ['--normal_max_nn', str(normals['normal_max_nn'])]
        if 'edge_radius' in normals: args += ['--edge_radius', str(normals['edge_radius'])]
        if 'voxel_downsampling' in normals: args += ['--voxel_downsampling', str(normals['voxel_downsampling'])]
        if 'sor_neighbors' in normals: args += ['--sor_neighbors', str(normals['sor_neighbors'])]
        if 'sor_std_ratio' in normals: args += ['--sor_std_ratio', str(normals['sor_std_ratio'])]

    return args


def main():
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else CONFIG_DEFAULT
    if not os.path.exists(cfg_path):
        print(f"Config not found: {cfg_path}")
        sys.exit(1)

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    cmd = build_command(cfg)
    print("Running:")
    print(" ", ' '.join(shlex.quote(c) for c in cmd))

    env = os.environ.copy()
    # Respect conda env if active; otherwise rely on system python
    try:
        subprocess.run(cmd, check=True, env=env)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
