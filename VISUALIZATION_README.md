# 3D Pipeline Visualization Tool

This tool provides comprehensive visualization capabilities for the 3D reconstruction pipeline, allowing you to view point clouds, meshes, and meshes with diffraction edges from customizable camera positions and angles.

## Features

- **Multi-step visualization**: View point clouds, meshes, and diffraction edges separately or in sequence
- **Customizable camera parameters**: Set position, look-at point, up vector, field of view, and zoom
- **Interactive mode**: Choose which step to visualize
- **Auto-advance mode**: Automatically cycle through all available steps
- **Corner view support**: Specialized views for corner positions with wide FOV
- **File auto-detection**: Automatically finds pipeline output files

**Note**: Open3D doesn't support direct FOV control, so FOV is approximated using zoom level adjustments.

## Installation

Make sure you have the required dependencies:

```bash
pip install open3d numpy matplotlib plyfile
```

## Usage

### Basic Usage

```bash
# Visualize all available pipeline steps with default camera
python visualization.py

# Visualize with custom camera position (corner view)
python visualization.py --camera_position "10,10,10" --camera_fov 90

# Visualize specific step only
python visualization.py --step point_cloud

# Auto-advance through steps
python visualization.py --auto_advance
```

### Camera Parameters

The visualization tool supports the following camera parameters:

- **`--camera_position`**: Camera position as "x,y,z" (default: "5,5,5")
- **`--camera_look_at`**: Point to look at as "x,y,z" (default: "0,0,0")
- **`--camera_up`**: Up vector as "x,y,z" (default: "0,1,0")
- **`--camera_fov`**: Field of view in degrees (approximated via zoom, default: 60.0)
- **`--camera_zoom`**: Zoom level (default: 0.8, auto-adjusted based on FOV)

### Input Files

- **`--point_cloud`**: Path to point cloud file (.ply)
- **`--mesh`**: Path to mesh file (.ply)
- **`--mesh_with_edges`**: Path to mesh with diffraction edges (.ply)

If no files are specified, the tool will automatically look for common pipeline output files in the current directory.

## Example Camera Configurations

### Corner View (Wide FOV)
```bash
python visualization.py \
  --camera_position "10,10,10" \
  --camera_look_at "0,0,0" \
  --camera_up "0,1,0" \
  --camera_fov 90 \
  --camera_zoom 0.6
```

### Close-up View (Narrow FOV)
```bash
python visualization.py \
  --camera_position "2,2,2" \
  --camera_look_at "0,0,0" \
  --camera_up "0,1,0" \
  --camera_fov 30 \
  --camera_zoom 1.2
```

### Top-down View
```bash
python visualization.py \
  --camera_position "0,15,0" \
  --camera_look_at "0,0,0" \
  --camera_up "0,0,-1" \
  --camera_fov 60 \
  --camera_zoom 0.8
```

### Side View
```bash
python visualization.py \
  --camera_position "15,0,0" \
  --camera_look_at "0,0,0" \
  --camera_up "0,1,0" \
  --camera_fov 60 \
  --camera_zoom 0.8
```

## Example Scripts

The `example_visualization.py` script provides pre-configured camera setups:

```bash
# Corner view with wide FOV
python example_visualization.py corner

# Close-up view with narrow FOV
python example_visualization.py close

# Top-down view
python example_visualization.py top

# Side view
python example_visualization.py side

# Interactive camera setup
python example_visualization.py interactive
```

## Interactive Mode

When running without `--auto_advance`, the tool will present a menu:

```
=== 3D Pipeline Visualization ===
Camera parameters:
  Position: [5, 5, 5]
  Look at: [0, 0, 0]
  Up vector: [0, 1, 0]
  FOV: 60.0°
  Zoom: 0.8

Found 3 steps to visualize:
  1. Point Cloud: point_cloud.ply
  2. Mesh: mesh.ply
  3. Mesh with Diffraction Edges: saalas_mesh.ply

Choose visualization step:
  1. Point Cloud
  2. Mesh
  3. Mesh with Diffraction Edges
  0. Exit

Enter choice (0-3):
```

## Controls

Once in the visualization window:

- **Mouse**: Rotate, pan, and zoom
- **Q**: Exit visualization
- **R**: Reset view
- **Arrow keys**: Rotate view
- **Mouse wheel**: Zoom in/out

## File Format Support

The visualization tool supports:

- **Point clouds**: PLY files with point data
- **Meshes**: PLY files with vertex and face data
- **Meshes with diffraction edges**: PLY files with additional edge data

### Expected PLY Structure for Meshes with Edges

```python
# Vertex element
vertex = [
    ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
    ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
    ('label', 'u4'), ('material', 'u4')
]

# Face element
face = [('vertex_indices', 'i4', (3,))]

# Edge element (for diffraction edges)
edge = [
    ('start_x', 'f4'), ('start_y', 'f4'), ('start_z', 'f4'),
    ('end_x', 'f4'), ('end_y', 'f4'), ('end_z', 'f4'),
    ('normal1_x', 'f4'), ('normal1_y', 'f4'), ('normal1_z', 'f4'),
    ('normal2_x', 'f4'), ('normal2_y', 'f4'), ('normal2_z', 'f4'),
    ('plane1', 'u4'), ('plane2', 'u4'),
    ('material1', 'u4'), ('material2', 'u4')
]
```

## Integration with Pipeline

The visualization tool is designed to work seamlessly with the 3D reconstruction pipeline:

1. **Point Cloud**: After reconstruction (COLMAP, TSDF, MAST3R)
2. **Mesh**: After segmentation and mesh generation
3. **Mesh with Edges**: After diffraction edge detection

### Typical Workflow

```bash
# Run the pipeline
python main.py --image_path images/ --output_dir output/

# Visualize results with corner view
python visualization.py \
  --point_cloud output/point_cloud.ply \
  --mesh_with_edges output/saalas_mesh.ply \
  --camera_position "8,8,8" \
  --camera_fov 75 \
  --auto_advance
```

## Troubleshooting

### Common Issues

1. **No visualization window appears**: Check if you have a display server running (X11 on Linux)
2. **Files not found**: Ensure the pipeline has been run and output files exist
3. **Poor visualization quality**: Adjust camera parameters, especially FOV and zoom
4. **Performance issues**: Reduce point cloud density or use smaller FOV

### Debug Mode

For debugging, you can add verbose output:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Advanced Usage

### Custom Camera Calculations

For precise camera positioning, you can calculate optimal parameters:

```python
from visualization import PipelineVisualizer

# Load point cloud to get bounds
pcd = o3d.io.read_point_cloud("point_cloud.ply")
bounds = pcd.get_axis_aligned_bounding_box()
center = bounds.get_center()
extent = bounds.get_extent()

# Calculate camera position based on bounds
max_extent = max(extent)
camera_distance = max_extent * 2
camera_position = center + [camera_distance, camera_distance, camera_distance]

# Create visualizer with calculated parameters
visualizer = PipelineVisualizer()
visualizer.set_camera_params(
    position=camera_position.tolist(),
    look_at=center.tolist(),
    fov=60.0,
    zoom=0.8
)
```

### Batch Visualization

For processing multiple datasets:

```bash
#!/bin/bash
for dataset in dataset1 dataset2 dataset3; do
    echo "Processing $dataset"
    python visualization.py \
        --point_cloud "$dataset/point_cloud.ply" \
        --mesh_with_edges "$dataset/saalas_mesh.ply" \
        --camera_position "10,10,10" \
        --camera_fov 90 \
        --step mesh_with_edges
done
```

## Contributing

To extend the visualization tool:

1. Add new camera presets to `example_visualization.py`
2. Implement additional file format support
3. Add new visualization modes (e.g., wireframe, point size adjustment)
4. Create custom color schemes for different materials

## License

This visualization tool is part of the 3D reconstruction pipeline and follows the same license terms. 