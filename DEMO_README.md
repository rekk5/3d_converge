# 🎯 Differential Point Cloud Demo

This demo showcases the differential point cloud analysis feature with realistic 3D scenes.

## 📝 What This Demo Does

The demo creates two related point cloud scenes:
1. **Base Scene**: A room with basic furniture (table, chairs, walls, floor)
2. **Modified Scene**: The same room with new objects added (laptop, coffee cup, books, potted plant)

Then it runs differential analysis to identify **only the points that are different** between the two scenes.

## 🚀 Quick Start

### Basic Demo
```bash
conda activate sam3d
python create_demo_pointclouds.py
```

### Enhanced Visualizations
```bash
# Full demo with side-by-side and overlay views
python enhanced_demo_visualization.py

# Quick gradient visualization only  
python enhanced_demo_visualization.py quick
```

## 📊 Demo Results

The differential analysis typically achieves:
- **99.7%+ registration fitness** (excellent alignment)
- **~18% differential points** (representing the new objects)
- **Precise isolation** of added objects while ignoring unchanged areas

## 🎨 Visualization Modes

### 1. Side-by-Side Comparison
- **Left (Green marker)**: Base scene
- **Center (Blue marker)**: Modified scene  
- **Right (Red marker)**: Differential points only

### 2. Overlay Visualization
- **Gray/Muted**: Original scene points
- **Bright Red**: Differential points highlighted

### 3. Gradient View
- **Height-based gradient**: Red → Orange → Yellow based on Y-coordinate

## 📁 Generated Files

| File | Description | Size |
|------|-------------|------|
| `demo_base_scene.ply` | Original room scene | ~338KB |
| `demo_modified_scene.ply` | Room with new objects | ~411KB |
| `demo_differential_result.ply` | Only the differences | ~57KB |

## 🔧 Customization

### Adjust Analysis Parameters
```python
# In create_demo_pointclouds.py, modify:
create_differential_pointcloud_with_custom_ply(
    base_path,
    modified_path,
    output_path,
    threshold=0.1,     # Distance threshold for differences
    voxel_size=0.05    # Voxel size for downsampling
)
```

### Create Your Own Scenes
Modify the `create_base_scene()` and `create_modified_scene()` functions to:
- Add different furniture/objects
- Change room layout
- Adjust colors and materials
- Add more complex geometry

## 🎯 Real-World Applications

This differential analysis technique is useful for:
- **Construction monitoring**: Detecting changes in building sites
- **Quality control**: Identifying manufacturing defects or additions
- **Archaeology**: Documenting excavation progress
- **Robotics**: Environment change detection
- **Security**: Monitoring for unauthorized objects

## 💡 Technical Details

The differential analysis:
1. **Loads** both point clouds
2. **Downsamples** for faster processing
3. **Aligns** clouds using centroid + ICP registration
4. **Computes** point-to-point distances
5. **Filters** points above the difference threshold
6. **Saves** results with custom PLY format including differential flags

## 🛠️ Dependencies

- `open3d` - Point cloud processing and visualization
- `numpy` - Numerical computations
- `plyfile` - Custom PLY file format support

## 📞 Usage in Your Code

```python
from comparison import create_differential_pointcloud_with_custom_ply

# Run differential analysis
diff_cloud, flags = create_differential_pointcloud_with_custom_ply(
    source_cloud="your_base.ply",
    target_cloud="your_modified.ply", 
    output_path="differences.ply",
    threshold=0.05,
    voxel_size=0.02
)
``` 