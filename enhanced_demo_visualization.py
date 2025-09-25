import open3d as o3d
import numpy as np
import copy
from comparison import load_differential_ply
from create_demo_pointclouds import run_differential_demo

def enhance_visualization_colors(point_cloud):
    """Enhance the colors of a point cloud for better visualization"""
    colors = np.asarray(point_cloud.colors)
    
    # Increase saturation and brightness slightly
    colors = np.clip(colors * 1.2, 0, 1)
    
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud

def create_side_by_side_visualization(base_path, modified_path, differential_path):
    """Create a side-by-side visualization of all three point clouds"""
    
    # Load the point clouds
    print("Loading point clouds for visualization...")
    base_cloud = o3d.io.read_point_cloud(base_path)
    modified_cloud = o3d.io.read_point_cloud(modified_path)
    
    # Load differential cloud with custom format
    try:
        diff_cloud, diff_flags = load_differential_ply(differential_path)
    except:
        # Fallback to regular loading if custom format fails
        diff_cloud = o3d.io.read_point_cloud(differential_path)
    
    # Enhance colors
    base_cloud = enhance_visualization_colors(base_cloud)
    modified_cloud = enhance_visualization_colors(modified_cloud)
    
    # Position clouds side by side
    spacing = 12  # Distance between scenes
    
    # Original base scene (left)
    base_display = copy.deepcopy(base_cloud)
    base_translation = np.array([-spacing, 0, 0])
    base_display.translate(base_translation)
    
    # Modified scene (center) - no translation needed
    modified_display = copy.deepcopy(modified_cloud)
    
    # Differential result (right) - enhanced red color
    diff_display = copy.deepcopy(diff_cloud)
    diff_display.paint_uniform_color([1.0, 0.1, 0.1])  # Bright red for differences
    diff_translation = np.array([spacing, 0, 0])
    diff_display.translate(diff_translation)
    
    # Create text labels (using coordinate frames as labels)
    def create_text_marker(position, scale=0.5):
        """Create a coordinate frame as a text marker"""
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=scale)
        coord_frame.translate(position)
        return coord_frame
    
    # Add coordinate frames as markers
    base_marker = create_text_marker(base_translation + [0, 4, 0])
    base_marker.paint_uniform_color([0.2, 0.8, 0.2])  # Green for base
    
    modified_marker = create_text_marker([0, 4, 0])
    modified_marker.paint_uniform_color([0.2, 0.2, 0.8])  # Blue for modified
    
    diff_marker = create_text_marker(diff_translation + [0, 4, 0])
    diff_marker.paint_uniform_color([0.8, 0.2, 0.2])  # Red for differences
    
    print(f"\nVisualization Summary:")
    print(f"  LEFT (Green marker): Base scene ({len(base_cloud.points)} points)")
    print(f"  CENTER (Blue marker): Modified scene ({len(modified_cloud.points)} points)")
    print(f"  RIGHT (Red marker): Differential points ({len(diff_cloud.points)} points)")
    print(f"\nThe differential cloud shows only the points that are different between base and modified scenes.")
    
    # Combine all geometries
    geometries = [
        base_display, modified_display, diff_display,
        base_marker, modified_marker, diff_marker
    ]
    
    return geometries

def create_overlay_visualization(base_path, modified_path, differential_path):
    """Create an overlay visualization showing differences on the modified scene"""
    
    print("Creating overlay visualization...")
    
    # Load point clouds
    modified_cloud = o3d.io.read_point_cloud(modified_path)
    
    try:
        diff_cloud, diff_flags = load_differential_ply(differential_path)
    except:
        diff_cloud = o3d.io.read_point_cloud(differential_path)
    
    # Create a copy of the modified scene with muted colors
    overlay_cloud = copy.deepcopy(modified_cloud)
    overlay_colors = np.asarray(overlay_cloud.colors)
    
    # Mute the original colors (make them more gray)
    muted_colors = overlay_colors * 0.3 + 0.4  # Mix with gray
    overlay_cloud.colors = o3d.utility.Vector3dVector(muted_colors)
    
    # Highlight the differential points in bright red
    diff_display = copy.deepcopy(diff_cloud)
    diff_display.paint_uniform_color([1.0, 0.0, 0.0])  # Bright red
    
    print(f"Overlay visualization:")
    print(f"  Muted colors: Original scene")
    print(f"  Bright red: Differential points ({len(diff_cloud.points)} points)")
    
    return [overlay_cloud, diff_display]

def run_enhanced_demo():
    """Run the complete enhanced demo with beautiful visualizations"""
    
    print("🎯 Creating Enhanced Point Cloud Differential Demo\n")
    
    # Generate demo data and run differential analysis
    base_path, modified_path, differential_path = run_differential_demo()
    
    print("\n" + "="*60)
    print("🎨 ENHANCED VISUALIZATIONS")
    print("="*60)
    
    # Create side-by-side visualization
    print("\n1. Side-by-Side Comparison:")
    side_by_side_geometries = create_side_by_side_visualization(base_path, modified_path, differential_path)
    
    print("\nShowing side-by-side visualization...")
    print("   - Use mouse to rotate, zoom, and pan")
    print("   - Press 'Q' to close and continue to overlay view")
    
    # Set up nice viewing parameters
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Side-by-Side Comparison", width=1400, height=800)
    
    for geometry in side_by_side_geometries:
        vis.add_geometry(geometry)
    
    # Set viewing parameters
    ctr = vis.get_view_control()
    ctr.set_zoom(0.6)
    
    vis.run()
    vis.destroy_window()
    
    # Create overlay visualization
    print("\n2. Overlay Visualization:")
    overlay_geometries = create_overlay_visualization(base_path, modified_path, differential_path)
    
    print("\nShowing overlay visualization...")
    print("   - Gray: Original scene")
    print("   - Red: New/different points")
    print("   - Press 'Q' to close")
    
    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name="Overlay Visualization", width=1000, height=800)
    
    for geometry in overlay_geometries:
        vis2.add_geometry(geometry)
    
    # Set viewing parameters
    ctr2 = vis2.get_view_control()
    ctr2.set_zoom(0.8)
    
    vis2.run()
    vis2.destroy_window()
    
    print("\n" + "="*60)
    print("✅ DEMO COMPLETE!")
    print("="*60)
    print(f"\nGenerated files:")
    print(f"  📁 {base_path}")
    print(f"  📁 {modified_path}")
    print(f"  📁 {differential_path}")
    print(f"\n🎯 The differential analysis successfully identified and isolated")
    print(f"   the new objects (laptop, coffee cup, books, plant) that were")
    print(f"   added to the modified scene!")

def quick_single_cloud_demo():
    """Quick demo showing just the differential result with nice colors"""
    
    print("🚀 Quick Differential Demo")
    
    # Generate demo data
    base_path, modified_path, differential_path = run_differential_demo()
    
    # Load and enhance the differential result
    try:
        diff_cloud, diff_flags = load_differential_ply(differential_path)
    except:
        diff_cloud = o3d.io.read_point_cloud(differential_path)
    
    # Create a beautiful gradient color scheme for the differences
    points = np.asarray(diff_cloud.points)
    colors = np.zeros_like(points)
    
    # Create gradient based on height (Y coordinate)
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    normalized_y = (points[:, 1] - y_min) / (y_max - y_min) if y_max > y_min else np.zeros(len(points))
    
    # Create a nice red-orange-yellow gradient
    colors[:, 0] = 1.0  # Red component
    colors[:, 1] = normalized_y * 0.7  # Green component (creates orange-yellow)
    colors[:, 2] = 0.1  # Blue component
    
    diff_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    print(f"\nShowing {len(points)} differential points with height-based gradient coloring")
    print("Press 'Q' to close")
    
    o3d.visualization.draw_geometries(
        [diff_cloud],
        window_name="Differential Points - Gradient View",
        width=1000,
        height=800
    )

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        quick_single_cloud_demo()
    else:
        run_enhanced_demo() 