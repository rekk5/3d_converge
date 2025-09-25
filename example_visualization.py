#!/usr/bin/env python3
"""
Example script demonstrating the 3D Pipeline Visualization Tool.

This script shows how to use the visualization tool with different camera configurations
and pipeline outputs.
"""

import os
import sys
from visualization import PipelineVisualizer


def example_corner_view():
    """Example: View from a corner position with wide FOV."""
    print("=== Example: Corner View ===")
    
    visualizer = PipelineVisualizer()
    
    # Set camera to corner position with wide FOV
    visualizer.set_camera_params(
        position=[10, 10, 10],    # Camera in corner
        look_at=[0, 0, 0],        # Looking at origin
        up=[0, 1, 0],             # Y-axis up
        fov=90.0,                 # Wide field of view
        zoom=0.6                  # Slightly zoomed out
    )
    
    # Try to find and visualize pipeline outputs
    visualizer.visualize_pipeline_steps(auto_advance=True)


def example_close_up_view():
    """Example: Close-up view with narrow FOV."""
    print("=== Example: Close-up View ===")
    
    visualizer = PipelineVisualizer()
    
    # Set camera for close-up view
    visualizer.set_camera_params(
        position=[2, 2, 2],       # Close to the scene
        look_at=[0, 0, 0],        # Looking at origin
        up=[0, 1, 0],             # Y-axis up
        fov=30.0,                 # Narrow field of view
        zoom=1.2                  # Zoomed in
    )
    
    # Try to find and visualize pipeline outputs
    visualizer.visualize_pipeline_steps(auto_advance=True)


def example_top_down_view():
    """Example: Top-down view."""
    print("=== Example: Top-down View ===")
    
    visualizer = PipelineVisualizer()
    
    # Set camera for top-down view
    visualizer.set_camera_params(
        position=[0, 15, 0],      # High above the scene
        look_at=[0, 0, 0],        # Looking straight down
        up=[0, 0, -1],            # Z-axis up (for top-down)
        fov=60.0,                 # Standard field of view
        zoom=0.8                  # Standard zoom
    )
    
    # Try to find and visualize pipeline outputs
    visualizer.visualize_pipeline_steps(auto_advance=True)


def example_side_view():
    """Example: Side view."""
    print("=== Example: Side View ===")
    
    visualizer = PipelineVisualizer()
    
    # Set camera for side view
    visualizer.set_camera_params(
        position=[15, 0, 0],      # To the side of the scene
        look_at=[0, 0, 0],        # Looking at origin
        up=[0, 1, 0],             # Y-axis up
        fov=60.0,                 # Standard field of view
        zoom=0.8                  # Standard zoom
    )
    
    # Try to find and visualize pipeline outputs
    visualizer.visualize_pipeline_steps(auto_advance=True)


def example_custom_view():
    """Example: Custom view with specific files."""
    print("=== Example: Custom View ===")
    
    visualizer = PipelineVisualizer()
    
    # Set custom camera parameters
    visualizer.set_camera_params(
        position=[8, 6, 12],      # Custom position
        look_at=[1, 0, 2],        # Custom look at point
        up=[0, 1, 0],             # Y-axis up
        fov=75.0,                 # Custom FOV
        zoom=0.7                  # Custom zoom
    )
    
    # Specify custom file paths (modify these to match your files)
    # Use consistent naming convention
    from naming_convention import get_naming_convention
    naming = get_naming_convention()

    point_cloud_path = naming.file_patterns['mast3r_point_cloud']  # Or other reconstruction method
    mesh_path = naming.file_patterns['diffraction_mesh']
    
    # Check if files exist
    if os.path.exists(point_cloud_path):
        print(f"Visualizing point cloud: {point_cloud_path}")
        visualizer.visualize_point_cloud(point_cloud_path, "Custom Point Cloud View")
    
    if os.path.exists(mesh_path):
        print(f"Visualizing mesh: {mesh_path}")
        visualizer.visualize_mesh_with_edges(mesh_path, "Custom Mesh View")


def interactive_camera_setup():
    """Interactive camera setup example."""
    print("=== Interactive Camera Setup ===")
    
    visualizer = PipelineVisualizer()
    
    print("Enter camera parameters:")
    
    try:
        # Get camera position
        pos_input = input("Camera position (x,y,z) [5,5,5]: ").strip()
        if pos_input:
            position = [float(x) for x in pos_input.split(',')]
        else:
            position = [5, 5, 5]
        
        # Get look at point
        look_input = input("Look at point (x,y,z) [0,0,0]: ").strip()
        if look_input:
            look_at = [float(x) for x in look_input.split(',')]
        else:
            look_at = [0, 0, 0]
        
        # Get up vector
        up_input = input("Up vector (x,y,z) [0,1,0]: ").strip()
        if up_input:
            up = [float(x) for x in up_input.split(',')]
        else:
            up = [0, 1, 0]
        
        # Get FOV
        fov_input = input("Field of view (degrees) [60.0]: ").strip()
        fov = float(fov_input) if fov_input else 60.0
        
        # Get zoom
        zoom_input = input("Zoom level [0.8]: ").strip()
        zoom = float(zoom_input) if zoom_input else 0.8
        
        # Set camera parameters
        visualizer.set_camera_params(
            position=position,
            look_at=look_at,
            up=up,
            fov=fov,
            zoom=zoom
        )
        
        print(f"\nCamera set to:")
        print(f"  Position: {position}")
        print(f"  Look at: {look_at}")
        print(f"  Up: {up}")
        print(f"  FOV: {fov}°")
        print(f"  Zoom: {zoom}")
        
        # Visualize pipeline steps
        visualizer.visualize_pipeline_steps(auto_advance=False)
        
    except (ValueError, KeyboardInterrupt) as e:
        print(f"Error: {e}")
        print("Using default camera parameters.")
        visualizer.visualize_pipeline_steps(auto_advance=False)


def main():
    """Main function to run examples."""
    print("3D Pipeline Visualization Examples")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        example = sys.argv[1].lower()
        
        if example == "corner":
            example_corner_view()
        elif example == "close":
            example_close_up_view()
        elif example == "top":
            example_top_down_view()
        elif example == "side":
            example_side_view()
        elif example == "custom":
            example_custom_view()
        elif example == "interactive":
            interactive_camera_setup()
        else:
            print(f"Unknown example: {example}")
            print_usage()
    else:
        print_usage()


def print_usage():
    """Print usage information."""
    print("\nUsage:")
    print("  python example_visualization.py [example]")
    print("\nExamples:")
    print("  corner      - Corner view with wide FOV")
    print("  close       - Close-up view with narrow FOV")
    print("  top         - Top-down view")
    print("  side        - Side view")
    print("  custom      - Custom view with specific files")
    print("  interactive - Interactive camera setup")
    print("\nOr run without arguments to see this help.")


if __name__ == "__main__":
    main() 