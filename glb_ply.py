import trimesh
import numpy as np
import argparse
import os
import open3d as o3d  # Added for PCD support
import struct

def convert_to_open3d(trimesh_geometry):
    """
    Convert a trimesh geometry to an Open3D point cloud
    
    Args:
        trimesh_geometry: A trimesh geometry (Trimesh or PointCloud)
        
    Returns:
        open3d.geometry.PointCloud: The converted Open3D point cloud
    """
    o3d_pc = o3d.geometry.PointCloud()
    
    # Extract vertices/points
    if isinstance(trimesh_geometry, trimesh.points.PointCloud):
        points = np.asarray(trimesh_geometry.vertices)
        o3d_pc.points = o3d.utility.Vector3dVector(points)
        
        # Copy colors if available
        if hasattr(trimesh_geometry, 'colors') and trimesh_geometry.colors is not None:
            try:
                colors = np.asarray(trimesh_geometry.colors)
                # Check if colors array is valid and has the right shape
                if colors.size > 0 and colors.ndim == 2:
                    # Make sure we have RGB or RGBA format (3 or 4 columns)
                    if colors.shape[1] >= 3:
                        # Use only the first 3 columns (RGB)
                        colors = colors[:, :3]
                        # Normalize colors to [0, 1] if they're in [0, 255]
                        if colors.max() > 1.0:
                            colors = colors / 255.0
                        o3d_pc.colors = o3d.utility.Vector3dVector(colors)
                    else:
                        print("Warning: Color data doesn't have enough channels, skipping colors")
                else:
                    print("Warning: Invalid color data shape, skipping colors")
            except Exception as e:
                print(f"Warning: Could not convert colors: {str(e)}")
            
        # Copy normals if available
        if hasattr(trimesh_geometry, 'normals') and trimesh_geometry.normals is not None:
            try:
                normals = np.asarray(trimesh_geometry.normals)
                if normals.shape == points.shape:
                    o3d_pc.normals = o3d.utility.Vector3dVector(normals)
                else:
                    print("Warning: Normal data shape doesn't match points, skipping normals")
            except Exception as e:
                print(f"Warning: Could not convert normals: {str(e)}")
            
    elif isinstance(trimesh_geometry, trimesh.Trimesh):
        # For meshes, we sample points on the surface
        points = np.asarray(trimesh_geometry.vertices)
        o3d_pc.points = o3d.utility.Vector3dVector(points)
        
        # Copy vertex colors if available
        if hasattr(trimesh_geometry.visual, 'vertex_colors') and trimesh_geometry.visual.vertex_colors is not None:
            try:
                colors = np.asarray(trimesh_geometry.visual.vertex_colors[:, :3])
                # Normalize colors to [0, 1] if they're in [0, 255]
                if colors.size > 0 and colors.max() > 1.0:
                    colors = colors / 255.0
                o3d_pc.colors = o3d.utility.Vector3dVector(colors)
            except Exception as e:
                print(f"Warning: Could not convert vertex colors: {str(e)}")
            
        # Copy vertex normals if available, otherwise compute them
        if hasattr(trimesh_geometry, 'vertex_normals') and trimesh_geometry.vertex_normals is not None:
            try:
                normals = np.asarray(trimesh_geometry.vertex_normals)
                o3d_pc.normals = o3d.utility.Vector3dVector(normals)
            except Exception as e:
                print(f"Warning: Could not convert vertex normals: {str(e)}")
    
    # If normals are not available, estimate them
    if not o3d_pc.has_normals():
        print("Computing point cloud normals...")
        try:
            o3d_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=0.1, max_nn=30))
            # Orient normals consistently
            o3d_pc.orient_normals_consistent_tangent_plane(k=15)
        except Exception as e:
            print(f"Warning: Could not estimate normals: {str(e)}")
    
    return o3d_pc

def write_nimbus_compatible_ply(o3d_pc, output_path):
    """
    Write a point cloud to PLY format compatible with Nimbus RT
    
    Args:
        o3d_pc: Open3D point cloud
        output_path: Output file path
        
    Returns:
        bool: True if successful
    """
    try:
        # Get points, normals, and colors as numpy arrays
        points = np.asarray(o3d_pc.points)
        normals = np.asarray(o3d_pc.normals) if o3d_pc.has_normals() else np.zeros((len(points), 3))
        colors = np.asarray(o3d_pc.colors) if o3d_pc.has_colors() else np.ones((len(points), 3))
        
        # Convert colors to 0-255 range if they're in 0-1 range
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        
        # Create dummy label and material arrays (all zeros)
        labels = np.zeros(len(points), dtype=np.uint32)
        materials = np.zeros(len(points), dtype=np.uint32)
        
        # Write the PLY file with custom header
        with open(output_path, 'wb') as f:
            # Write header
            header = [
                "ply",
                "format binary_little_endian 1.0",
                "comment Created by Open3D (edited)",
                f"element vertex {len(points)}",
                "property float x",
                "property float y",
                "property float z",
                "property float nx",
                "property float ny",
                "property float nz",
                "property uchar red",
                "property uchar green",
                "property uchar blue",
                "property uint label",
                "property uint material",
                "end_header"
            ]
            header_str = '\n'.join(header) + '\n'
            f.write(header_str.encode())
            
            # Write data
            for i in range(len(points)):
                # Write point coordinates (float32)
                f.write(struct.pack('<fff', 
                                   float(points[i, 0]), 
                                   float(points[i, 1]), 
                                   float(points[i, 2])))
                
                # Write normals (float32)
                f.write(struct.pack('<fff', 
                                   float(normals[i, 0]), 
                                   float(normals[i, 1]), 
                                   float(normals[i, 2])))
                
                # Write colors (uint8)
                f.write(struct.pack('<BBB', 
                                   int(colors[i, 0]), 
                                   int(colors[i, 1]), 
                                   int(colors[i, 2])))
                
                # Write label and material (uint32)
                f.write(struct.pack('<II', labels[i], materials[i]))
        
        print(f"Saved Nimbus RT compatible PLY file to: {output_path}")
        return True
        
    except Exception as e:
        print(f"Error writing Nimbus RT compatible PLY: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def inspect_glb(input_path):
    """
    Inspect a GLB file and print information about its contents
    
    Args:
        input_path (str): Path to the input GLB file
    """
    try:
        # Load the GLB file
        scene = trimesh.load(input_path)
        
        print("\nInspecting GLB file:", input_path)
        print("=" * 20)
        
        # Check if it's a scene or a single mesh
        if isinstance(scene, trimesh.Scene):
            print(f"Scene with {len(scene.geometry)} geometries")
            
            # Print information about each geometry
            for name, geom in scene.geometry.items():
                print(f"  - {name}: {type(geom)}")
                if isinstance(geom, trimesh.Trimesh):
                    print(f"    Vertices: {len(geom.vertices)}, Faces: {len(geom.faces)}")
                elif isinstance(geom, trimesh.points.PointCloud):
                    print(f"    Points: {len(geom.vertices)}")
                    if hasattr(geom, 'colors') and geom.colors is not None:
                        print(f"    Has colors: {geom.colors.shape}")
                    if hasattr(geom, 'normals') and geom.normals is not None:
                        print(f"    Has normals: {geom.normals.shape}")
            
            # Print information about the scene graph
            print("\nScene graph (transformations):")
            try:
                for node_name, node in scene.graph.nodes_geometry.items():
                    print(f"  - {node_name}: {node}")
            except Exception as e:
                print(f"  Scene graph nodes_geometry is not a dictionary: {type(scene.graph.nodes_geometry)}")
        else:
            print("Single geometry (not a scene)")
            if isinstance(scene, trimesh.Trimesh):
                print(f"  Vertices: {len(scene.vertices)}, Faces: {len(scene.faces)}")
            elif isinstance(scene, trimesh.points.PointCloud):
                print(f"  Points: {len(scene.vertices)}")
                if hasattr(scene, 'colors') and scene.colors is not None:
                    print(f"  Has colors: {scene.colors.shape}")
                if hasattr(scene, 'normals') and scene.normals is not None:
                    print(f"  Has normals: {scene.normals.shape}")
        
        print("\n=== Inspection complete ===")
        
    except Exception as e:
        print(f"Error inspecting GLB file: {str(e)}")
        import traceback
        traceback.print_exc()

def convert_glb_to_point_cloud(input_path, output_path=None, output_format='ply', extract_pointcloud=False, separate_meshes=False, nimbus_compatible=False):
    """
    Convert a GLB file to point cloud format (PLY or PCD)
    
    Args:
        input_path (str): Path to the input GLB file
        output_path (str, optional): Path for the output file. If not provided,
                                    will use the same name as input with appropriate extension
        output_format (str): Format to save the point cloud ('ply' or 'pcd')
        extract_pointcloud (bool): If True, extract only point clouds from the scene
        separate_meshes (bool): If True, export each mesh as a separate file
        nimbus_compatible (bool): If True, create PLY files compatible with Nimbus RT
    
    Returns:
        str or list: Path(s) to the created file(s)
    """
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = f"{base_name}.{output_format}"
    
    print(f"Loading GLB file: {input_path}")
    
    try:
        # Load the GLB file
        scene = trimesh.load(input_path)
        
        # Check if it's a scene or a single mesh
        if isinstance(scene, trimesh.Scene):
            print(f"Scene detected with {len(scene.geometry)} geometries")
            geometries = list(scene.geometry.values())
        else:
            print(f"Single mesh detected")
            geometries = [scene]
        
        # Filter for point clouds if requested
        if extract_pointcloud:
            pointclouds = [g for g in geometries if isinstance(g, trimesh.points.PointCloud)]
            print(f"Found {len(pointclouds)} point cloud(s) in the scene")
            
            if len(pointclouds) == 0:
                print("No point clouds found in the scene")
                return None
            
            geometries = pointclouds
        
        # Process the geometries
        result_paths = []
        for i, geom in enumerate(geometries):
            # Generate output path for this geometry
            if separate_meshes:
                base_name = os.path.splitext(output_path)[0]
                geom_path = f"{base_name}_{i}.{output_format}"
            else:
                geom_path = output_path
            
            print(f"Exporting point cloud to: {geom_path}")
            
            # Convert to Open3D format
            o3d_pc = convert_to_open3d(geom)
            
            # Export based on format and compatibility requirements
            if output_format == 'ply' and nimbus_compatible:
                success = write_nimbus_compatible_ply(o3d_pc, geom_path)
                if not success:
                    print(f"Failed to create Nimbus RT compatible PLY file")
                    return None
            else:
                # Standard export
                if output_format == 'pcd':
                    o3d.io.write_point_cloud(geom_path, o3d_pc)
                else:
                    o3d.io.write_point_cloud(geom_path, o3d_pc, write_ascii=False)
                    print(f"Exported point cloud with normals to: {geom_path}")
            
            result_paths.append(geom_path)
            
            # If not exporting separately, just use the first geometry
            if not separate_meshes:
                break
        
        if len(result_paths) == 1:
            print(f"Successfully converted to: {result_paths[0]}")
            return result_paths[0]
        else:
            print(f"Successfully converted to {len(result_paths)} files")
            return result_paths
        
    except Exception as e:
        print(f"Error during conversion: \n{str(e)}")
        import traceback
        traceback.print_exc()
        print("Conversion failed.")
        return None

# Alias for backward compatibility
convert_glb_to_ply = lambda input_path, output_path=None, extract_pointcloud=False, separate_meshes=False: \
    convert_glb_to_point_cloud(input_path, output_path, 'ply', extract_pointcloud, separate_meshes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert GLB files to point cloud formats (PLY or PCD)")
    parser.add_argument("input", help="Input GLB file path")
    parser.add_argument("-o", "--output", help="Output file path (optional)")
    parser.add_argument("--format", choices=['ply', 'pcd'], default='ply', 
                        help="Output format (ply or pcd, default: ply)")
    parser.add_argument("--inspect", action="store_true", help="Only inspect the GLB file without converting")
    parser.add_argument("--pointcloud", action="store_true", help="Extract only point clouds")
    parser.add_argument("--separate", action="store_true", help="Export each geometry as a separate file")
    parser.add_argument("--nimbus", action="store_true", help="Create PLY files compatible with Nimbus RT")
    
    args = parser.parse_args()
    
    # Print output format and file
    if not args.inspect:
        print(f"Output format: {args.format.upper()}")
        if args.output:
            print(f"Output file: {args.output}")
    
    # Inspect the GLB file if requested
    if args.inspect:
        inspect_glb(args.input)
    else:
        # Convert the GLB file to point cloud
        result = convert_glb_to_point_cloud(
            args.input, 
            args.output,
            output_format=args.format,
            extract_pointcloud=args.pointcloud,
            separate_meshes=args.separate,
            nimbus_compatible=args.nimbus
        )