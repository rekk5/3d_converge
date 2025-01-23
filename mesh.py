import open3d as o3d
import numpy as np
import argparse

def downsample_pointcloud(pcd, voxel_size=0.05):
    """Downsample point cloud using voxel grid filter"""
    print(f"Points before downsampling: {len(pcd.points)}")
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"Points after downsampling: {len(downsampled_pcd.points)}")
    return downsampled_pcd

def pointcloud_to_mesh(points, method='poisson', downsample=True, voxel_size=0.05, **kwargs):
    """
    Convert point cloud to mesh with downsampling option
    """
    # Create point cloud object
    if isinstance(points, str):
        print("Loading point cloud...")
        pcd = o3d.io.read_point_cloud(points)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

    # Downsample if requested
    if downsample:
        pcd = downsample_pointcloud(pcd, voxel_size)

    # Estimate normals if they don't exist
    if not pcd.has_normals():
        print("Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.orient_normals_consistent_tangent_plane(k=30)

    # Choose reconstruction method
    if method == 'poisson':
        print("Running Poisson surface reconstruction...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd,
            depth=kwargs.get('depth', 8),  # Reduced from 9 to 8 for memory
            width=kwargs.get('width', 0),
            scale=kwargs.get('scale', 1.1),
            linear_fit=kwargs.get('linear_fit', False)
        )
        # Remove low density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
    elif method == 'ball_pivot':
        print("Running Ball Pivoting reconstruction...")
        # Increased radius for downsampled point cloud
        radii = kwargs.get('radii', [0.05, 0.1, 0.2])
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        
    elif method == 'alpha_shape':
        print("Running Alpha Shape reconstruction...")
        alpha = kwargs.get('alpha', 0.1)  # Increased alpha for downsampled cloud
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha
        )
    
    else:
        raise ValueError(f"Unknown method: {method}")

    # Post-processing
    print("Computing vertex normals...")
    mesh.compute_vertex_normals()
    
    return mesh, pcd

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Convert point cloud to mesh')
    parser.add_argument('input_path', help='Path to input PLY file')
    parser.add_argument('output_path', help='Path to output mesh file')
    parser.add_argument('--method', type=str, choices=['poisson', 'ball_pivot', 'alpha_shape'],
                        default='poisson', help='Mesh reconstruction method')
    parser.add_argument('--voxel_size', type=float, default=0.05,
                        help='Voxel size for downsampling (default: 0.05)')
    parser.add_argument('--no_downsample', action='store_true',
                        help='Disable downsampling')
    
    # Method-specific arguments
    parser.add_argument('--depth', type=int, default=8,
                        help='Depth for Poisson reconstruction (default: 8)')
    parser.add_argument('--alpha', type=float, default=0.1,
                        help='Alpha value for Alpha Shape reconstruction (default: 0.1)')
    parser.add_argument('--radii', type=float, nargs='+', default=[0.05, 0.1, 0.2],
                        help='Radii for Ball Pivoting (default: 0.05 0.1 0.2)')
    
    args = parser.parse_args()
    
    # Prepare kwargs based on selected method
    method_kwargs = {}
    if args.method == 'poisson':
        method_kwargs['depth'] = args.depth
    elif args.method == 'alpha_shape':
        method_kwargs['alpha'] = args.alpha
    elif args.method == 'ball_pivot':
        method_kwargs['radii'] = args.radii
    
    # Create mesh
    mesh, pcd = pointcloud_to_mesh(
        args.input_path,
        method=args.method,
        downsample=not args.no_downsample,
        voxel_size=args.voxel_size,
        **method_kwargs
    )
    
    # Visualize result
    print("Visualizing result...")
    o3d.visualization.draw_geometries([pcd, mesh])
    
    # Save mesh
    print(f"Saving mesh to {args.output_path}")
    o3d.io.write_triangle_mesh(args.output_path, mesh)