import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import argparse
import sys
from sklearn.neighbors import KDTree


def densify_plane(plane_points, normal, density=0.03):
    """Add points to a plane segment using a regular grid."""
    points = np.asarray(plane_points.points)
    
    # Project points onto the plane
    basis = np.eye(3) - np.outer(normal, normal)
    projected_points = points @ basis.T
    
    # Find bounds of the projected points
    min_bounds = np.min(projected_points, axis=0)
    max_bounds = np.max(projected_points, axis=0)
    
    # Create regular grid
    x = np.arange(min_bounds[0], max_bounds[0], density)
    y = np.arange(min_bounds[1], max_bounds[1], density)
    xx, yy = np.meshgrid(x, y)
    
    # Create new points
    grid_points_2d = np.column_stack((xx.flatten(), yy.flatten(), np.zeros_like(xx.flatten())))
    
    # Project back to 3D
    grid_points_3d = grid_points_2d @ basis
    
    # Use KDTree to remove points too far from original surface
    tree = KDTree(points)
    distances, _ = tree.query(grid_points_3d, k=1)
    mask = distances.flatten() < density * 2
    
    new_points = grid_points_3d[mask]
    
    # Create new point cloud
    dense_plane = o3d.geometry.PointCloud()
    dense_plane.points = o3d.utility.Vector3dVector(new_points)
    
    return dense_plane


def process_point_cloud(input_path, output_path, max_planes=12, voxel_size=0.03, densify=True):
    print(f"Processing {input_path}...")
    
    # Load point cloud
    try:
        pcd = o3d.io.read_point_cloud(input_path)
        print("Original point cloud:", len(np.asarray(pcd.points)), "points")
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        sys.exit(1)

    # Center the point cloud
    pcd_center = pcd.get_center()
    pcd.translate(-pcd_center)
    
    # Skip outlier removal, use original point cloud
    filtered_pcd = pcd
    
    # Estimate normals
    print("Estimating normals...")
    nn_distance = np.mean(filtered_pcd.compute_nearest_neighbor_distance())
    radius_normals = nn_distance * 4
    filtered_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_normals, 
            max_nn=16
        ),
        fast_normal_computation=True
    )
    
    # Plane segmentation
    print(f"Segmenting up to {max_planes} planes...")
    segment_models = {}
    segments = {}
    rest = filtered_pcd
    all_point_clouds = []
    
    for i in range(max_planes):
        if len(np.asarray(rest.points)) < 1000:  # Stop if too few points remain
            break
            
        print(f"\nProcessing plane {i+1}/{max_planes}...")
        print(f"Points remaining: {len(np.asarray(rest.points))}")
        colors = plt.get_cmap("tab20")(i)
        
        # Segment plane
        plane_model, inliers = rest.segment_plane(
            distance_threshold=0.03,
            ransac_n=3,
            num_iterations=20000
        )
        
        # Print plane equation and orientation
        [a, b, c, d] = plane_model
        print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        normal = np.array([a, b, c])
        angle_with_up = np.arccos(np.abs(np.dot(normal, [0, 0, 1]))) * 180 / np.pi
        print(f"Angle with vertical: {angle_with_up:.2f} degrees")
        
        # Extract plane and color it
        segments[i] = rest.select_by_index(inliers)
        print(f"Points in this plane: {len(inliers)}")
        segments[i].paint_uniform_color(list(colors[:3]))
        
        if densify:
            dense_segment = densify_plane(segments[i], normal)
            dense_segment.paint_uniform_color(list(colors[:3]))
            segments[i] = dense_segment
            
        all_point_clouds.append(segments[i])
        
        # Update remaining points
        rest = rest.select_by_index(inliers, invert=True)
    
    # Add remaining points
    if len(np.asarray(rest.points)) > 0:
        print(f"\nRemaining points: {len(np.asarray(rest.points))}")
        rest.paint_uniform_color([0.7, 0.7, 0.7])
        all_point_clouds.append(rest)
    
    # Combine all segments
    print("\nCombining segments...")
    combined_point_cloud = all_point_clouds[0]
    for pc in all_point_clouds[1:]:
        combined_point_cloud += pc
    
    # Translate back to original position
    combined_point_cloud.translate(pcd_center)
    
    # Save result
    print(f"Saving result to {output_path}...")
    try:
        o3d.io.write_point_cloud(output_path, combined_point_cloud)
        print("Processing complete!")
    except Exception as e:
        print(f"Error saving point cloud: {e}")
        sys.exit(1)
    
    return combined_point_cloud

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Segment planes in a point cloud.')
    parser.add_argument('input_path', help='Path to input PLY file')
    parser.add_argument('--output_path', help='Path to output PLY file', default=None)
    parser.add_argument('--max_planes', type=int, default=12, help='Maximum number of planes to segment')
    parser.add_argument('--voxel_size', type=float, default=0.03, help='Voxel size for downsampling')
    parser.add_argument('--densify', action='store_true', help='Densify plane segments')
    
    args = parser.parse_args()
    
    # If output path not specified, create one based on input path
    if args.output_path is None:
        args.output_path = args.input_path.rsplit('.', 1)[0] + '_segmented.ply'
    
    # Process the point cloud
    process_point_cloud(
        args.input_path,
        args.output_path,
        args.max_planes,
        args.voxel_size,
        args.densify
    )

if __name__ == "__main__":
    main()