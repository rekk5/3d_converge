import open3d as o3d
import numpy as np
import argparse

def create_differential_pointcloud(source_cloud, target_cloud, threshold=0.005, voxel_size=0.02):
    """
    Create a differential point cloud where only the differences are highlighted
    
    Args:
        source_cloud: Path to source point cloud or o3d.geometry.PointCloud
        target_cloud: Path to target point cloud or o3d.geometry.PointCloud
        threshold: Distance threshold for differences (default: 0.005 meters)
        voxel_size: Size of voxels for downsampling (default: 0.02 meters)
    """
    # Load point clouds if paths are provided
    if isinstance(source_cloud, str):
        source = o3d.io.read_point_cloud(source_cloud)
    else:
        source = source_cloud

    if isinstance(target_cloud, str):
        target = o3d.io.read_point_cloud(target_cloud)
    else:
        target = target_cloud

    # Voxel downsampling
    print("Downsampling point clouds...")
    source_down = source.voxel_down_sample(voxel_size)
    target_down = target.voxel_down_sample(voxel_size)
    
    print(f"Source points: {len(source.points)} -> {len(source_down.points)}")
    print(f"Target points: {len(target.points)} -> {len(target_down.points)}")

    # Pre-align the point clouds using their centroids
    source_centroid = np.mean(np.asarray(source_down.points), axis=0)
    target_centroid = np.mean(np.asarray(target_down.points), axis=0)
    
    # Create initial transformation to align centroids
    init_translation = np.eye(4)
    init_translation[:3, 3] = target_centroid - source_centroid
    source_down.transform(init_translation)
    
    # Register downsampled point clouds starting from centroid-aligned position
    print("Performing registration...")
    result_icp = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold,
        np.identity(4),  # Start from identity since we're already pre-aligned
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    # Apply both transformations to original source cloud
    source.transform(init_translation)
    source.transform(result_icp.transformation)
    
    # Compute differences using full resolution clouds
    distances_target_to_source = np.asarray(target.compute_point_cloud_distance(source))
    
    # Create differential cloud from full resolution target
    diff_cloud = o3d.geometry.PointCloud()
    diff_cloud.points = target.points
    
    if target.has_colors():
        diff_cloud.colors = target.colors
    else:
        diff_cloud.paint_uniform_color([0.5, 0.5, 0.5])
    
    colors = np.asarray(diff_cloud.colors)
    different_points_mask = distances_target_to_source > threshold
    colors[different_points_mask] = [1, 0, 0]  # Red color
    diff_cloud.colors = o3d.utility.Vector3dVector(colors)
    
    # Print statistics
    total_points = len(np.asarray(target.points))
    diff_points = np.sum(different_points_mask)
    print("\nResults:")
    print(f"Total points: {total_points}")
    print(f"Different points: {diff_points}")
    print(f"Percentage different: {(diff_points/total_points)*100:.2f}%")
    print(f"Registration fitness: {result_icp.fitness}")
    print(f"Registration RMSE: {result_icp.inlier_rmse}")

    o3d.visualization.draw_geometries([diff_cloud])
    
    return diff_cloud

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Create differential point cloud comparison')
    parser.add_argument('source_path', help='Path to source PLY file')
    parser.add_argument('target_path', help='Path to target PLY file')
    parser.add_argument('--threshold', type=float, default=0.005,
                        help='Distance threshold for differences (default: 0.005 meters)')
    parser.add_argument('--voxel_size', type=float, default=0.02,
                        help='Voxel size for downsampling (default: 0.02 meters)')
    
    args = parser.parse_args()
    
    # Create differential point cloud with command line arguments
    diff_cloud = create_differential_pointcloud(
        args.source_path,
        args.target_path,
        threshold=args.threshold,
        voxel_size=args.voxel_size
    )
