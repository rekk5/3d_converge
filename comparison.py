import open3d as o3d
import numpy as np
import argparse
from plyfile import PlyData, PlyElement
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.cluster import DBSCAN
import cv2

def create_differential_pointcloud(source_cloud, target_cloud, threshold=0.15, voxel_size=0.02, save_with_flag=True):
    """
    Create a differential point cloud containing only the points that differ from the base cloud
    
    Args:
        source_cloud: Path to source point cloud or o3d.geometry.PointCloud (base cloud)
        target_cloud: Path to target point cloud or o3d.geometry.PointCloud (new cloud)
        threshold: Distance threshold for differences (default: 0.04 meters)
        voxel_size: Size of voxels for downsampling (default: 0.02 meters)
        save_with_flag: Whether to save with differential flag property
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
    
    # Create differential cloud containing ONLY the different points
    different_points_mask = distances_target_to_source > threshold
    
    # Extract only the different points from the target cloud
    diff_cloud = target.select_by_index(np.where(different_points_mask)[0])
    
    # Ensure we have normals
    if not diff_cloud.has_normals():
        diff_cloud.estimate_normals()
    
    # Set colors for differential points (red for differences)
    diff_cloud.paint_uniform_color([1, 0, 0])  # Red color for all differential points
    
    # Create differential flags: 1 for all differential points (since we're only keeping different ones)
    differential_flags = np.ones(len(np.asarray(diff_cloud.points)), dtype=np.uint8)
    
    # Print statistics
    total_target_points = len(np.asarray(target.points))
    diff_points = len(np.asarray(diff_cloud.points))
    print("\nResults:")
    print(f"Total target points: {total_target_points}")
    print(f"Differential points (kept): {diff_points}")
    print(f"Points removed (within threshold): {total_target_points - diff_points}")
    print(f"Percentage different: {(diff_points/total_target_points)*100:.2f}%")
    print(f"Registration fitness: {result_icp.fitness}")
    print(f"Registration RMSE: {result_icp.inlier_rmse}")

    o3d.visualization.draw_geometries([diff_cloud])
    
    return diff_cloud, differential_flags

def enhanced_point_cloud_comparison(source_cloud, target_cloud, threshold=0.15, voxel_size=0.02, 
                                   use_multi_scale=True, use_confidence=True, use_color_coding=True):
    """
    Enhanced point cloud comparison with multiple analysis methods
    
    Args:
        source_cloud: Path to source point cloud or o3d.geometry.PointCloud (base cloud)
        target_cloud: Path to target point cloud or o3d.geometry.PointCloud (new cloud)
        threshold: Distance threshold for differences (default: 0.15 meters)
        voxel_size: Size of voxels for downsampling (default: 0.02 meters)
        use_multi_scale: Whether to use multi-scale analysis
        use_confidence: Whether to compute confidence scores
        use_color_coding: Whether to use color-coded differences
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

    print("=== ENHANCED POINT CLOUD COMPARISON ===")
    print(f"Source points: {len(source.points)}")
    print(f"Target points: {len(target.points)}")
    
    # Multi-scale analysis
    if use_multi_scale:
        print("\n--- Multi-scale Analysis ---")
        scales = [voxel_size * 0.5, voxel_size, voxel_size * 2.0]
        multi_scale_results = {}
        
        for scale in scales:
            print(f"Analyzing at scale: {scale:.3f}m")
            source_down = source.voxel_down_sample(scale)
            target_down = target.voxel_down_sample(scale)
            
            # Registration at this scale
            source_centroid = np.mean(np.asarray(source_down.points), axis=0)
            target_centroid = np.mean(np.asarray(target_down.points), axis=0)
            
            init_translation = np.eye(4)
            init_translation[:3, 3] = target_centroid - source_centroid
            source_down.transform(init_translation)
            
            result_icp = o3d.pipelines.registration.registration_icp(
                source_down, target_down, threshold,
                np.identity(4),
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            
            multi_scale_results[scale] = {
                'fitness': result_icp.fitness,
                'rmse': result_icp.inlier_rmse,
                'transformation': result_icp.transformation
            }
            print(f"  Scale {scale:.3f}m - Fitness: {result_icp.fitness:.4f}, RMSE: {result_icp.inlier_rmse:.4f}")
    
    # Use the best scale for final registration
    if use_multi_scale:
        best_scale = max(multi_scale_results.keys(), key=lambda x: multi_scale_results[x]['fitness'])
        print(f"\nUsing best scale: {best_scale:.3f}m")
        source_down = source.voxel_down_sample(best_scale)
        target_down = target.voxel_down_sample(best_scale)
    else:
        source_down = source.voxel_down_sample(voxel_size)
        target_down = target.voxel_down_sample(voxel_size)
    
    # Enhanced registration with multiple methods
    print("\n--- Enhanced Registration ---")
    
    # Method 1: Point-to-point ICP
    source_centroid = np.mean(np.asarray(source_down.points), axis=0)
    target_centroid = np.mean(np.asarray(target_down.points), axis=0)
    
    init_translation = np.eye(4)
    init_translation[:3, 3] = target_centroid - source_centroid
    source_down.transform(init_translation)
    
    result_icp_pt2pt = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold,
        np.identity(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    # Method 2: Point-to-plane ICP (if normals available)
    if source_down.has_normals() and target_down.has_normals():
        result_icp_pt2plane = o3d.pipelines.registration.registration_icp(
            source_down, target_down, threshold,
            np.identity(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane())
        
        # Use the better result
        if result_icp_pt2plane.fitness > result_icp_pt2pt.fitness:
            final_result = result_icp_pt2plane
            print("Using point-to-plane ICP (better fitness)")
        else:
            final_result = result_icp_pt2pt
            print("Using point-to-point ICP (better fitness)")
    else:
        final_result = result_icp_pt2pt
        print("Using point-to-point ICP (no normals available)")
    
    # Apply transformation to original source cloud
    source.transform(init_translation)
    source.transform(final_result.transformation)
    
    # Compute distances and differences
    print("\n--- Computing Differences ---")
    distances_target_to_source = np.asarray(target.compute_point_cloud_distance(source))
    distances_source_to_target = np.asarray(source.compute_point_cloud_distance(target))
    
    # Enhanced difference detection with confidence
    different_points_mask = distances_target_to_source > threshold
    
    # Confidence scoring
    if use_confidence:
        print("\n--- Confidence Analysis ---")
        # Compute local density for confidence
        target_points = np.asarray(target.points)
        source_points = np.asarray(source.points)
        
        # KNN for local density estimation
        target_tree = o3d.geometry.KDTreeFlann(target)
        source_tree = o3d.geometry.KDTreeFlann(source)
        
        confidence_scores = np.zeros(len(target_points))
        
        for i, point in enumerate(target_points):
            # Find neighbors in target cloud
            [k_target, idx_target, _] = target_tree.search_knn_vector_3d(point, min(10, len(target_points)))
            # Find neighbors in source cloud
            [k_source, idx_source, _] = source_tree.search_knn_vector_3d(point, min(10, len(source_points)))
            
            # Local density ratio as confidence
            local_density_target = k_target
            local_density_source = k_source
            
            if local_density_source > 0:
                confidence_scores[i] = min(1.0, local_density_target / local_density_source)
            else:
                confidence_scores[i] = 1.0  # High confidence if no source neighbors
        
        # Adjust threshold based on confidence
        adjusted_threshold = threshold * (1 + (1 - confidence_scores) * 0.5)
        different_points_mask = distances_target_to_source > adjusted_threshold
        
        print(f"Confidence scores - Min: {confidence_scores.min():.3f}, Max: {confidence_scores.max():.3f}, Mean: {confidence_scores.mean():.3f}")
    
    # Extract differential points
    diff_cloud = target.select_by_index(np.where(different_points_mask)[0])
    
    # Color coding based on difference magnitude
    if use_color_coding:
        print("\n--- Color Coding Differences ---")
        diff_distances = distances_target_to_source[different_points_mask]
        
        # Normalize distances for color mapping
        min_dist = diff_distances.min()
        max_dist = diff_distances.max()
        
        if max_dist > min_dist:
            normalized_distances = (diff_distances - min_dist) / (max_dist - min_dist)
        else:
            normalized_distances = np.zeros_like(diff_distances)
        
        # Create color map (blue -> green -> yellow -> red)
        colors = np.zeros((len(diff_distances), 3))
        for i, norm_dist in enumerate(normalized_distances):
            if norm_dist < 0.25:
                # Blue to green
                colors[i] = [0, norm_dist * 4, 1 - norm_dist * 4]
            elif norm_dist < 0.5:
                # Green to yellow
                colors[i] = [(norm_dist - 0.25) * 4, 1, 0]
            elif norm_dist < 0.75:
                # Yellow to orange
                colors[i] = [1, 1 - (norm_dist - 0.5) * 4, 0]
            else:
                # Orange to red
                colors[i] = [1, 0, 0]
        
        diff_cloud.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Default red color
        diff_cloud.paint_uniform_color([1, 0, 0])
    
    # Ensure normals
    if not diff_cloud.has_normals():
        diff_cloud.estimate_normals()
    
    # Statistical analysis
    print("\n--- Statistical Analysis ---")
    total_target_points = len(np.asarray(target.points))
    diff_points = len(np.asarray(diff_cloud.points))
    
    if diff_points > 0:
        diff_distances = distances_target_to_source[different_points_mask]
        print(f"Difference statistics:")
        print(f"  Mean difference: {diff_distances.mean():.4f}m")
        print(f"  Std difference: {diff_distances.std():.4f}m")
        print(f"  Min difference: {diff_distances.min():.4f}m")
        print(f"  Max difference: {diff_distances.max():.4f}m")
        print(f"  Median difference: {np.median(diff_distances):.4f}m")
        
        # Spatial clustering of differences
        if diff_points > 10:
            diff_points_array = np.asarray(diff_cloud.points)
            clustering = DBSCAN(eps=threshold*2, min_samples=5).fit(diff_points_array)
            n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
            print(f"  Spatial clusters of differences: {n_clusters}")
    
    # Create enhanced differential flags
    differential_flags = np.ones(len(np.asarray(diff_cloud.points)), dtype=np.uint8)
    
    # Add confidence scores to the point cloud if available
    if use_confidence:
        diff_cloud.confidence_scores = confidence_scores[different_points_mask]
    
    print(f"\nFinal Results:")
    print(f"  Total target points: {total_target_points}")
    print(f"  Differential points: {diff_points}")
    print(f"  Percentage different: {(diff_points/total_target_points)*100:.2f}%")
    print(f"  Registration fitness: {final_result.fitness:.4f}")
    print(f"  Registration RMSE: {final_result.inlier_rmse:.4f}")
    
    return diff_cloud, differential_flags, {
        'registration_fitness': final_result.fitness,
        'registration_rmse': final_result.inlier_rmse,
        'total_points': total_target_points,
        'differential_points': diff_points,
        'percentage_different': (diff_points/total_target_points)*100,
        'multi_scale_results': multi_scale_results if use_multi_scale else None
    }

def save_differential_ply(point_cloud, output_path, differential_flags=None):
    """
    Save point cloud with differential flag property to PLY file.
    
    Args:
        point_cloud: Point cloud with differential_flags attribute
        output_path: Path to save the PLY file
        differential_flags: Array of differential flags (0=base, 1=new, 2=different)
    """
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    normals = np.asarray(point_cloud.normals) if point_cloud.has_normals() else np.zeros_like(points)
    
    # Get differential flags
    if differential_flags is not None:
        # Use provided differential flags
        pass
    elif hasattr(point_cloud, 'differential_flags'):
        differential_flags = point_cloud.differential_flags
    else:
        # If no flags, assume all are base points
        differential_flags = np.zeros(len(points), dtype=np.uint8)
    
    # Convert colors to 0-255 range
    colors_uint8 = (colors * 255).astype(np.uint8)
    
    # Create vertex data
    vertex_data = []
    for i in range(len(points)):
        vertex_data.append((
            points[i, 0],  # x
            points[i, 1],  # y
            points[i, 2],  # z
            colors_uint8[i, 0],  # red
            colors_uint8[i, 1],  # green
            colors_uint8[i, 2],  # blue
            normals[i, 0],  # nx
            normals[i, 1],  # ny
            normals[i, 2],  # nz
            0,  # label (placeholder)
            0,  # material (placeholder)
            differential_flags[i]  # differential_flag (0=base, 1=new, 2=different)
        ))
    
    # Define PLY vertex element
    vertex_element = PlyElement.describe(
        np.array(vertex_data, dtype=[
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            ('red', 'u1'),
            ('green', 'u1'),
            ('blue', 'u1'),
            ('nx', 'f4'),
            ('ny', 'f4'),
            ('nz', 'f4'),
            ('label', 'u4'),
            ('material', 'u4'),
            ('differential_flag', 'u4')
        ]), 'vertex'
    )
    
    # Create PLY data
    ply_data = PlyData([vertex_element])
    
    # Write PLY file
    ply_data.write(output_path)
    
    # Print summary of what was saved
    diff_count = len(differential_flags)
    print(f"Differential point cloud saved to: {output_path}")
    print(f"  Differential points: {diff_count}")
    print(f"  Total points: {len(points)}")

def save_enhanced_differential_ply(point_cloud, output_path, differential_flags=None, confidence_scores=None):
    """
    Save enhanced point cloud with differential flags and confidence scores to PLY file.
    
    Args:
        point_cloud: Point cloud with differential_flags attribute
        output_path: Path to save the PLY file
        differential_flags: Array of differential flags (0=base, 1=new, 2=different)
        confidence_scores: Array of confidence scores (optional)
    """
    points = np.asarray(point_cloud.points)
    colors = np.asarray(point_cloud.colors)
    normals = np.asarray(point_cloud.normals) if point_cloud.has_normals() else np.zeros_like(points)
    
    # Get differential flags
    if differential_flags is not None:
        # Use provided differential flags
        pass
    elif hasattr(point_cloud, 'differential_flags'):
        differential_flags = point_cloud.differential_flags
    else:
        # If no flags, assume all are base points
        differential_flags = np.zeros(len(points), dtype=np.uint8)
    
    # Get confidence scores
    if confidence_scores is not None:
        # Use provided confidence scores
        pass
    elif hasattr(point_cloud, 'confidence_scores'):
        confidence_scores = point_cloud.confidence_scores
    else:
        # If no confidence scores, use default
        confidence_scores = np.ones(len(points), dtype=np.float32)
    
    # Convert colors to 0-255 range
    colors_uint8 = (colors * 255).astype(np.uint8)
    
    # Create vertex data with enhanced properties
    vertex_data = []
    for i in range(len(points)):
        vertex_data.append((
            points[i, 0],  # x
            points[i, 1],  # y
            points[i, 2],  # z
            colors_uint8[i, 0],  # red
            colors_uint8[i, 1],  # green
            colors_uint8[i, 2],  # blue
            normals[i, 0],  # nx
            normals[i, 1],  # ny
            normals[i, 2],  # nz
            0,  # label (placeholder)
            0,  # material (placeholder)
            differential_flags[i],  # differential_flag (0=base, 1=new, 2=different)
            confidence_scores[i]  # confidence_score
        ))
    
    # Define enhanced PLY vertex element
    vertex_element = PlyElement.describe(
        np.array(vertex_data, dtype=[
            ('x', 'f4'),
            ('y', 'f4'),
            ('z', 'f4'),
            ('red', 'u1'),
            ('green', 'u1'),
            ('blue', 'u1'),
            ('nx', 'f4'),
            ('ny', 'f4'),
            ('nz', 'f4'),
            ('label', 'u4'),
            ('material', 'u4'),
            ('differential_flag', 'u4'),
            ('confidence_score', 'f4')
        ]), 'vertex'
    )
    
    # Create PLY data
    ply_data = PlyData([vertex_element])
    
    # Write PLY file
    ply_data.write(output_path)
    
    # Print summary of what was saved
    diff_count = len(differential_flags)
    print(f"Enhanced differential point cloud saved to: {output_path}")
    print(f"  Differential points: {diff_count}")
    print(f"  Total points: {len(points)}")
    print(f"  Confidence scores included: {confidence_scores is not None}")

def load_enhanced_differential_ply(input_path):
    """
    Load enhanced point cloud with differential flags and confidence scores from PLY file.
    
    Args:
        input_path: Path to PLY file with differential flags and confidence scores
        
    Returns:
        Tuple of (point_cloud, differential_flags, confidence_scores)
    """
    # Read PLY file
    ply_data = PlyData.read(input_path)
    vertex_data = ply_data['vertex']
    
    # Extract point cloud data
    points = np.column_stack([
        vertex_data['x'],
        vertex_data['y'],
        vertex_data['z']
    ])
    
    colors = np.column_stack([
        vertex_data['red'],
        vertex_data['green'],
        vertex_data['blue']
    ]) / 255.0  # Convert to 0-1 range
    
    normals = np.column_stack([
        vertex_data['nx'],
        vertex_data['ny'],
        vertex_data['nz']
    ])
    
    # Extract differential flags and confidence scores
    differential_flags = vertex_data['differential_flag']
    confidence_scores = vertex_data['confidence_score']
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.differential_flags = differential_flags
    pcd.confidence_scores = confidence_scores
    
    # Print summary of loaded point cloud
    diff_count = len(differential_flags)
    print(f"Loaded enhanced differential point cloud: {len(points)} total points")
    print(f"  Differential points: {diff_count}")
    print(f"  Confidence scores: Min={confidence_scores.min():.3f}, Max={confidence_scores.max():.3f}, Mean={confidence_scores.mean():.3f}")
    
    return pcd, differential_flags, confidence_scores

def load_differential_ply(input_path):
    """
    Load point cloud with differential flags from PLY file.
    
    Args:
        input_path: Path to PLY file with differential flags
        
    Returns:
        Tuple of (point_cloud, differential_flags)
    """
    # Read PLY file
    ply_data = PlyData.read(input_path)
    vertex_data = ply_data['vertex']
    
    # Extract point cloud data
    points = np.column_stack([
        vertex_data['x'],
        vertex_data['y'],
        vertex_data['z']
    ])
    
    colors = np.column_stack([
        vertex_data['red'],
        vertex_data['green'],
        vertex_data['blue']
    ]) / 255.0  # Convert to 0-1 range
    
    normals = np.column_stack([
        vertex_data['nx'],
        vertex_data['ny'],
        vertex_data['nz']
    ])
    
    # Extract differential flags
    differential_flags = vertex_data['differential_flag']
    
    # Create point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    pcd.differential_flags = differential_flags
    
    # Print summary of loaded point cloud
    diff_count = len(differential_flags)
    print(f"Loaded differential point cloud: {len(points)} total points")
    print(f"  Differential points: {diff_count}")
    
    return pcd, differential_flags

def create_differential_pointcloud_with_custom_ply(source_cloud, target_cloud, output_path, threshold=0.04, voxel_size=0.02):
    """
    Create differential point cloud containing only points that differ from the base cloud, saved with custom PLY format.
    
    Args:
        source_cloud: Path to source point cloud or o3d.geometry.PointCloud (base cloud)
        target_cloud: Path to target point cloud or o3d.geometry.PointCloud (new cloud)
        output_path: Path to save the differential PLY file
        threshold: Distance threshold for differences (default: 0.04 meters)
        voxel_size: Size of voxels for downsampling (default: 0.02 meters)
    """
    # Create differential point cloud
    diff_cloud, differential_flags = create_differential_pointcloud(source_cloud, target_cloud, threshold, voxel_size)
    
    # Save with custom PLY format
    save_differential_ply(diff_cloud, output_path, differential_flags)
    
    return diff_cloud, differential_flags

def create_enhanced_differential_pointcloud_with_custom_ply(source_cloud, target_cloud, output_path, threshold=0.04, voxel_size=0.02,
                                                          use_multi_scale=True, use_confidence=True, use_color_coding=True):
    """
    Create enhanced differential point cloud with advanced analysis, saved with custom PLY format.
    
    Args:
        source_cloud: Path to source point cloud or o3d.geometry.PointCloud (base cloud)
        target_cloud: Path to target point cloud or o3d.geometry.PointCloud (new cloud)
        output_path: Path to save the differential PLY file
        threshold: Distance threshold for differences (default: 0.04 meters)
        voxel_size: Size of voxels for downsampling (default: 0.02 meters)
        use_multi_scale: Whether to use multi-scale analysis
        use_confidence: Whether to compute confidence scores
        use_color_coding: Whether to use color-coded differences
    """
    # Create enhanced differential point cloud
    diff_cloud, differential_flags, stats = enhanced_point_cloud_comparison(
        source_cloud, target_cloud, threshold, voxel_size, 
        use_multi_scale, use_confidence, use_color_coding
    )
    
    # Get confidence scores if available
    confidence_scores = None
    if hasattr(diff_cloud, 'confidence_scores'):
        confidence_scores = diff_cloud.confidence_scores
    
    # Save with enhanced PLY format
    save_enhanced_differential_ply(diff_cloud, output_path, differential_flags, confidence_scores)
    
    return diff_cloud, differential_flags, stats

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Create differential point cloud comparison')
    parser.add_argument('source_path', help='Path to source PLY file')
    parser.add_argument('target_path', help='Path to target PLY file')
    parser.add_argument('--output_path', help='Path to save differential PLY file')
    parser.add_argument('--threshold', type=float, default=0.04,
                        help='Distance threshold for differences (default: 0.04 meters)')
    parser.add_argument('--voxel_size', type=float, default=0.02,
                        help='Voxel size for downsampling (default: 0.02 meters)')
    
    args = parser.parse_args()
    
    # Determine output path
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = args.target_path.rsplit('.', 1)[0] + '_differential.ply'
    
    # Create differential point cloud with custom PLY format
    diff_cloud, differential_flags = create_differential_pointcloud_with_custom_ply(
        args.source_path,
        args.target_path,
        output_path,
        threshold=args.threshold,
        voxel_size=args.voxel_size
    )
