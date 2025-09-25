import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import argparse
import sys
import yaml
import os


class PointCloudSegmentation:
    """Handles point cloud processing and plane segmentation."""

    def __init__(self, max_planes=18, voxel_size=0.03, distance_threshold=0.03, min_points=500, 
                 normal_threshold=0.20, ransac_n=3, iterations=20000, enhanced_mode=False):
        """
        Initialize the segmentation class.

        :param max_planes: Maximum number of planes to extract
        :param voxel_size: Size of voxels for downsampling
        :param distance_threshold: Distance threshold for plane segmentation
        :param min_points: Minimum number of points for a valid plane
        :param normal_threshold: Threshold for normal consistency in post-processing
        :param ransac_n: Number of points to sample for RANSAC
        :param iterations: Maximum iterations for RANSAC
        :param enhanced_mode: Use enhanced RANSAC+DBSCAN method
        """
        self.max_planes = max_planes
        self.voxel_size = voxel_size
        self.distance_threshold = distance_threshold
        self.min_points = min_points
        self.normal_threshold = normal_threshold
        self.ransac_n = ransac_n
        self.iterations = iterations
        self.enhanced_mode = enhanced_mode
        self.segments = {}
        self.segment_models = {}
        self.segment_colors = {}  # Store colors for diffraction compatibility

    def load_and_preprocess(self, input_path):
        """Load and preprocess the point cloud."""
        try:
            self.pcd = o3d.io.read_point_cloud(input_path)
            print("Original point cloud:", len(np.asarray(self.pcd.points)), "points")
        except Exception as e:
            print(f"Error loading point cloud: {e}")
            sys.exit(1)

        # Center and downsample
        self.center = self.pcd.get_center()
        self.pcd.translate(-self.center)
        
        # Remove outliers
        print("Removing outliers...")
        # Fix: Use the correct method directly on the point cloud object
        cl, ind = self.pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        self.pcd = self.pcd.select_by_index(ind)
        print("After outlier removal:", len(np.asarray(self.pcd.points)), "points")
        
        if self.voxel_size > 0:
            print(f"Downsampling point cloud with voxel size: {self.voxel_size}")
            self.pcd = self.pcd.voxel_down_sample(self.voxel_size)
            print("Downsampled point cloud:", len(np.asarray(self.pcd.points)), "points")

        # Estimate normals with adaptive radius
        print("Estimating normals...")
        nn_distance = np.mean(self.pcd.compute_nearest_neighbor_distance())
        radius_normals = nn_distance * 4
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normals, 
                max_nn=30
            ),
            fast_normal_computation=False
        )
        
        # Orient normals consistently
        self.pcd.orient_normals_consistent_tangent_plane(k=20)

    def segment_single_plane(self, visualize=False):
        """
        Segment a single plane from the point cloud using RANSAC.
        
        :param visualize: Whether to visualize the results
        :return: (inlier_cloud, outlier_cloud, plane_model)
        """
        print("Performing single plane segmentation...")
        
        plane_model, inliers = self.pcd.segment_plane(
            distance_threshold=self.distance_threshold,
            ransac_n=self.ransac_n,
            num_iterations=self.iterations
        )
        
        inlier_cloud = self.pcd.select_by_index(inliers)
        outlier_cloud = self.pcd.select_by_index(inliers, invert=True)
        
        # Paint inliers red and outliers grey
        inlier_cloud.paint_uniform_color([1, 0, 0])
        outlier_cloud.paint_uniform_color([0.6, 0.6, 0.6])
        
        print(f"Plane found with {len(inliers)} inlier points")
        print(f"Plane equation: {plane_model[0]:.3f}x + {plane_model[1]:.3f}y + {plane_model[2]:.3f}z + {plane_model[3]:.3f} = 0")
        
        if visualize:
            o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
        
        return inlier_cloud, outlier_cloud, plane_model

    def cluster_dbscan(self, eps=0.05, min_points=10, visualize=False):
        """
        Perform DBSCAN clustering on the point cloud.
        
        :param eps: Maximum distance between points in a cluster
        :param min_points: Minimum number of points required to form a cluster
        :param visualize: Whether to visualize the results
        :return: (clustered_cloud, labels)
        """
        print(f"Performing DBSCAN clustering (eps={eps}, min_points={min_points})...")
        
        labels = np.array(self.pcd.cluster_dbscan(eps=eps, min_points=min_points))
        max_label = labels.max()
        
        print(f"Point cloud has {max_label + 1} clusters")
        
        # Color the clusters
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0  # Noise points are black
        
        clustered_cloud = o3d.geometry.PointCloud(self.pcd)
        clustered_cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
        
        if visualize:
            o3d.visualization.draw_geometries([clustered_cloud])
        
        return clustered_cloud, labels

    def segment_planes_enhanced(self):
        """
        Enhanced plane segmentation using RANSAC + DBSCAN refinement.
        Compatible with diffraction edge detection - keeps segments separate.
        
        :return: Result point cloud with segmented planes
        """
        print(f"Enhanced plane segmentation with DBSCAN refinement...")
        print(f"Segmenting up to {self.max_planes} planes...")
        
        rest = o3d.geometry.PointCloud(self.pcd)
        segment_counter = 0  # Track actual segments for diffraction compatibility
        
        for i in range(self.max_planes):
            if len(np.asarray(rest.points)) < self.min_points:
                print(f"Stopping: Remaining points ({len(np.asarray(rest.points))}) below threshold")
                break

            print(f"\nProcessing plane {i+1}/{self.max_planes}...")
            print(f"Points remaining: {len(np.asarray(rest.points))}")
            
            # Get color for this segment (ensure diffraction compatibility)
            colors = plt.get_cmap("tab20")(segment_counter % 20)  # Cycle through colors if needed

            # RANSAC plane segmentation
            plane_model, inliers = rest.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=self.ransac_n,
                num_iterations=self.iterations
            )
            
            if len(inliers) < self.min_points:
                print(f"Skipping plane with only {len(inliers)} points (below threshold)")
                continue

            # Extract initial plane segment
            segment = rest.select_by_index(inliers)
            
            # Apply DBSCAN clustering to refine the segment
            print(f"Applying DBSCAN refinement to plane segment...")
            eps_dbscan = self.distance_threshold * 10  # Empirical choice
            labels = np.array(segment.cluster_dbscan(eps=eps_dbscan, min_points=10))
            
            if len(np.unique(labels)) > 1:  # If clustering found multiple groups
                # Count points in each cluster (excluding noise points)
                unique_labels = np.unique(labels)
                valid_labels = unique_labels[unique_labels >= 0]  # Exclude noise (-1)
                
                if len(valid_labels) == 0:
                    print("No valid clusters found after DBSCAN, using original segment")
                    rest = rest.select_by_index(inliers, invert=True)
                else:
                    candidates = [len(np.where(labels == j)[0]) for j in valid_labels]
                    
                    # Find the largest cluster (best candidate)
                    best_candidate_idx = np.argmax(candidates)
                    best_candidate = valid_labels[best_candidate_idx]
                    
                    print(f"DBSCAN found {len(valid_labels)} valid clusters, selecting largest with {candidates[best_candidate_idx]} points")
                    
                    # Keep only the largest cluster as the refined segment
                    best_cluster_indices = np.where(labels == best_candidate)[0]
                    refined_segment = segment.select_by_index(best_cluster_indices)
                    
                    # Add non-selected points back to rest
                    other_indices = np.where(labels != best_candidate)[0]
                    if len(other_indices) > 0:
                        other_points = segment.select_by_index(other_indices)
                        rest = rest.select_by_index(inliers, invert=True) + other_points
                    else:
                        rest = rest.select_by_index(inliers, invert=True)
                    
                    segment = refined_segment
            else:
                # No refinement needed, use original segment
                rest = rest.select_by_index(inliers, invert=True)
            
            # Verify final segment quality
            if len(np.asarray(segment.points)) < self.min_points:
                print(f"Refined segment too small ({len(np.asarray(segment.points))} points), skipping")
                continue
            
            # Store the refined segment with consistent coloring for diffraction compatibility
            self.segment_models[segment_counter] = plane_model
            self.segments[segment_counter] = segment
            self.segment_colors[segment_counter] = list(colors[:3])
            
            # Paint segment with uniform color (required for diffraction edge detection)
            segment.paint_uniform_color(list(colors[:3]))
            
            segment_counter += 1

        # Final clustering of remaining points - create additional segments for diffraction
        if len(np.asarray(rest.points)) > 0:
            print(f"\nClustering remaining {len(np.asarray(rest.points))} points...")
            labels = np.array(rest.cluster_dbscan(eps=0.05, min_points=5))
            max_label = labels.max()
            
            if max_label >= 0:
                print(f"Remaining points formed {max_label + 1} clusters")
                
                # Process each cluster as a separate segment
                for cluster_id in range(max_label + 1):
                    cluster_indices = np.where(labels == cluster_id)[0]
                    if len(cluster_indices) >= 5:  # Only keep substantial clusters
                        cluster_segment = rest.select_by_index(cluster_indices)
                        
                        # Assign unique color to each remaining cluster
                        cluster_color = plt.get_cmap("tab10")(cluster_id % 10)
                        cluster_segment.paint_uniform_color(list(cluster_color[:3]))
                        
                        # Store as additional segments for diffraction detection
                        self.segments[segment_counter] = cluster_segment
                        self.segment_colors[segment_counter] = list(cluster_color[:3])
                        segment_counter += 1
                
                # Handle noise points separately if needed
                noise_indices = np.where(labels == -1)[0]
                if len(noise_indices) > 0:
                    noise_segment = rest.select_by_index(noise_indices)
                    noise_segment.paint_uniform_color([0.7, 0.7, 0.7])  # Grey for noise
                    self.segments[segment_counter] = noise_segment
                    self.segment_colors[segment_counter] = [0.7, 0.7, 0.7]
            else:
                print("No clusters found in remaining points")
                if len(np.asarray(rest.points)) > 0:
                    rest.paint_uniform_color([0.7, 0.7, 0.7])
                    self.segments[segment_counter] = rest
                    self.segment_colors[segment_counter] = [0.7, 0.7, 0.7]

        print(f"\nSegmentation complete: {len(self.segments)} segments found")
        
        # Create result by combining segments ONLY for compatibility with existing pipeline
        if self.segments:
            all_segments = list(self.segments.values())
            self.result = all_segments[0]
            for segment in all_segments[1:]:
                self.result += segment
                
            # Translate back to original position
            self.result.translate(self.center)
            
            # Verify diffraction compatibility
            self.verify_diffraction_compatibility()
            
            return self.result
        else:
            print("No segments found!")
            return None

    def get_individual_segments(self):
        """
        Get individual segments as separate point clouds.
        Useful for diffraction edge detection.
        
        :return: Dictionary of segment point clouds
        """
        if not self.segments:
            print("No segments available. Run segmentation first.")
            return {}
            
        # Translate segments back to original position
        translated_segments = {}
        for idx, segment in self.segments.items():
            translated_segment = o3d.geometry.PointCloud(segment)
            translated_segment.translate(self.center)
            translated_segments[idx] = translated_segment
            
        return translated_segments

    def save_individual_segments(self, output_dir):
        """
        Save each segment as a separate PLY file.
        
        :param output_dir: Directory to save segment files
        """
        if not self.segments:
            print("No segments to save. Run segmentation first.")
            return
            
        os.makedirs(output_dir, exist_ok=True)
        segments = self.get_individual_segments()
        
        for idx, segment in segments.items():
            filename = os.path.join(output_dir, f"segment_{idx:03d}.ply")
            o3d.io.write_point_cloud(filename, segment)
            print(f"Saved segment {idx} to {filename}")

    def visualize_segments(self):
        """Visualize all segments together."""
        if not self.segments:
            print("No segments to visualize. Run segmentation first.")
            return
            
        segments = self.get_individual_segments()
        o3d.visualization.draw_geometries(list(segments.values()))

    def verify_diffraction_compatibility(self):
        """Verify that the segmented point cloud is compatible with diffraction edge detection."""
        if self.result is None:
            return False
            
        colors = np.asarray(self.result.colors)
        unique_colors = np.unique(colors, axis=0)
        
        print(f"Segmentation result: {len(np.asarray(self.result.points))} points, {len(unique_colors)} unique segments")
        print("Diffraction compatibility: ✓" if len(unique_colors) >= 2 else "✗ (Need at least 2 segments)")
        
        # Check if segments have sufficient points for plane fitting
        segment_sizes = []
        for color in unique_colors:
            color_mask = np.all(np.isclose(colors, color, atol=1e-3), axis=1)
            segment_size = np.sum(color_mask)
            segment_sizes.append(segment_size)
        
        print(f"Segment sizes: {segment_sizes}")
        suitable_segments = sum(1 for size in segment_sizes if size >= 20)  # DiffractionEdgeDetector.min_points
        print(f"Segments suitable for diffraction detection: {suitable_segments}")
        
        return len(unique_colors) >= 2 and suitable_segments >= 2

    def segment_planes(self):
        """
        Segment planes from the point cloud using the selected method.
        """
        if self.enhanced_mode:
            return self.segment_planes_enhanced()
        else:
            return self.segment_planes_original()
    
    def segment_planes_original(self):
        """Original plane segmentation method."""
        print(f"Segmenting up to {self.max_planes} planes...")
        rest = self.pcd
        all_point_clouds = []

        for i in range(self.max_planes):
            if len(np.asarray(rest.points)) < self.min_points:
                print(f"Stopping: Remaining points ({len(np.asarray(rest.points))}) below threshold")
                break

            print(f"\nProcessing plane {i+1}/{self.max_planes}...")
            print(f"Points remaining: {len(np.asarray(rest.points))}")
            colors = plt.get_cmap("tab20")(i)

            # Segment plane with improved parameters
            plane_model, inliers = rest.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=self.ransac_n,
                num_iterations=self.iterations
            )
            
            # Skip if too few points
            if len(inliers) < self.min_points:
                print(f"Skipping plane with only {len(inliers)} points (below threshold)")
                continue

            # Extract the plane segment
            segment = rest.select_by_index(inliers)
            
            # Verify plane quality through normal consistency
            normals = np.asarray(segment.normals)
            plane_normal = plane_model[:3]
            normal_consistency = np.abs(np.dot(normals, plane_normal))
            avg_consistency = np.mean(normal_consistency)
            
            print(f"Plane normal consistency: {avg_consistency:.4f}")
            if avg_consistency < self.normal_threshold:
                print(f"Skipping plane with poor normal consistency")
                continue

            # Store plane information
            self.segment_models[i] = plane_model
            self.segments[i] = segment
            self.segments[i].paint_uniform_color(list(colors[:3]))
            all_point_clouds.append(self.segments[i])

            # Update remaining points
            rest = rest.select_by_index(inliers, invert=True)

        # Add remaining points
        if len(np.asarray(rest.points)) > 0:
            print(f"\nRemaining points: {len(np.asarray(rest.points))}")
            rest.paint_uniform_color([0.7, 0.7, 0.7])
            all_point_clouds.append(rest)

        # Combine segments
        print("\nCombining segments...")
        if all_point_clouds:
            self.result = all_point_clouds[0]
            for pc in all_point_clouds[1:]:
                self.result += pc

            # Translate back to original position
            self.result.translate(self.center)
            return self.result
        else:
            print("No segments found!")
            return None

    def save_result(self, output_path):
        """Save the segmented point cloud."""
        print(f"Saving result to {output_path}...")
        try:
            o3d.io.write_point_cloud(output_path, self.result)
            print("Processing complete!")
        except Exception as e:
            print(f"Error saving point cloud: {e}")
            sys.exit(1)


def validate_point_cloud_file(file_path):
    """
    Validates that the provided file exists and can be opened as a point cloud.
    
    Args:
        file_path: Path to the point cloud file
        
    Returns:
        bool: True if the file is valid, False otherwise
    """
    # Check if file exists
    if not os.path.isfile(file_path):
        print(f"Error: File not found: {file_path}")
        return False
    
    # Check file extension
    valid_extensions = ['.ply', '.pcd', '.xyz', '.xyzn', '.xyzrgb', '.pts']
    _, ext = os.path.splitext(file_path)
    if ext.lower() not in valid_extensions:
        print(f"Warning: File extension {ext} might not be supported. Valid extensions: {', '.join(valid_extensions)}")
    
    # Try to open the file with Open3D
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        if len(np.asarray(pcd.points)) == 0:
            print(f"Error: Point cloud file is empty: {file_path}")
            return False
        return True
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        return False


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Segment planes in a point cloud.')
    parser.add_argument('input_path', help='Path to input point cloud file')
    
    args = parser.parse_args()
    
    # Validate input file
    if not validate_point_cloud_file(args.input_path):
        sys.exit(1)
    
    # Create output path based on input path
    output_path = args.input_path.rsplit('.', 1)[0] + '_segmented.ply'
    
    # Process the point cloud with default parameters
    segmenter = PointCloudSegmentation()
    
    segmenter.load_and_preprocess(args.input_path)
    segmenter.segment_planes()
    segmenter.save_result(output_path)
    
    print(f"Segmented point cloud saved to: {output_path}")


if __name__ == "__main__":
    main()