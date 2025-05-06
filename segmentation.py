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
                 normal_threshold=0.60, ransac_n=3, iterations=20000):
        """
        Initialize the segmentation class.

        :param max_planes: Maximum number of planes to extract
        :param voxel_size: Size of voxels for downsampling
        :param distance_threshold: Distance threshold for plane segmentation
        :param min_points: Minimum number of points for a valid plane
        :param normal_threshold: Threshold for normal consistency in post-processing
        :param ransac_n: Number of points to sample for RANSAC
        :param iterations: Maximum iterations for RANSAC
        """
        self.max_planes = max_planes
        self.voxel_size = voxel_size
        self.distance_threshold = distance_threshold
        self.min_points = min_points
        self.normal_threshold = normal_threshold
        self.ransac_n = ransac_n
        self.iterations = iterations
        self.segments = {}
        self.segment_models = {}

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

    def segment_planes(self):
        """Segment planes from the point cloud."""
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

    def refine_segments(self):
        """Refine segments by merging similar planes."""
        print("Refining segments...")
        
        # Skip if no segments were found
        if not self.segments:
            return
        
        # Check pairs of segments for potential merging
        merged = True
        while merged:
            merged = False
            keys = list(self.segment_models.keys())
            
            for i in range(len(keys)):
                if i not in self.segment_models:
                    continue
                
                for j in range(i+1, len(keys)):
                    if j not in self.segment_models:
                        continue
                    
                    # Get plane models
                    model_i = self.segment_models[i]
                    model_j = self.segment_models[j]
                    
                    # Check if normals are similar
                    normal_i = model_i[:3]
                    normal_j = model_j[:3]
                    normal_dot = np.abs(np.dot(normal_i, normal_j))
                    
                    # Check if planes are close in distance (parallel planes)
                    dist_i = model_i[3]
                    dist_j = model_j[3]
                    dist_diff = abs(dist_i - dist_j)
                    
                    # If planes are similar, merge them
                    if normal_dot > 0.95 and dist_diff < self.distance_threshold * 3:
                        print(f"Merging planes {i} and {j} (normal similarity: {normal_dot:.4f}, distance: {dist_diff:.4f})")
                        
                        # Merge point clouds
                        self.segments[i] += self.segments[j]
                        
                        # Recalculate plane model with all points
                        points = np.asarray(self.segments[i].points)
                        if len(points) > 3:
                            plane_model, _ = self.segments[i].segment_plane(
                                distance_threshold=self.distance_threshold,
                                ransac_n=3,
                                num_iterations=1000
                            )
                            self.segment_models[i] = plane_model
                        
                        # Remove the merged segment
                        del self.segments[j]
                        del self.segment_models[j]
                        merged = True
                        break
                    
                if merged:
                    break

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
    segmenter.refine_segments()  # Always perform refinement
    segmenter.save_result(output_path)
    
    print(f"Segmented point cloud saved to: {output_path}")


if __name__ == "__main__":
    main()