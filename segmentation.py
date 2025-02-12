import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import argparse
import sys


class PointCloudSegmentation:
    """Handles point cloud processing and plane segmentation."""

    def __init__(self, max_planes=12, voxel_size=0.03, distance_threshold=0.03):
        """
        Initialize the segmentation class.

        :param max_planes: Maximum number of planes to extract
        :param voxel_size: Size of voxels for downsampling
        :param distance_threshold: Distance threshold for plane segmentation
        """
        self.max_planes = max_planes
        self.voxel_size = voxel_size
        self.distance_threshold = distance_threshold
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
        
        if self.voxel_size > 0:
            print(f"Downsampling point cloud with voxel size: {self.voxel_size}")
            self.pcd = self.pcd.voxel_down_sample(self.voxel_size)
            print("Downsampled point cloud:", len(np.asarray(self.pcd.points)), "points")

        # Estimate normals
        print("Estimating normals...")
        nn_distance = np.mean(self.pcd.compute_nearest_neighbor_distance())
        radius_normals = nn_distance * 4
        self.pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=radius_normals, 
                max_nn=16
            ),
            fast_normal_computation=True
        )

    def segment_planes(self):
        """Segment planes from the point cloud."""
        print(f"Segmenting up to {self.max_planes} planes...")
        rest = self.pcd
        all_point_clouds = []

        for i in range(self.max_planes):
            if len(np.asarray(rest.points)) < 1000:
                break

            print(f"\nProcessing plane {i+1}/{self.max_planes}...")
            print(f"Points remaining: {len(np.asarray(rest.points))}")
            colors = plt.get_cmap("tab20")(i)

            # Segment plane
            plane_model, inliers = rest.segment_plane(
                distance_threshold=self.distance_threshold,
                ransac_n=3,
                num_iterations=20000
            )

            # Store plane information
            self.segment_models[i] = plane_model
            self.segments[i] = rest.select_by_index(inliers)
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
        self.result = all_point_clouds[0]
        for pc in all_point_clouds[1:]:
            self.result += pc

        # Translate back to original position
        self.result.translate(self.center)
        return self.result

    def save_result(self, output_path):
        """Save the segmented point cloud."""
        print(f"Saving result to {output_path}...")
        try:
            o3d.io.write_point_cloud(output_path, self.result)
            print("Processing complete!")
        except Exception as e:
            print(f"Error saving point cloud: {e}")
            sys.exit(1)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Segment planes in a point cloud.')
    parser.add_argument('input_path', help='Path to input PLY file')
    parser.add_argument('--output_path', help='Path to output PLY file', default=None)
    parser.add_argument('--max_planes', type=int, default=12, help='Maximum number of planes to segment')
    parser.add_argument('--voxel_size', type=float, default=0.03, help='Voxel size for downsampling')
    
    args = parser.parse_args()
    
    # If output path not specified, create one based on input path
    if args.output_path is None:
        args.output_path = args.input_path.rsplit('.', 1)[0] + '_segmented.ply'
    
    # Process the point cloud
    segmenter = PointCloudSegmentation(
        max_planes=args.max_planes,
        voxel_size=args.voxel_size
    )
    segmenter.load_and_preprocess(args.input_path)
    segmenter.segment_planes()
    segmenter.save_result(args.output_path)


if __name__ == "__main__":
    main()