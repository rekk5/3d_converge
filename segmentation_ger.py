import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import argparse
import sys
import os


class PointCloudSegmentation:
    """Handles point cloud preprocessing and segmentation using RANSAC and DBSCAN."""

    def __init__(self,
                 # Preprocessing
                 voxel_size=0.03,
                 # RANSAC Plane Segmentation
                 ransac_distance_threshold=0.02, # Increased default slightly
                 ransac_n=3,
                 ransac_num_iterations=1000,
                 max_planes=50, # Max planes to find iteratively
                 # DBSCAN (for refinement and final clustering)
                 dbscan_eps=0.05, # Can be used for final clustering
                 dbscan_min_points=10, # Can be used for final clustering
                 refine_clusters=True, # Flag to enable DBSCAN refinement of planes
                 refine_dbscan_eps_factor=5, # Factor to multiply ransac_distance_threshold for refine eps
                 refine_dbscan_min_points=10):
        """
        Initialize the segmentation class.

        :param voxel_size: Size of voxels for downsampling. Set to 0 to disable.
        :param ransac_distance_threshold: RANSAC distance threshold for plane fitting.
        :param ransac_n: Number of points to sample for RANSAC.
        :param ransac_num_iterations: Number of RANSAC iterations.
        :param max_planes: Maximum number of planes to iteratively segment.
        :param dbscan_eps: DBSCAN epsilon distance (primarily for final clustering of remaining points).
        :param dbscan_min_points: DBSCAN minimum points (primarily for final clustering).
        :param refine_clusters: If True, use DBSCAN to refine found planes, keeping only the largest cluster.
        :param refine_dbscan_eps_factor: Multiplies ransac_distance_threshold to get eps for plane refinement DBSCAN.
        :param refine_dbscan_min_points: Min points for plane refinement DBSCAN.
        """
        # Preprocessing
        self.voxel_size = voxel_size

        # RANSAC parameters
        self.ransac_distance_threshold = ransac_distance_threshold
        self.ransac_n = ransac_n
        self.ransac_num_iterations = ransac_num_iterations
        self.max_planes = max_planes

        # DBSCAN parameters
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_points = dbscan_min_points

        # Refinement parameters
        self.refine_clusters = refine_clusters
        self.refine_dbscan_eps = ransac_distance_threshold * refine_dbscan_eps_factor # Calculate refine eps based on threshold
        self.refine_dbscan_min_points = refine_dbscan_min_points


        # Internal state
        self.pcd = None
        self.result = None
        self.center = None # Store center for potential translation back

    def preprocess(self, pcd_input):
        """Preprocess the input point cloud."""
        if not isinstance(pcd_input, o3d.geometry.PointCloud):
            raise TypeError("Input must be an open3d.geometry.PointCloud object.")

        self.pcd = o3d.geometry.PointCloud(pcd_input) # Work on a copy
        print("Original point cloud:", len(np.asarray(self.pcd.points)), "points")

        # Center the point cloud (optional, but often helpful)
        self.center = self.pcd.get_center()
        self.pcd.translate(-self.center)

        # Remove outliers
        print("Removing statistical outliers...")
        cl, ind = self.pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        self.pcd = self.pcd.select_by_index(ind)
        print("After outlier removal:", len(np.asarray(self.pcd.points)), "points")

        if self.voxel_size > 0:
            print(f"Downsampling point cloud with voxel size: {self.voxel_size}")
            self.pcd = self.pcd.voxel_down_sample(self.voxel_size)
            print("Downsampled point cloud:", len(np.asarray(self.pcd.points)), "points")

        # Normal estimation is not strictly required for RANSAC plane or DBSCAN
        # but can be useful for other tasks or visualization. Removing for simplicity here.
        # If needed, it can be added back.
        print("Skipping normal estimation for RANSAC/DBSCAN pipeline.")


    def segment_dbscan(self):
        """Segment the point cloud using ONLY DBSCAN."""
        if self.pcd is None or len(np.asarray(self.pcd.points)) == 0:
            print("Preprocessing step must be run first, or point cloud is empty.")
            self.result = o3d.geometry.PointCloud() # Return empty point cloud
            return

        print(f"Clustering with DBSCAN (eps={self.dbscan_eps}, min_points={self.dbscan_min_points})...")

        # Perform DBSCAN clustering
        labels = np.array(self.pcd.cluster_dbscan(eps=self.dbscan_eps, min_points=self.dbscan_min_points, print_progress=True))

        max_label = labels.max()
        print(f"Found {max_label + 1} clusters (excluding noise).")

        # Assign colors based on labels
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = [0.5, 0.5, 0.5, 1.0] # Gray for noise
        self.pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

        # Translate back to original position
        if self.center is not None:
            self.pcd.translate(self.center)

        self.result = self.pcd # Store the colored point cloud as the result
        print("DBSCAN-only segmentation complete.")


    # --- NEW METHOD: RANSAC + DBSCAN ---
    def segment_ransac_dbscan(self):
        """Segment using iterative RANSAC for planes, optionally refine planes
           with DBSCAN, and cluster remaining points with DBSCAN."""

        if self.pcd is None or len(np.asarray(self.pcd.points)) == 0:
            print("Preprocessing step must be run first, or point cloud is empty.")
            self.result = o3d.geometry.PointCloud()
            return

        # Use a copy of the preprocessed point cloud
        rest = o3d.geometry.PointCloud(self.pcd)
        num_points_total = len(rest.points)
        all_segments_colored = [] # List to store final colored PointCloud segments
        plane_colors = plt.get_cmap("tab20") # Colormap for planes

        print(f"\n--- Starting RANSAC Plane Segmentation (max_planes={self.max_planes}) ---")
        print(f"Params: distance_threshold={self.ransac_distance_threshold}, n={self.ransac_n}, iter={self.ransac_num_iterations}")
        if self.refine_clusters:
             print(f"Plane Refinement: Enabled (eps={self.refine_dbscan_eps:.3f}, min_points={self.refine_dbscan_min_points})")


        for i in range(self.max_planes):
            if len(rest.points) < self.ransac_n * 3: # Need sufficient points for RANSAC
                print(f"\nStopping RANSAC early: Only {len(rest.points)} points remaining (less than {self.ransac_n * 3}).")
                break

            try:
                plane_model, inliers = rest.segment_plane(
                    distance_threshold=self.ransac_distance_threshold,
                    ransac_n=self.ransac_n,
                    num_iterations=self.ransac_num_iterations)
            except Exception as e:
                 print(f"RANSAC failed on iteration {i+1}: {e}")
                 break # Stop if RANSAC fails

            num_inliers = len(inliers)
            # Heuristic: Stop if the found plane is very small compared to remaining points or absolute minimum
            min_plane_points = max(self.refine_dbscan_min_points * 2, 50) # Adjust minimum plane size if needed
            if num_inliers < min_plane_points:
                print(f"\nStopping RANSAC early: Plane {i+1} too small ({num_inliers} points < {min_plane_points}).")
                break

            print(f"\nPass {i+1}/{self.max_planes}: Found plane with {num_inliers} points.")
            plane_segment = rest.select_by_index(inliers)
            rest = rest.select_by_index(inliers, invert=True) # Update rest immediately

            if self.refine_clusters:
                # Refine the plane using DBSCAN to keep the largest connected component
                labels = np.array(plane_segment.cluster_dbscan(
                    eps=self.refine_dbscan_eps,
                    min_points=self.refine_dbscan_min_points,
                    print_progress=False)) # No progress for refinement

                unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True) # Ignore noise (-1)

                if len(counts) > 0:
                    best_label = unique_labels[np.argmax(counts)]
                    best_indices = np.where(labels == best_label)[0]
                    noise_indices = np.where(labels == -1)[0]
                    other_cluster_indices = np.where((labels != best_label) & (labels != -1))[0]

                    refined_plane = plane_segment.select_by_index(best_indices)
                    num_refined = len(refined_plane.points)

                    # Add points from noise and smaller clusters back to 'rest'
                    noise_points = plane_segment.select_by_index(noise_indices)
                    other_cluster_points = plane_segment.select_by_index(other_cluster_indices)
                    rest += noise_points
                    rest += other_cluster_points

                    print(f"  Refined plane {i+1}: Kept largest cluster ({num_refined} points). "
                          f"Added {len(noise_points.points)} noise points and "
                          f"{len(other_cluster_points.points)} other cluster points back to rest.")
                    plane_to_color = refined_plane
                else:
                    # No clusters found, treat the whole segment as noise/unstructured
                    print(f"  Refinement failed for plane {i+1} (no clusters found). Adding all {num_inliers} points back to rest.")
                    rest += plane_segment # Add the whole thing back
                    plane_to_color = None # Don't add this segment
            else:
                 # No refinement, use the whole RANSAC segment
                 plane_to_color = plane_segment
                 print(f"  Using full RANSAC plane {i+1} ({num_inliers} points).")


            if plane_to_color is not None and len(plane_to_color.points) > 0:
                 # Assign color and add to list
                 color = plane_colors(i / self.max_planes) # Normalize index for colormap
                 plane_to_color.paint_uniform_color(list(color[:3]))
                 all_segments_colored.append(plane_to_color)

            print(f"  Points remaining for next iteration: {len(rest.points)}")


        # --- Cluster the final remaining points ---
        print(f"\n--- Clustering Remaining Points ({len(rest.points)}) ---")
        if len(rest.points) > self.dbscan_min_points:
            print(f"Params: eps={self.dbscan_eps}, min_points={self.dbscan_min_points}")
            labels_rest = np.array(rest.cluster_dbscan(
                eps=self.dbscan_eps,
                min_points=self.dbscan_min_points,
                print_progress=True))
            max_label_rest = labels_rest.max()
            num_clusters_rest = max_label_rest + 1
            num_noise_rest = len(np.where(labels_rest == -1)[0])
            print(f"Found {num_clusters_rest} clusters and {num_noise_rest} noise points in remaining points.")

            # Use a different colormap (e.g., tab10) or just continue tab20
            # Offset cluster numbers to avoid color collision with planes if using same map
            # For simplicity, let's use a different map here.
            colors_rest = plt.get_cmap("tab10")(labels_rest / (max_label_rest if max_label_rest > 0 else 1))
            colors_rest[labels_rest < 0] = [0.5, 0.5, 0.5, 1.0] # Gray for noise
            rest.colors = o3d.utility.Vector3dVector(colors_rest[:, :3])
            all_segments_colored.append(rest) # Add the colored remaining points
        elif len(rest.points) > 0:
             print("Remaining points too few to cluster, coloring gray.")
             rest.paint_uniform_color([0.5, 0.5, 0.5]) # Color leftovers gray
             all_segments_colored.append(rest)
        else:
            print("No points remaining after plane segmentation.")


        # --- Combine all segments into the final result ---
        if all_segments_colored:
            final_pcd = o3d.geometry.PointCloud()
            for seg in all_segments_colored:
                final_pcd += seg
            print(f"\nCombined {len(all_segments_colored)} segments into final point cloud with {len(final_pcd.points)} points.")
            if len(final_pcd.points) != num_points_total:
                 print(f"Warning: Final point count ({len(final_pcd.points)}) differs from initial preprocessed count ({num_points_total}). Check refinement logic.")

        else:
            print("\nNo segments found.")
            final_pcd = o3d.geometry.PointCloud() # Empty result

        # Translate back to original position
        if self.center is not None:
            final_pcd.translate(self.center)
            print("Translated result back to original center.")


        self.result = final_pcd
        print("\n--- RANSAC + DBSCAN Segmentation Complete ---")


    def get_result(self):
        """Return the processed and segmented point cloud."""
        if self.result is None:
             print("Warning: No result generated yet. Run a segmentation method first.")
             return o3d.geometry.PointCloud() # Return empty point cloud
        # Ensure result has points before printing count
        if self.result.has_points():
            print(f"Returning result point cloud with {len(self.result.points)} points.")
        else:
            print("Returning empty result point cloud.")
        return self.result


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Segment point cloud using RANSAC and DBSCAN.')
    parser.add_argument('input_path', help='Path to input point cloud file (e.g., .ply, .pcd)')
    parser.add_argument('--output_path', help='Path to save the segmented point cloud (optional)', default=None)
    parser.add_argument('--visualize', action='store_true', help='Visualize the result')

    # --- Add arguments for the new parameters ---
    # Preprocessing
    parser.add_argument('--voxel_size', type=float, default=0.03, help='Voxel size for downsampling (0 to disable)')
    # RANSAC
    parser.add_argument('--ransac_dist', type=float, default=0.02, help='RANSAC distance threshold')
    parser.add_argument('--ransac_n', type=int, default=3, help='RANSAC samples')
    parser.add_argument('--ransac_iter', type=int, default=1000, help='RANSAC iterations')
    parser.add_argument('--max_planes', type=int, default=20, help='Max planes to find with RANSAC')
    # DBSCAN (final clustering)
    parser.add_argument('--dbscan_eps', type=float, default=0.05, help='DBSCAN epsilon for final clustering')
    parser.add_argument('--dbscan_min_pts', type=int, default=10, help='DBSCAN min points for final clustering')
    # Refinement
    parser.add_argument('--refine', action='store_true', help='Enable DBSCAN refinement of RANSAC planes')
    parser.add_argument('--refine_eps_factor', type=float, default=5.0, help='Factor * ransac_dist for refine DBSCAN eps')
    parser.add_argument('--refine_min_pts', type=int, default=10, help='Min points for refine DBSCAN')
    # --- End new arguments ---

    # --- Add argument to choose method ---
    parser.add_argument('--method', type=str, default='ransac_dbscan', choices=['ransac_dbscan', 'dbscan_only'],
                        help='Segmentation method to use')


    args = parser.parse_args()

    # Validate input file existence (basic check)
    if not os.path.isfile(args.input_path):
        print(f"Error: Input file not found: {args.input_path}")
        sys.exit(1)

    # Load the point cloud
    print(f"Loading point cloud from: {args.input_path}")
    try:
        pcd_input = o3d.io.read_point_cloud(args.input_path)
        if not pcd_input.has_points():
            print(f"Error: Point cloud file is empty: {args.input_path}")
            sys.exit(1)
    except Exception as e:
        print(f"Error loading point cloud: {e}")
        sys.exit(1)


    # --- Initialize the class with all parameters ---
    segmenter = PointCloudSegmentation(
        voxel_size=args.voxel_size,
        ransac_distance_threshold=args.ransac_dist,
        ransac_n=args.ransac_n,
        ransac_num_iterations=args.ransac_iter,
        max_planes=args.max_planes,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_points=args.dbscan_min_pts,
        refine_clusters=args.refine,
        refine_dbscan_eps_factor=args.refine_eps_factor,
        refine_dbscan_min_points=args.refine_min_pts
    )

    # Preprocess the loaded point cloud
    segmenter.preprocess(pcd_input)

    # --- Perform selected segmentation method ---
    if args.method == 'ransac_dbscan':
        segmenter.segment_ransac_dbscan()
    elif args.method == 'dbscan_only':
        segmenter.segment_dbscan()
    else:
        print(f"Error: Unknown method '{args.method}'")
        sys.exit(1)
    # --- End method selection ---


    # Get the resulting point cloud
    result_pcd = segmenter.get_result()


    # Handle the result (save and/or visualize)
    if result_pcd and result_pcd.has_points():
        print(f"\nProcessing complete. Result has {len(np.asarray(result_pcd.points))} points.")

        # Determine output path if not provided
        output_path = args.output_path
        if output_path is None:
             base, ext = os.path.splitext(args.input_path)
             # Include method in default output name
             output_path = f"{base}_{args.method}_segmented{ext}"

        # Save the result
        print(f"Saving segmented point cloud to: {output_path}")
        try:
            o3d.io.write_point_cloud(output_path, result_pcd, write_ascii=False) # Binary often smaller/faster
            print("Save successful.")
        except Exception as e:
            print(f"Error saving point cloud: {e}")
            # Don't exit, might still want to visualize

        # Visualize the result if requested
        if args.visualize:
            print("Visualizing point cloud...")
            o3d.visualization.draw_geometries([result_pcd], window_name=f"{args.method.capitalize()} Segmentation Result")

    else:
        print("\nProcessing resulted in an empty point cloud. Nothing to save or visualize.")


if __name__ == "__main__":
    main()