import open3d as o3d
import numpy as np
import time # To measure performance
import argparse # For command-line arguments
from plyfile import PlyData, PlyElement # For saving PLY
import os # For path checking

def detect_edges_local_normals(pcd, normal_radius=0.1, normal_max_nn=30,
                               neighbor_radius=0.1, angle_threshold_deg=30.0):
    """
    Detects edges in a point cloud based on local normal variations.

    Edge points are colored red, others gray.

    Args:
        pcd (o3d.geometry.PointCloud): Input point cloud.
        normal_radius (float): Radius for normal estimation search.
        normal_max_nn (int): Max neighbors for normal estimation.
        neighbor_radius (float): Radius to find neighbors for edge detection comparison.
        angle_threshold_deg (float): Angle threshold in degrees. If the maximum angle
                                     between a point's normal and any neighbor's normal
                                     exceeds this, it's marked as an edge.

    Returns:
        tuple: (o3d.geometry.PointCloud, np.ndarray)
            - Point cloud with edge points colored red.
            - NumPy array containing the coordinates of the detected edge points (Nx3).
    """
    start_time = time.time()

    if not pcd.has_points():
        print("Error: Input point cloud has no points.")
        return pcd, np.empty((0, 3))

    print(f"Input point cloud has {len(pcd.points)} points.")

    # 1. Estimate Normals
    print("Estimating normals...")
    normal_start_time = time.time()
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))
    # Optional: Orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k=normal_max_nn // 2) # Use a subset of neighbors for orientation
    print(f"Normals estimated in {time.time() - normal_start_time:.2f} seconds.")

    if not pcd.has_normals():
        print("Error: Normals could not be estimated. Check parameters and point cloud density.")
        # Return original cloud or an empty one
        pcd.paint_uniform_color([0.5, 0.5, 0.5]) # Color gray to indicate failure
        return pcd, np.empty((0, 3))

    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    num_points = len(points)
    colors = np.full((num_points, 3), [0.8, 0.8, 0.8]) # Default color: gray
    is_edge = np.zeros(num_points, dtype=bool)

    # 2. Build KDTree for efficient neighbor search
    print("Building KDTree for neighbor search...")
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    print("KDTree built.")

    angle_threshold_rad = np.deg2rad(angle_threshold_deg)

    # 3. Iterate through points and check local normal variation
    print(f"Detecting edges with angle threshold {angle_threshold_deg} degrees...")
    detection_start_time = time.time()
    edge_count = 0
    for i in range(num_points):
        point_i = points[i]
        normal_i = normals[i]

        # Ensure normal_i is valid (not zero vector)
        if np.linalg.norm(normal_i) < 1e-6:
            continue

        # 3a. Find neighbors within neighbor_radius
        [k, idx, _] = pcd_tree.search_radius_vector_3d(point_i, neighbor_radius)

        if k < 2: # Need at least one neighbor besides potentially itself
            continue

        max_angle = 0.0
        # 3b. Calculate max angle difference with neighbors
        for j_idx in idx:
            if i == j_idx: # Don't compare point to itself
                continue

            normal_j = normals[j_idx]

            # Ensure normal_j is valid
            if np.linalg.norm(normal_j) < 1e-6:
                continue

            # Calculate dot product (cosine of angle)
            dot_product = np.dot(normal_i, normal_j)
            # Clamp dot product to avoid potential domain errors with arccos
            # due to floating point inaccuracies near +/- 1.0
            dot_product = np.clip(dot_product, -1.0, 1.0)

            angle = np.arccos(dot_product)
            max_angle = max(max_angle, angle)

        # 3c. Thresholding
        if max_angle > angle_threshold_rad:
            is_edge[i] = True
            edge_count += 1

        # Optional: Progress indicator
        if (i + 1) % (num_points // 10) == 0:
             print(f"  Processed {i+1}/{num_points} points...")


    print(f"Edge detection loop finished in {time.time() - detection_start_time:.2f} seconds.")

    # 4. Color the edge points red and collect edge coordinates
    edge_indices = np.where(is_edge)[0]
    colors[edge_indices] = [1.0, 0.0, 0.0] # Red color
    edge_points_coords = points[edge_indices] # Get coordinates of edge points

    # Create a new point cloud with colors
    pcd_colored = o3d.geometry.PointCloud()
    pcd_colored.points = o3d.utility.Vector3dVector(points)
    pcd_colored.colors = o3d.utility.Vector3dVector(colors)
    # Optionally copy normals if needed for further processing
    if pcd.has_normals():
         pcd_colored.normals = o3d.utility.Vector3dVector(normals) # Copy normals too

    print(f"Detected {edge_count} edge points ({edge_count / num_points * 100:.2f}%).")
    print(f"Total processing time: {time.time() - start_time:.2f} seconds.")

    return pcd_colored, edge_points_coords # Return colored cloud AND edge coordinates

# --- Function to save results in PLY format ---
def save_ply_with_edges(output_file, pcd_processed, edge_points):
    """
    Saves the processed point cloud and detected edge points to a PLY file.

    Args:
        output_file (str): Path to the output PLY file.
        pcd_processed (o3d.geometry.PointCloud): Point cloud with points colored
                                                 (e.g., edges red, others gray).
                                                 Should have points, colors, and optionally normals.
        edge_points (np.ndarray): Nx3 array of coordinates for detected edge points.
    """
    print(f"Saving results to {output_file}...")
    vertices = np.asarray(pcd_processed.points)
    colors = (np.asarray(pcd_processed.colors) * 255).astype(np.uint8)
    has_normals = pcd_processed.has_normals()
    if has_normals:
        normals = np.asarray(pcd_processed.normals)

    # Create vertex data array
    vertex_data = []
    if has_normals:
        vertex_dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')
        ]
        for v, c, n in zip(vertices, colors, normals):
            vertex_data.append(tuple(v) + tuple(c) + tuple(n))
    else:
        vertex_dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
        ]
        for v, c in zip(vertices, colors):
            vertex_data.append(tuple(v) + tuple(c))

    vertex_element = PlyElement.describe(np.array(vertex_data, dtype=vertex_dtype), 'vertex')

    # Create edge point data array
    edge_data = []
    edge_dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    for ep in edge_points:
        edge_data.append(tuple(ep))

    # Only add edge element if edges were found
    elements = [vertex_element]
    if len(edge_data) > 0:
        edge_element = PlyElement.describe(np.array(edge_data, dtype=edge_dtype), 'detected_edge_point')
        elements.append(edge_element)
        print(f"Including {len(edge_data)} edge points in the 'detected_edge_point' element.")
    else:
        print("No edge points detected to save in a separate element.")

    # Write PLY file
    PlyData(elements, text=True).write(output_file)
    print("Save complete.")


# --- Main execution ---
if __name__ == "__main__":
    # --- Argument Parser ---
    parser = argparse.ArgumentParser(description="Detect edges in a point cloud based on local normal variations and save results.")
    parser.add_argument("input_file", type=str, help="Path to the input point cloud file (.ply, .pcd, .xyz, etc.)")
    parser.add_argument("output_file", type=str, help="Path to save the output .ply file with detected edges.")
    parser.add_argument("--voxel_size", type=float, default=0.02, help="Voxel size for downsampling (default: 0.02). Set to 0 to disable.")
    parser.add_argument("--sor_neighbors", type=int, default=20, help="Number of neighbors for statistical outlier removal (default: 20). Set to 0 to disable.")
    parser.add_argument("--sor_std_ratio", type=float, default=2.0, help="Standard deviation ratio for SOR (default: 2.0).")
    parser.add_argument("--normal_radius", type=float, default=0.05, help="Radius for normal estimation (default: 0.05).")
    parser.add_argument("--normal_nn", type=int, default=30, help="Max neighbors for normal estimation (default: 30).")
    parser.add_argument("--edge_radius", type=float, default=0.05, help="Radius for edge detection neighbor search (default: 0.05).")
    parser.add_argument("--edge_angle", type=float, default=45.0, help="Angle threshold in degrees for edge detection (default: 45.0).")
    parser.add_argument("--visualize", action="store_true", help="Show visualization of the result.")

    args = parser.parse_args()

    # --- Parameters ---
    INPUT_FILENAME = args.input_file
    OUTPUT_FILENAME = args.output_file
    VOXEL_SIZE = args.voxel_size
    SOR_NB_NEIGHBORS = args.sor_neighbors
    SOR_STD_RATIO = args.sor_std_ratio
    NORMAL_ESTIMATION_RADIUS = args.normal_radius
    NORMAL_ESTIMATION_MAX_NN = args.normal_nn
    EDGE_NEIGHBOR_RADIUS = args.edge_radius
    ANGLE_THRESHOLD_DEGREES = args.edge_angle
    # --- End Parameters ---

    # Load your point cloud
    print(f"Loading point cloud from: {INPUT_FILENAME}")
    if not os.path.exists(INPUT_FILENAME):
        print(f"Error: Input file not found: {INPUT_FILENAME}")
        exit() # Use exit() instead of sys.exit() if sys is not imported

    try:
        pcd = o3d.io.read_point_cloud(INPUT_FILENAME)
        if not pcd.has_points():
             print(f"Error: Failed to load point cloud from {INPUT_FILENAME} or it's empty.")
             # Create a dummy point cloud for testing if loading fails
             print("Creating a dummy cube point cloud for demonstration.")
             pcd = o3d.geometry.TriangleMesh.create_box().sample_points_poisson_disk(number_of_points=5000)
             # Give the dummy cloud some base color if it doesn't have one
             if not pcd.has_colors():
                 pcd.paint_uniform_color([0.6, 0.6, 0.9]) # Light blue-ish gray
        else:
            print(f"Successfully loaded point cloud with {len(pcd.points)} points.")
            # We will estimate normals *after* preprocessing

    except Exception as e:
        print(f"Error loading or processing point cloud file: {e}")
        print("Please ensure the file exists and is a supported format.")
        print("Creating a dummy cube point cloud for demonstration.")
        pcd = o3d.geometry.TriangleMesh.create_box().sample_points_poisson_disk(number_of_points=5000)
        if not pcd.has_colors():
            pcd.paint_uniform_color([0.6, 0.6, 0.9])

    # --- Preprocessing Steps ---
    print("\n--- Preprocessing ---")
    pcd_processed = pcd # Start with the original pcd

    # 1. Voxel Downsampling (Optional)
    if VOXEL_SIZE > 0:
        print(f"Applying Voxel Downsampling with voxel size: {VOXEL_SIZE}")
        pcd_downsampled = pcd_processed.voxel_down_sample(voxel_size=VOXEL_SIZE)
        print(f"Point cloud downsampled from {len(pcd_processed.points)} to {len(pcd_downsampled.points)} points.")
        pcd_processed = pcd_downsampled
    else:
        print("Skipping Voxel Downsampling.")

    # 2. Statistical Outlier Removal (Optional)
    if SOR_NB_NEIGHBORS > 0:
        print(f"Applying Statistical Outlier Removal (nb_neighbors={SOR_NB_NEIGHBORS}, std_ratio={SOR_STD_RATIO})")
        pcd_processed_sor, ind = pcd_processed.remove_statistical_outlier(
            nb_neighbors=SOR_NB_NEIGHBORS,
            std_ratio=SOR_STD_RATIO
        )
        num_removed = len(pcd_processed.points) - len(pcd_processed_sor.points)
        print(f"Removed {num_removed} outlier points.")
        pcd_processed = pcd_processed_sor
    else:
        print("Skipping Statistical Outlier Removal.")

    print(f"Processed point cloud has {len(pcd_processed.points)} points.")
    print("--- Preprocessing Complete ---\n")
    # --- End Preprocessing ---


    # Detect edges using the *processed* point cloud
    # Note: Normals are estimated inside the detect_edges function on the processed cloud
    pcd_edges_colored, edge_points_coords = detect_edges_local_normals( # Capture both outputs
        pcd_processed, # Use the preprocessed cloud
        normal_radius=NORMAL_ESTIMATION_RADIUS,
        normal_max_nn=NORMAL_ESTIMATION_MAX_NN,
        neighbor_radius=EDGE_NEIGHBOR_RADIUS,
        angle_threshold_deg=ANGLE_THRESHOLD_DEGREES
    )

    # --- Save the results ---
    save_ply_with_edges(OUTPUT_FILENAME, pcd_edges_colored, edge_points_coords)

    # --- Optional Visualization ---
    if args.visualize:
        print("Visualizing point cloud with detected edges (Red)...")
        o3d.visualization.draw_geometries([pcd_edges_colored], window_name="Edge Detection using Local Normals (Preprocessed)")